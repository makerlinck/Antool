"""任务调度器

根据批量大小智能选择执行策略：
- 小批量：直接处理（低开销）
- 大批量：进程池（绕过 GIL）

使用 UUID 追踪任务，结果与输入通过 UID 关联。
支持任务取消和指标收集。
"""
from concurrent.futures import ProcessPoolExecutor
import logging
import time

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor
from infrastructure.cancel import CancellationManager, CancelledError, get_cancellation
from infrastructure.metrics import get_metrics

logger = logging.getLogger(__name__)


class BatchScheduler:
    """智能批量调度器

    通过 UID 关联输入输出，不依赖顺序。
    支持任务取消和指标收集。
    """

    def __init__(
        self,
        thread_threshold: int = 32,  # 提高阈值，小批量直接处理
        max_workers: int = 4,
        batch_size: int = 32,
    ):
        self.thread_threshold = thread_threshold
        self.max_workers = max_workers
        self.batch_size = batch_size

    def submit(
        self,
        tasks: list[ImageTask],
        processor: ImageProcessor,
    ) -> list[EvaluationResult]:
        """提交任务

        Args:
            tasks: 图像任务列表（带 UID）
            processor: 处理器

        Returns:
            评估结果列表（UID 与输入对应）

        Raises:
            CancelledError: 如果任务被取消
        """
        # 检查取消状态
        cancellation = get_cancellation()
        if cancellation.is_cancelled:
            raise CancelledError(cancellation.reason or cancellation._reason)

        if not tasks:
            return []

        if len(tasks) < self.thread_threshold:
            # 小批量：直接处理
            return processor.process(tasks)
        else:
            # 大批量：进程池处理
            return self._submit_to_process_pool(tasks, processor)

    def _submit_to_process_pool(
        self,
        tasks: list[ImageTask],
        _processor: ImageProcessor,  # noqa: ARG002 - 进程池中重新创建处理器
    ) -> list[EvaluationResult]:
        """提交到进程池

        分批次并行处理，结果通过 UID 关联。
        支持取消检查。
        """
        t0 = time.perf_counter()
        logger.info(f"[scheduler] Submitting {len(tasks)} tasks to process pool")

        # 检查取消状态
        cancellation = get_cancellation()
        if cancellation.is_cancelled:
            raise CancelledError(cancellation.reason or cancellation._reason)

        # 1. 智能分批 - 让 batch 数量等于 worker 数量，最大化并行度
        num_batches = min(self.max_workers, len(tasks))
        batch_size = max(1, len(tasks) // num_batches)
        batches = [
            tasks[i:i + batch_size]
            for i in range(0, len(tasks), batch_size)
        ]

        # 2. 构建所有 UID 到预期结果的映射
        uid_to_result: dict[str, EvaluationResult] = {}

        # 3. 提交所有任务
        logger.info(f"[scheduler] Creating ProcessPoolExecutor with {self.max_workers} workers, {len(batches)} batches (batch_size={batch_size})")
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(_process_batch_wrapper, batch)
                for batch in batches
            ]

            # 4. 收集结果，按 UID 存储
            for i, future in enumerate(futures):
                # 检查取消状态
                if cancellation.is_cancelled:
                    logger.warning(f"[scheduler] Cancelled while waiting for batch {i+1}/{len(futures)}")
                    for f in futures:
                        f.cancel()
                    raise CancelledError(cancellation.reason or cancellation._reason)

                logger.info(f"[scheduler] Waiting for batch {i+1}/{len(futures)} result...")
                batch_results = future.result()
                logger.info(f"[scheduler] Batch {i+1} done in {time.perf_counter()-t0:.3f}s, got {len(batch_results)} results")
                for result in batch_results:
                    uid_to_result[result.uid] = result

        logger.info(f"[scheduler] All batches done in {time.perf_counter()-t0:.3f}s")

        # 5. 按原始输入顺序返回结果
        return [uid_to_result[task.uid] for task in tasks]


def _process_batch_wrapper(tasks: list[ImageTask]) -> list[EvaluationResult]:
    """进程池处理包装函数

    注意：进程池中需要重新初始化处理器，
    因为处理器包含无法 pickle 的 TensorFlow 模型。
    """
    from infrastructure.evaluations.processor import create_processor

    processor = create_processor()
    return processor.process(tasks)
