"""任务调度器

性能监控点：
1. 任务分批策略
2. 进程池创建开销
3. 各批次处理时间
4. 结果收集时间
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor

logger = logging.getLogger(__name__)


class BatchScheduler:
    """智能批量调度器

    策略：
    - 小批量（< thread_threshold）：直接处理（无进程开销）
    - 大批量（>= thread_threshold）：进程池并行（绕过 GIL）

    性能监控：分批数量、批次耗时、总耗时
    """

    def __init__(
        self,
        thread_threshold: int = 32,
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

        Returns:
            评估结果列表（按输入顺序）
        """
        if not tasks:
            return []

        t0 = time.perf_counter()
        logger.info(
            f"[scheduler] submit: {len(tasks)} tasks, threshold={self.thread_threshold}"
        )

        if len(tasks) < self.thread_threshold:
            # 小批量：直接处理
            t_direct = time.perf_counter()
            results = processor.process(tasks)
            elapsed = (time.perf_counter() - t_direct) * 1000
            logger.info(f"[scheduler] Direct: {elapsed:.1f}ms for {len(tasks)} tasks")
            return results
        else:
            # 大批量：进程池
            return self._submit_to_process_pool(tasks, processor)

    def _submit_to_process_pool(
        self,
        tasks: list[ImageTask],
        _processor: ImageProcessor,
    ) -> list[EvaluationResult]:
        """进程池处理"""
        t0 = time.perf_counter()

        # 智能分批：让 batch 数量接近 worker 数量
        num_batches = min(self.max_workers, len(tasks))
        batch_size = max(1, len(tasks) // num_batches)
        batches = [tasks[i : i + batch_size] for i in range(0, len(tasks), batch_size)]

        logger.info(
            f"[scheduler] ProcessPool: {len(tasks)} tasks -> {num_batches} batches, "
            f"batch_size={batch_size}, workers={self.max_workers}"
        )

        # 收集结果
        uid_to_result: dict[str, EvaluationResult] = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_process_batch_wrapper, batch) for batch in batches]

            for i, future in enumerate(futures):
                batch_results = future.result()
                batch_time = time.perf_counter() - t0
                logger.info(
                    f"[scheduler] Batch {i+1}/{num_batches}: {len(batch_results)} results, "
                    f"elapsed={batch_time*1000:.0f}ms"
                )
                for result in batch_results:
                    uid_to_result[result.uid] = result

        total_time = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[scheduler] ProcessPool done: {total_time:.1f}ms total, {total_time/len(tasks):.1f}ms/image"
        )

        # 按原始顺序返回
        return [uid_to_result[task.uid] for task in tasks]


def _process_batch_wrapper(tasks: list[ImageTask]) -> list[EvaluationResult]:
    """进程池处理包装函数

    注意：进程池中需要重新初始化处理器。
    """
    from infrastructure.evaluations.processor import create_processor

    processor = create_processor()
    return processor.process(tasks)
