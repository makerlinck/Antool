"""图像评估处理器

性能监控点：
1. 预处理耗时
2. 模型推理耗时
3. 后处理耗时
"""

import logging
import time
from typing import Optional

import numpy as np

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor

logger = logging.getLogger(__name__)


class ModelGateway:
    """模型网关"""

    def __init__(self, model):
        self._model = model

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._model.input_shape

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self._model.predict(batch, verbose=0)


class ImageEvaluationProcessor(ImageProcessor):
    """图像评估处理器

    职责：预处理 → 推理 → 后处理
    内置性能监控，各阶段耗时通过 logger 输出。
    """

    def __init__(
        self,
        model: ModelGateway,
        tags: list[str],
        threshold: float = 0.5,
    ):
        self._model = model
        self._tags = tags
        self.threshold = threshold

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        """批量处理图像任务

        性能监控：预处理 → 推理 → 后处理
        """
        if not tasks:
            return []

        t0 = time.perf_counter()

        # 1. 预处理
        t_pre = time.perf_counter()
        target_size = (self._model.input_shape[1], self._model.input_shape[2])
        valid_tasks, preprocessed = self._preprocess_batch(tasks, target_size)
        pre_time = (time.perf_counter() - t_pre) * 1000
        logger.debug(f"[processor] Preprocess: {pre_time:.1f}ms for {len(tasks)} images, valid={len(valid_tasks)}")

        if len(preprocessed) == 0:
            return []

        # 2. 推理
        t_infer = time.perf_counter()
        scores = self._model.predict(preprocessed)
        infer_time = (time.perf_counter() - t_infer) * 1000
        logger.info(f"[processor] Inference: {infer_time:.1f}ms for {len(preprocessed)} images, shape={preprocessed.shape}")

        # 3. 后处理
        t_post = time.perf_counter()
        results = self._postprocess_batch(scores, valid_tasks)
        post_time = (time.perf_counter() - t_post) * 1000
        logger.debug(f"[processor] Postprocess: {post_time:.1f}ms for {len(results)} results")

        total_time = (time.perf_counter() - t0) * 1000
        logger.debug(f"[processor] Total: {total_time:.1f}ms, per-image: {total_time/len(tasks):.1f}ms")

        return results

    def _preprocess_batch(
        self,
        tasks: list[ImageTask],
        target_size: tuple[int, int],
    ) -> tuple[list[ImageTask], np.ndarray]:
        """批量预处理（串行，numpy 向量化）

        Returns:
            (有效的任务列表, 预处理的图像数组)
        """
        from infrastructure.evaluations.preprocess import preprocess_image

        valid_tasks = []
        processed = []

        for task in tasks:
            if task.image is None:
                continue
            result = preprocess_image(task.image, target_size)
            if result is not None:
                valid_tasks.append(task)
                processed.append(result)

        if not processed:
            return [], np.array([])

        return valid_tasks, np.stack(processed)

    def _postprocess_batch(
        self,
        scores: np.ndarray,
        tasks: list[ImageTask],
    ) -> list[EvaluationResult]:
        """批量后处理"""
        results = []
        for i, task in enumerate(tasks):
            result = self._postprocess_one(task, scores[i])
            results.append(result)
        return results

    def _postprocess_one(self, task: ImageTask, scores: np.ndarray) -> EvaluationResult:
        """单张图像后处理"""
        from infrastructure.evaluations.filter import filter_tags, weighted_result

        # squeeze
        if scores.ndim == 2:
            scores = scores.squeeze(0)

        # 过滤和加权
        raw = filter_tags(scores, self._tags, self.threshold)
        result = weighted_result(raw, self._tags, self._tags)

        return EvaluationResult.from_raw(
            uid=task.uid,
            rating=result.rating,
            tags=result.tags,
            path=task.path,
        )


# =============================================================================
# 工厂函数（用于进程池中重新初始化）
# =============================================================================

_processor: Optional[ImageEvaluationProcessor] = None


def create_processor() -> ImageEvaluationProcessor:
    """创建处理器实例（用于进程池或首次创建）"""
    from configs import Config
    from infrastructure.evaluations.model_loader import SharedModelLoader

    config = Config()
    loader = SharedModelLoader(config.model_path / "v3-20211112-sgd-e28")

    if loader._model is None:
        loader.load()

    gateway = ModelGateway(loader.model)

    return ImageEvaluationProcessor(
        model=gateway,
        tags=loader.tags,
        threshold=0.5,
    )


def get_processor() -> ImageEvaluationProcessor:
    """获取全局处理器实例（主进程使用）"""
    global _processor
    if _processor is None:
        _processor = create_processor()
    return _processor
