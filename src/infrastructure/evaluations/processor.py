"""图像处理器

负责图像的预处理、推理和后处理。
"""

from typing import Optional

import numpy as np

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor


class ModelGateway:
    """模型网关（简化版协议实现）"""

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
    输入输出都带 UID，保证结果可追踪。
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

        Args:
            tasks: 图像任务列表（带 UID）

        Returns:
            评估结果列表（带对应 UID）
        """
        if not tasks:
            return []

        # 提取 UID 和图像
        uids = [t.uid for t in tasks]
        images = [t.image for t in tasks]

        # 1. 预处理
        target_size = (self._model.input_shape[1], self._model.input_shape[2])
        preprocessed, valid_uids = self._preprocess_batch(images, uids, target_size)

        if len(preprocessed) == 0:
            return []

        # 2. 推理
        scores = self._model.predict(preprocessed)

        # 3. 后处理（带 UID）
        return self._postprocess_batch(scores, valid_uids)

    def _preprocess_batch(
        self,
        images: list[np.ndarray],
        uids: list[str],
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, list[str]]:
        """批量预处理

        Returns:
            (预处理的图像数组, 有效的 UID 列表)
        """
        from concurrent.futures import ThreadPoolExecutor
        from infrastructure.evaluations.preprocess import preprocess_image

        # 并行预处理
        def process_one(args):
            uid, img = args
            result = preprocess_image(img, target_size)
            return uid, result

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_one, zip(uids, images)))

        processed = []
        valid_uids = []
        for uid, result in results:
            if result is not None:
                processed.append(result)
                valid_uids.append(uid)

        if not processed:
            return np.array([]), []

        return np.stack(processed), valid_uids

    def _postprocess_batch(
        self,
        scores: np.ndarray,
        uids: list[str],
    ) -> list[EvaluationResult]:
        """批量后处理"""
        results = []
        for i, uid in enumerate(uids):
            result = self._postprocess_one(uid, scores[i])
            results.append(result)
        return results

    def _postprocess_one(self, uid: str, scores: np.ndarray) -> EvaluationResult:
        """单张图像后处理"""
        from infrastructure.evaluations.filter import filter_tags, weighted_result

        # squeeze 如果需要
        if scores.ndim == 2:
            scores = scores.squeeze(0)

        # 过滤和加权
        raw = filter_tags(scores, self._tags, self.threshold)
        result = weighted_result(raw, self._tags, self._tags)

        return EvaluationResult.from_raw(
            uid=uid,
            rating=result.rating,
            tags=result.tags,
        )


# 全局处理器实例（用于进程池）
_processor: Optional[ImageEvaluationProcessor] = None


def create_processor() -> ImageEvaluationProcessor:
    """创建处理器实例

    用于进程池中重新初始化处理器。
    """
    from configs import Config
    from infrastructure.evaluations.model_loader import SharedModelLoader

    config = Config()
    loader = SharedModelLoader(config.model_path / "v3-20211112-sgd-e28")

    # 如果未加载，则加载
    if loader._model is None:
        loader.load()

    gateway = ModelGateway(loader.model)

    return ImageEvaluationProcessor(
        model=gateway,
        tags=loader.tags,
        threshold=0.5,
    )


def get_processor() -> ImageEvaluationProcessor:
    """获取全局处理器实例"""
    global _processor
    if _processor is None:
        _processor = create_processor()
    return _processor
