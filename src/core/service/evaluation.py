"""图像评估领域服务

纯粹��标签评估业务逻辑，不依赖具体基础设施。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class EvaluationResult:
    """评估结果"""

    rating: tuple[str, float]
    tags: list[tuple[str, float]]


class EvaluationModel(Protocol):
    """评估模型协议"""

    def predict(self, image: np.ndarray) -> np.ndarray: ...


class EvaluationService:
    """图像评估领域服务

    核心业务：图像 → 标签评估结果
    """

    def __init__(
        self,
        model: EvaluationModel,
        tags: list[str],
        threshold: float = 0.5,
    ):
        self.model = model
        self.tags = tags
        self.threshold = threshold

    def evaluate(self, image: np.ndarray) -> EvaluationResult | None:
        """评估单张图像"""
        from infrastructure.evaluations import evaluate_image

        result = evaluate_image(
            image_input=image,
            model=self.model,
            lang_tags=self.tags,
            zero_tags=self.tags,
            threshold=self.threshold,
        )

        if result is None:
            return None

        rating, tags = result
        return EvaluationResult(rating=rating, tags=tags)
