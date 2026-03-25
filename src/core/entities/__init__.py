"""核心实体层

Clean Architecture 最内层，无任何外部依赖。
包含业务实体和值对象。
"""
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


@dataclass(frozen=True, slots=True)
class Tag:
    """标签值对象"""
    name: str
    score: float

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")


@dataclass(frozen=True, slots=True)
class Rating:
    """评级值对象"""
    label: str
    score: float

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")


@dataclass(frozen=True, slots=True)
class ImageTask:
    """图像任务实体

    包装图像数据，附带唯一标识。
    用于追踪处理流程中的图像。
    """
    image: np.ndarray | None
    uid: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_image(cls, image: np.ndarray, uid: str | None = None) -> "ImageTask":
        """从图像创建任务"""
        return cls(image=image, uid=uid or uuid4().hex)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """评估结果实体

    包含原始任务 UID，支持乱序返回后的结果匹配。
    """
    uid: str              # 对应 ImageTask 的 UID
    rating: Rating
    tags: tuple[Tag, ...]

    @classmethod
    def from_raw(
        cls,
        uid: str,
        rating: tuple[str, float],
        tags: list[tuple[str, float]],
    ) -> "EvaluationResult":
        """从原始数据创建"""
        return cls(
            uid=uid,
            rating=Rating(label=rating[0], score=rating[1]),
            tags=tuple(Tag(name=n, score=s) for n, s in tags),
        )

    def to_raw(self) -> tuple[str, tuple[str, float], list[tuple[str, float]]]:
        """转换为原始数据"""
        return (
            self.uid,
            (self.rating.label, self.rating.score),
            [(t.name, t.score) for t in self.tags],
        )

    @property
    def tag_names(self) -> list[str]:
        """获取标签名列表"""
        return [t.name for t in self.tags]
