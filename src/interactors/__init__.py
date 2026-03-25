"""用例层

Clean Architecture 用例层，包含业务逻辑。
依赖 Entities 和 Interfaces，不依赖具体实现。
"""

from .evaluate_image import ImageEvaluationInteractor

__all__ = ["ImageEvaluationInteractor"]
