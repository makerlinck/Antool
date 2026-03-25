"""评估相关接口定义

接口适配层，定义 Use Case 与基础设施之间的边界。
"""

from typing import Protocol

from core.entities import ImageTask, EvaluationResult


class ImageProcessor(Protocol):
    """图像处理器接口

    定义图像处理流程的抽象接口。
    输入输出都带 UID，保证结果可追踪。
    """

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        """处理图像任务列表

        Args:
            tasks: 图像任务列表（每个任务带 UID）

        Returns:
            评估结果列表（每个结果带对应 UID，顺序可与输入不同）
        """
        ...


class TaskScheduler(Protocol):
    """任务调度器接口

    定义任务调度的抽象接口。
    """

    def submit(
        self,
        tasks: list[ImageTask],
        processor: ImageProcessor,
    ) -> list[EvaluationResult]:
        """提交任务

        Args:
            tasks: 图像任务列表
            processor: 处理器

        Returns:
            评估结果列表（UID 与输入对应）
        """
        ...
