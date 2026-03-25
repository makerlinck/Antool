"""评估引擎模块

基于 DeepDanbooru 的图像标签评估系统。
提供从图像输入到标签输出的完整处理流程。

设计目标：
1. 单一入口：`evaluate()` 函数
2. 独立运行：不依赖外部服务或复杂配置
3. 性能监控：内置计时和指标收集
4. 可配置：支持阈值、并行度等参数

使用示例：
    from infrastructure.evaluations import evaluate

    results = evaluate(
        images=[(uid, path, image_array), ...],
        model_path="resources/models/v3-20211112-sgd-e28",
        threshold=0.5,
    )
"""

from .processor import ImageEvaluationProcessor, create_processor
from .scheduler import BatchScheduler

__all__ = ["ImageEvaluationProcessor", "create_processor", "BatchScheduler"]
