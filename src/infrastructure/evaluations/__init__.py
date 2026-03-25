"""评估引擎模块

基于 DeepDanbooru (https://github.com/KichangKim/DeepDanbooru) 的图像标签评估系统，使用 tf.data 流水线优化性能。

架构：
- ImageEvaluationProcessor: tf.data 预处理 + 批量推理 + 后处理
- BatchScheduler: 智能调度（小批量直接处理，大批量进程池）
- SharedModelLoader: 模型单例加载，XLA JIT 编译

使用示例：
    from infrastructure.evaluations import create_processor, BatchScheduler

    processor = create_processor(batch_size=16)
    scheduler = BatchScheduler(thread_threshold=1000)

    results = scheduler.submit(tasks, processor)
"""

from .processor import ImageEvaluationProcessor, create_processor
from .processor_cpu import CPUOptimizedProcessor, create_cpu_processor
from .scheduler import BatchScheduler

__all__ = [
    "ImageEvaluationProcessor",
    "create_processor",
    "CPUOptimizedProcessor",
    "create_cpu_processor",
    "BatchScheduler",
]
