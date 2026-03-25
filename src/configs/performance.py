"""性能配置"""
from dataclasses import dataclass


@dataclass
class PerformanceConfig:
    """性能配置"""

    # 并发控制
    max_concurrent: int = 4
    max_tasks: int = 100

    # 批量调度
    batch_thread_threshold: int = 1000  # ThreadPoolExecutor 无需重载模型，用大阈值禁用 ProcessPool
    batch_size: int = 32             # 每批处理数量

    # TensorFlow 优化
    xla_boost_enable_auto: bool = True
    # Intel OneDNN
    tf_enable_onednn_opts: bool = True
    intra_op_parallelism: int = 4    # TF 操作内并行度
    inter_op_parallelism: int = 4    # TF 操作间并行度

    def __init__(
        self,
        max_concurrent: int = 4,
        max_tasks: int = 100,
        xla_boost_enable_auto: bool = True,
        batch_thread_threshold: int = 1000,
        batch_size: int = 32,
        intra_op_parallelism: int = 4,
        inter_op_parallelism: int = 4,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.max_tasks = max_tasks
        self.xla_boost_enable_auto = xla_boost_enable_auto
        self.batch_thread_threshold = batch_thread_threshold
        self.batch_size = batch_size
        self.intra_op_parallelism = intra_op_parallelism
        self.inter_op_parallelism = inter_op_parallelism
