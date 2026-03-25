from pathlib import Path

from .common import CommonConfig
from .logger import LoggerConfig, LogLevel
from .performance import PerformanceConfig
from .web_app import WebAppConfig


class Config(
    CommonConfig,
    LoggerConfig,
    PerformanceConfig,
    WebAppConfig,
):
    """统一配置聚合类"""

    def __init__(
        self,
        path_root: Path | str | None = None,
        *,
        # Logger
        log_file_dir: str = "logs",
        metrics_file_dir: str = "logs",
        min_log_level: LogLevel = LogLevel.INFO,
        # Performance
        max_concurrent: int = 4,
        max_tasks: int = 100,
        xla_boost_enable_auto: bool = True,
        batch_thread_threshold: int = 8,
        batch_size: int = 32,
        intra_op_parallelism: int = 4,
        inter_op_parallelism: int = 4,
        # WebApp
        app_name: str = "Antool API",
        app_version: str = "0.1.1",
        app_description: str = "图片分类归档服务程序",
    ) -> None:
        CommonConfig.__init__(self, path_root)
        LoggerConfig.__init__(self, log_file_dir, metrics_file_dir, min_log_level)
        PerformanceConfig.__init__(
            self,
            max_concurrent,
            max_tasks,
            xla_boost_enable_auto,
            batch_thread_threshold,
            batch_size,
            intra_op_parallelism,
            inter_op_parallelism,
        )
        WebAppConfig.__init__(self, app_name, app_version, app_description)

    def get_config(self, attr: str):
        """获取指定配置项"""
        return getattr(self, attr, None)

    def get_all_configs(self) -> dict:
        """获取所有配置项"""
        return {key: getattr(self, key) for key in dir(self) if not key.startswith("_")}

    def init_logging(self) -> "logging.Logger":
        """初始化日志系统

        Returns:
            配置好的日志记录器
        """
        from infrastructure.logging import Logger

        logger = Logger(
            name=self.app_name.lower().replace(" ", "_"),
            min_level=self.min_log_level,
            log_file_dir=self.log_file_dir,
        )
        return logger.setup()


__all__ = [
    "Config",
    "CommonConfig",
    "LoggerConfig",
    "PerformanceConfig",
    "WebAppConfig",
    "LogLevel",
]
