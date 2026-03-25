import enum
import logging


class LogLevel(enum.Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @property
    def level(self) -> int:
        return self.value


class LoggerConfig:
    """日志配置"""

    log_file_dir: str = "logs"
    metrics_file_dir: str = "logs"
    min_log_level: LogLevel = LogLevel.INFO

    def __init__(
        self,
        log_file_dir: str = "logs",
        metrics_file_dir: str = "logs",
        min_log_level: LogLevel = LogLevel.INFO,
    ) -> None:
        self.log_file_dir = log_file_dir
        self.metrics_file_dir = metrics_file_dir
        self.min_log_level = min_log_level
