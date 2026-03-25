"""日志系统

基于标准库 logging，提供统一的日志配置。
日志文件按日期命名：YYYY-MM-DD.log
"""
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from configs.logger import LogLevel


class Logger:
    """日志管理器

    统一管理应用日志，支持控制台和文件输出。
    """

    _root_logger: Optional[logging.Logger] = None

    def __init__(
        self,
        name: str = "antool",
        min_level: LogLevel = LogLevel.INFO,
        log_file_dir: str = "logs",
    ):
        self.name = name
        self.min_level = min_level
        self.log_file_dir = Path(log_file_dir)

    def setup(self) -> logging.Logger:
        """配置并返回日志记录器"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.min_level.level)
        logger.handlers.clear()

        # 格式化器
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, date_fmt)

        # 控制台处理器
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.min_level.level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        # 文件处理器（按日期命名）
        if self.log_file_dir:
            self.log_file_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_file_dir / f"{date.today().isoformat()}.log"
            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8",
            )
            file_handler.setLevel(self.min_level.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        Logger._root_logger = logger
        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """获取日志记录器

        Args:
            name: 模块名称，如 __name__

        Returns:
            日志记录器实例
        """
        if Logger._root_logger is None:
            # 默认配置
            Logger(name=name).setup()

        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """便捷函数：获取日志记录器"""
    return Logger.get_logger(name)
