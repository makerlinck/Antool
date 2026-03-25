import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class DownloadProgress:
    """下载进度"""

    total_bytes: int
    downloaded_bytes: int
    speed_bps: float | None = None  # bytes per second

    @property
    def progress(self) -> float:
        """下载进度百分比 [0, 100]"""
        if self.total_bytes == 0:
            return 0.0
        return self.downloaded_bytes / self.total_bytes * 100


@dataclass
class DownloadOptions:
    """下载选项"""

    chunk_size: int = 8192  # 每次读取的块大小
    timeout: float = 30.0  # 超时时间（秒）
    headers: dict[str, str] | None = None  # 自定义请求头
    resume: bool = True  # 支持断点续传


class DownloaderCallbacks(Protocol):
    """下载器回调协议"""

    def on_progress(self, progress: DownloadProgress) -> None:
        """进度回调"""
        ...

    def on_complete(self, local_path: Path) -> None:
        """下载完成回调"""
        ...

    def on_error(self, error: Exception) -> None:
        """错误回调"""
        ...


class BaseFileDownloader(abc.ABC):
    """文件下载器抽象基类（同步）"""

    @abc.abstractmethod
    def download_file(
        self,
        url: str,
        dest: Path,
        *,
        options: DownloadOptions | None = None,
        callbacks: DownloaderCallbacks | None = None,
    ) -> Path | None:
        """下载文件

        Args:
            url: 文件 URL
            dest: 目标路径
            options: 下载选项
            callbacks: 回调函数

        Returns:
            下载成功返回文件路径，失败返回 None
        """
        pass

    @abc.abstractmethod
    def validate(self, file_path: Path) -> bool:
        """验证文件完整性

        Args:
            file_path: 文件路径

        Returns:
            验证通过返回 True，否则返回 False
        """
        pass

    @abc.abstractmethod
    def get_file_size(self, url: str) -> int | None:
        """获取远程文件大小

        Args:
            url: 文件 URL

        Returns:
            文件大小（字节），获取失败返回 None
        """
        pass


class AsyncBaseFileDownloader(abc.ABC):
    """文件下载器抽象基类（异步）"""

    @abc.abstractmethod
    async def download_file(
        self,
        url: str,
        dest: Path,
        *,
        options: DownloadOptions | None = None,
        callbacks: DownloaderCallbacks | None = None,
    ) -> Path | None:
        """下载文件

        Args:
            url: 文件 URL
            dest: 目标路径
            options: 下载选项
            callbacks: 回调函数

        Returns:
            下载成功返回文件路径，失败返回 None
        """
        pass

    @abc.abstractmethod
    async def validate(self, file_path: Path) -> bool:
        """验证文件完整性

        Args:
            file_path: 文件路径

        Returns:
            验证通过返回 True，否则返回 False
        """
        pass

    @abc.abstractmethod
    async def get_file_size(self, url: str) -> int | None:
        """获取远程文件大小

        Args:
            url: 文件 URL

        Returns:
            文件大小（字节），获取失败返回 None
        """
        pass
