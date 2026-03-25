import abc
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

FileType = TypeVar("FileType")
PathType = TypeVar("PathType")


@dataclass
class FileEntry(Generic[PathType, FileType]):
    """文件条目，包含路径和内容"""
    path: PathType
    content: FileType


class FileProvider(abc.ABC, Generic[FileType, PathType]):
    """文件提供者抽象基类

    支持本地文件系统和远程存储的统一接口。
    """

    @abc.abstractmethod
    def read(self, path: PathType) -> Optional[FileType]:
        """读取单个文件

        Args:
            path: 文件路径

        Returns:
            文件内容，不存在时返回 None
        """
        pass

    @abc.abstractmethod
    def list(self, path: PathType) -> list[FileEntry[PathType, FileType]] | None:
        """列出目录内所有文件

        Args:
            path: 目录路径

        Returns:
            文件条目列表（含路径和内容），目录不存在时返回 None
        """
        pass

    @abc.abstractmethod
    def save(self, path: PathType, content: FileType, *, overwrite: bool = False) -> bool:
        """保存文件

        Args:
            path: 目标路径
            content: 文件内容
            overwrite: 是否覆盖已存在的文件

        Returns:
            保存成功返回 True，失败返回 False
        """
        pass

    @abc.abstractmethod
    def delete(self, path: PathType) -> bool:
        """删除文件

        Args:
            path: 文件路径

        Returns:
            删除成功返回 True，文件不存在或失败返回 False
        """
        pass

    @abc.abstractmethod
    def exists(self, path: PathType) -> bool:
        """检查文件是否存在

        Args:
            path: 文件路径

        Returns:
            存在返回 True，否则返回 False
        """
        pass
