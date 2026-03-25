import abc
from typing import Generic, List, Optional, TypeVar


T = TypeVar("T")
ID = TypeVar("ID")


class BaseRepository(abc.ABC, Generic[T, ID]):

    @abc.abstractmethod
    def find(self, id: ID) -> Optional[T]:
        """查找实体( 根据ID )"""
        pass

    @abc.abstractmethod
    def find_all(self) -> List[T]:
        """查找所有实体"""
        pass

    @abc.abstractmethod
    def save(self, entity: T) -> T:
        pass

    @abc.abstractmethod
    def delete(self, id: ID) -> bool:
        """根据ID删除实体"""
        pass

    @abc.abstractmethod
    def exists(self, id: ID) -> bool:
        """检查实体是否存在"""
        pass

    @abc.abstractmethod
    def count(self) -> int:
        """获取实体总数"""
        pass


class AsyncBaseRepository(abc.ABC, Generic[T, ID]):

    @abc.abstractmethod
    async def find(self, id: ID) -> Optional[T]:
        """异步 查找实体( 根据ID )"""
        pass

    @abc.abstractmethod
    async def find_all(self) -> List[T]:
        """异步 查找所有实体"""
        pass

    @abc.abstractmethod
    async def save(self, entity: T) -> T:
        pass

    @abc.abstractmethod
    async def delete(self, id: ID) -> bool:
        """异步 根据ID删除实体"""
        pass

    @abc.abstractmethod
    async def exists(self, id: ID) -> bool:
        """异步 检查实体是否存在"""
        pass

    @abc.abstractmethod
    async def count(self) -> int:
        """异步 获取实体总数"""
        pass
