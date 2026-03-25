import abc
from dataclasses import dataclass
from typing import Any, TypeVar

ResponseData = TypeVar("ResponseData")


@dataclass
class HttpResponse:
    """HTTP 响应封装"""
    status_code: int
    content: bytes
    headers: dict[str, str]
    encoding: str = "utf-8"

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding)

    def json(self) -> Any:
        import json
        return json.loads(self.content)


class BaseHttpConnectionAdapter(abc.ABC):
    """同步网络连接适配器抽象基类"""

    @abc.abstractmethod
    def get(self, url: str, **kwargs) -> HttpResponse:
        """发送 GET 请求"""
        pass

    @abc.abstractmethod
    def post(self, url: str, **kwargs) -> HttpResponse:
        """发送 POST 请求"""
        pass

    @abc.abstractmethod
    def put(self, url: str, **kwargs) -> HttpResponse:
        """发送 PUT 请求"""
        pass

    @abc.abstractmethod
    def delete(self, url: str, **kwargs) -> HttpResponse:
        """发送 DELETE 请求"""
        pass

    @abc.abstractmethod
    def patch(self, url: str, **kwargs) -> HttpResponse:
        """发送 PATCH 请求"""
        pass


class AsyncBaseHttpConnectionAdapter(abc.ABC):
    """异步网络连接适配器抽象基类"""

    @abc.abstractmethod
    async def get(self, url: str, **kwargs) -> HttpResponse:
        """异步发送 GET 请求"""
        pass

    @abc.abstractmethod
    async def post(self, url: str, **kwargs) -> HttpResponse:
        """异步发送 POST 请求"""
        pass

    @abc.abstractmethod
    async def put(self, url: str, **kwargs) -> HttpResponse:
        """异步发送 PUT 请求"""
        pass

    @abc.abstractmethod
    async def delete(self, url: str, **kwargs) -> HttpResponse:
        """异步发送 DELETE 请求"""
        pass

    @abc.abstractmethod
    async def patch(self, url: str, **kwargs) -> HttpResponse:
        """异步发送 PATCH 请求"""
        pass


# =============================================================================
# WebSocket 适配器
# =============================================================================

class BaseWebSocketAdapter(abc.ABC):
    """同步 WebSocket 适配器抽象基类"""

    @abc.abstractmethod
    def connect(self) -> None:
        """建立连接"""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abc.abstractmethod
    def send(self, data: Any) -> None:
        """发送消息"""
        pass

    @abc.abstractmethod
    def recv(self) -> Any:
        """接收消息"""
        pass

    @abc.abstractmethod
    def send_text(self, data: str) -> None:
        """发送文本消息"""
        pass

    @abc.abstractmethod
    def send_binary(self, data: bytes) -> None:
        """发送二进制消息"""
        pass


class AsyncBaseWebSocketAdapter(abc.ABC):
    """异步 WebSocket 适配器抽象基类"""

    @abc.abstractmethod
    async def connect(self) -> None:
        """建立连接"""
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass

    @abc.abstractmethod
    async def send(self, data: Any) -> None:
        """发送消息"""
        pass

    @abc.abstractmethod
    async def recv(self) -> Any:
        """接收消息"""
        pass

    @abc.abstractmethod
    async def send_text(self, data: str) -> None:
        """发送文本消息"""
        pass

    @abc.abstractmethod
    async def send_binary(self, data: bytes) -> None:
        """发送二进制消息"""
        pass
