"""HTTP 连接适配器实现"""

import json
from typing import Any
from urllib.request import urlopen, Request

from core.interfaces import BaseHttpConnectionAdapter, AsyncBaseHttpConnectionAdapter
from core.interfaces.base_net_connection_adapter import HttpResponse

import aiohttp  # pip install aiohttp


class HttpConnectionAdapter(BaseHttpConnectionAdapter):
    """同步 HTTP 连接适配器

    基于标准库 urllib 实现，无外部依赖。
    """

    def __init__(
        self, *, timeout: float = 30.0, default_headers: dict[str, str] | None = None
    ):
        self.timeout = timeout
        self.default_headers = default_headers or {}

    def _build_headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        merged = dict(self.default_headers)
        if headers:
            merged.update(headers)
        return merged

    def _do_request(
        self, method: str, url: str, headers: dict[str, str], body: Any
    ) -> HttpResponse:
        """执行 HTTP 请求"""
        headers = self._build_headers(headers)

        # 将 body 序列化为 JSON
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        with urlopen(
            Request(url, data=body, headers=headers, method=method),
            timeout=self.timeout,
        ) as resp:
            return HttpResponse(
                status_code=resp.status,
                content=resp.read(),
                headers=dict(resp.headers),
            )

    def get(self, url: str, **kwargs) -> HttpResponse:
        headers = kwargs.pop("headers", None)
        return self._do_request("GET", url, headers, None)

    def post(self, url: str, **kwargs) -> HttpResponse:
        headers = kwargs.pop("headers", None)
        data = kwargs.pop("json", None) or kwargs.pop("data", None)
        return self._do_request("POST", url, headers, data)

    def put(self, url: str, **kwargs) -> HttpResponse:
        headers = kwargs.pop("headers", None)
        data = kwargs.pop("json", None) or kwargs.pop("data", None)
        return self._do_request("PUT", url, headers, data)

    def delete(self, url: str, **kwargs) -> HttpResponse:
        headers = kwargs.pop("headers", None)
        return self._do_request("DELETE", url, headers, None)

    def patch(self, url: str, **kwargs) -> HttpResponse:
        headers = kwargs.pop("headers", None)
        data = kwargs.pop("json", None) or kwargs.pop("data", None)
        return self._do_request("PATCH", url, headers, data)


class AsyncHttpConnectionAdapter(AsyncBaseHttpConnectionAdapter):
    """异步 HTTP 连接适配器

    基于 aiohttp 实现。
    """

    def __init__(
        self, *, timeout: float = 30.0, default_headers: dict[str, str] | None = None
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.default_headers = default_headers or {}
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.default_headers,
            )
        return self._session

    async def _do_request(self, method: str, url: str, **kwargs) -> HttpResponse:
        session = await self._get_session()
        data = kwargs.pop("json", None) or kwargs.pop("data", None)
        headers = kwargs.pop("headers", None)

        async with session.request(
            method,
            url,
            json=data,
            headers=headers,
            **kwargs,
        ) as resp:
            return HttpResponse(
                status_code=resp.status,
                content=await resp.read(),
                headers=dict(resp.headers),
            )

    async def get(self, url: str, **kwargs) -> HttpResponse:
        return await self._do_request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> HttpResponse:
        return await self._do_request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> HttpResponse:
        return await self._do_request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> HttpResponse:
        return await self._do_request("DELETE", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> HttpResponse:
        return await self._do_request("PATCH", url, **kwargs)

    async def close(self) -> None:
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncBaseHttpConnectionAdapter":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
