"""远程文件 Provider（支持在线读取和缓存）"""

from pathlib import Path
from typing import Optional
from core.interfaces import FileEntry, FileProvider


class RemoteFileProvider(FileProvider[bytes, str]):
    """远程文件提供者

    支持在线读取远程文件（如图片 URL 缓存到本地），
    若读取失败则返回 None。
    """

    def __init__(
        self,
        http_adapter,
        *,
        cache_dir: Path | str | None = None,
    ):
        """
        Args:
            http_adapter: HTTP 连接适配器
            cache_dir: 本地缓存目录，None 则不缓存
        """
        self._http = http_adapter
        self._cache_dir = Path(cache_dir) if cache_dir else None

    # ========================================================================
    # FileProvider 接口实现
    # ========================================================================

    def read(self, path: str) -> Optional[bytes]:
        """在线读取远程文件

        优先从本地缓存读取，缓存不存在则从 URL 下载。
        读取失败返回 None。
        """
        url = path
        cache_path = self._get_cache_path(url) if self._cache_dir else None

        # 优先从缓存读取
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()

        # 从 URL 下载
        try:
            resp = self._http.get(url)
            if resp.status_code != 200:
                return None
            content = resp.content
        except Exception:
            return None

        # 保存到缓存
        if cache_path and content:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)

        return content

    def list(self, path: str) -> list[FileEntry[str, bytes]] | None:
        """远程 Provider 不支持目录列表"""
        return None

    def save(self, path: str, content: bytes, *, overwrite: bool = False) -> bool:
        """远程 Provider 不支持保存到远程"""
        return False

    def delete(self, path: str) -> bool:
        """删除远程文件（通常不支持）"""
        return False

    def exists(self, path: str) -> bool:
        """检查远程文件是否存在（HEAD 请求）"""
        url = path
        try:
            resp = self._http._do_request("HEAD", url, {}, None)  # type: ignore
            return resp.status_code == 200
        except Exception:
            return False

    # ========================================================================
    # 缓存管理
    # ========================================================================

    def _get_cache_path(self, url: str) -> Path | None:
        """生成本地缓存路径"""
        if not self._cache_dir:
            return None
        import hashlib

        key = hashlib.md5(url.encode()).hexdigest()
        return self._cache_dir / f"{key}.cache"

    def clear_cache(self, url: str | None = None) -> int:
        """清除缓存

        Args:
            url: 指定 URL 的缓存，None 则清除所有缓存

        Returns:
            清除的文件数量
        """
        if not self._cache_dir:
            return 0

        if url:
            cache_path = self._get_cache_path(url)
            if cache_path and cache_path.exists():
                cache_path.unlink()
                return 1
            return 0

        # 清除所有缓存
        count = 0
        if self._cache_dir.exists():
            for f in self._cache_dir.iterdir():
                if f.suffix == ".cache":
                    f.unlink()
                    count += 1
        return count
