"""本地磁盘文件 Provider

"""
from pathlib import Path
from typing import Optional, override

from core.interfaces import FileEntry, FileProvider


class LocalDiskFileProvider(FileProvider[bytes, Path]):
    """本地磁盘文件提供者

    FileType: bytes (原始字节)
    PathType: Path (pathlib.Path)
    """

    def __init__(self, base_dir: Path | str | None = None):
        """初始化

        Args:
            base_dir: 基础目录，所有路径相对于此目录解析
        """
        self.base_dir = Path(base_dir) if base_dir else None

    def _resolve_path(self, path: Path | str) -> Path:
        """解析路径"""
        p = Path(path)
        if self.base_dir and not p.is_absolute():
            return self.base_dir / p
        return p

    @override
    def read(self, path: Path | str) -> Optional[bytes]:
        p = self._resolve_path(path)
        if not p.exists():
            return None
        return p.read_bytes()

    @override
    def list(self, path: Path | str) -> list[FileEntry[Path, bytes]] | None:
        p = self._resolve_path(path)
        if not p.is_dir():
            return None

        entries = []
        for file_path in p.iterdir():
            if file_path.is_file():
                content = file_path.read_bytes()
                entries.append(FileEntry(path=file_path, content=content))
        return entries

    @override
    def save(self, path: Path | str, content: bytes, *, overwrite: bool = False) -> bool:
        p = self._resolve_path(path)

        if p.exists() and not overwrite:
            return False

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return True

    @override
    def delete(self, path: Path | str) -> bool:
        p = self._resolve_path(path)
        if not p.exists():
            return False
        p.unlink()
        return True

    @override
    def exists(self, path: Path | str) -> bool:
        return self._resolve_path(path).exists()
