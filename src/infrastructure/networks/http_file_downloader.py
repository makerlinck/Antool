"""HTTP 文件下载器实现"""
from pathlib import Path
from urllib.request import urlopen, Request

from core.interfaces.base_file_downloader import (
    BaseFileDownloader,
    DownloadOptions,
    DownloadProgress,
)


class HttpFileDownloader(BaseFileDownloader):
    """HTTP 文件下载器

    基于标准库 urllib 实现，支持断点续传和进度回调。
    """

    def __init__(self):
        pass

    def get_file_size(self, url: str) -> int | None:
        """获取远程文件大小"""
        try:
            with urlopen(Request(url, method="HEAD")) as resp:
                return int(resp.headers.get("Content-Length", 0)) or None
        except Exception:
            return None

    def download_file(
        self,
        url: str,
        dest: Path,
        *,
        options: DownloadOptions | None = None,
        callbacks=None,
    ) -> Path | None:
        """下载文件"""
        opts = options or DownloadOptions()
        dest = Path(dest)

        try:
            req = Request(url, headers=opts.headers or {})

            # 断点续传：检查已有文件大小
            downloaded = 0
            mode = "wb"
            if opts.resume and dest.exists():
                downloaded = dest.stat().st_size
                req.add_header("Range", f"bytes={downloaded}-")
                mode = "ab"

            with urlopen(req, timeout=opts.timeout) as resp:
                # 检查是否支持断点续传
                content_range = resp.headers.get("Content-Range", "")
                if content_range and not content_range.startswith("bytes "):
                    downloaded = 0
                    mode = "wb"
                    dest.unlink(missing_ok=True)

                total = int(resp.headers.get("Content-Length", 0)) or None
                if total:
                    total += downloaded

                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(opts.chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if callbacks:
                            progress = DownloadProgress(
                                total_bytes=total or 0,
                                downloaded_bytes=downloaded,
                            )
                            callbacks.on_progress(progress)

            if callbacks:
                callbacks.on_complete(dest)
            return dest

        except Exception as e:
            if callbacks:
                callbacks.on_error(e)
            return None

    def validate(self, file_path: Path) -> bool:
        """验证文件是否存在且非空"""
        p = Path(file_path)
        return p.exists() and p.stat().st_size > 0
