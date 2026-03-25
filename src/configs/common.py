from pathlib import Path


class CommonConfig:
    """通用配置 - 路径定位的唯一来源"""

    path_root: Path
    src_path: Path
    env_file: str = ".env"
    model_files_dir: str = "resources/models"
    locale_text_dir: str = "resources/locales"
    verbose_enabled: bool = False
    default_locale: str = "origin"
    experimental: bool = False
    
    def __init__(self, path_root: Path | str | None = None) -> None:
        if path_root is None:
            path_root = self._find_root()
        self.path_root = Path(path_root)
        self.src_path = self.path_root / "src"
        
    @staticmethod
    def _find_root(start: Path | None = None) -> Path:
        """向上查找项目根目录（从 start 向上找到包含 src 的目录，其父目录即为根目录）"""
        if start is None:
            start = Path.cwd()
            
        path = Path(start).resolve()
        for parent in path.parents:
            if (parent / "src").is_dir():
                return parent
            
        # 回退：找不到则使用当前目录
        return path
    
    @property
    def model_path(self) -> Path:
        """模型文件目录完整路径"""
        return self.path_root / self.model_files_dir
