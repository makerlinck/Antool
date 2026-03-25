"""模型加载器

负责 TensorFlow 模型的加载和管理。
采用单例模式确保共享内存。
"""

import threading
from pathlib import Path

import numpy as np
import tensorflow as tf


class SharedModelLoader:
    """共享内存模型加载器

    - 单例模式：确保全局唯一实例
    - 共享内存：fork() 后子进程共享
    - 预热机制：避免首次推理延迟
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_dir: Path):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: Path):
        if hasattr(self, "_initialized"):
            return

        self.model_dir = Path(model_dir)
        self._model = None
        self._tags: list[str] | None = None
        self._initialized = True

    def load(self) -> "SharedModelLoader":
        """同步加载模型和标签"""
        self._load_model()
        self._load_tags()
        self._warmup()
        return self

    def _load_model(self) -> None:
        """加载 TensorFlow 模型"""
        model_path = self.model_dir / "model-resnet_custom_v3.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 启用 XLA JIT 编译（加速 CPU 推理）
        tf.config.optimizer.set_jit(True)
        self._model = tf.keras.models.load_model(str(model_path), compile=False)

    def _load_tags(self) -> None:
        """加载标签文件"""
        tags_path = self.model_dir / "tags.txt"
        if not tags_path.exists():
            raise FileNotFoundError(f"Tags file not found: {tags_path}")

        with open(tags_path, encoding="utf-8") as f:
            self._tags = [line.strip() for line in f if line.strip()]

    def _warmup(self) -> None:
        """预热推理 - 使用随机数据确保 XLA 编译完整计算路径"""
        # 模型加载后直接使用
        model = self.model
        dummy = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
        model.predict(dummy, verbose=0)

    @property
    def model(self):
        """获取模型实例"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tags(self) -> list[str]:
        """获取标签列表"""
        if self._tags is None:
            raise RuntimeError("Tags not loaded. Call load() first.")
        return self._tags

    @property
    def input_shape(self) -> tuple[int, ...]:
        """获取模型输入形状"""
        return self.model.input_shape
