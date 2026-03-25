"""图像评估处理器 - CPU 优化版本

CPU 优化策略：
1. 使用 TensorFlow 预处理（比 skimage 快）
2. 批量推理（虽然 CPU 无并行优化，但减少调用开销）
3. XLA JIT 编译
"""

import logging
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor

logger = logging.getLogger(__name__)


class CPUOptimizedProcessor(ImageProcessor):
    """图像评估处理器 - CPU 优化版本

    职责：预处理 → 推理 → 后处理
    """

    def __init__(
        self,
        model: tf.keras.Model,
        tags: list[str],
        threshold: float = 0.5,
        batch_size: int = 16,
    ):
        self._model = model
        self._tags = tags
        self.threshold = threshold
        self.batch_size = batch_size

        # 获取模型输入尺寸
        self.input_height = model.input_shape[1]
        self.input_width = model.input_shape[2]

        # XLA 编译预处理函数
        self._resize_fn = tf.function(self._resize_with_pad)

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        """批量处理图像任务"""
        if not tasks:
            return []

        t0 = time.perf_counter()

        # 分离有效任务
        valid_tasks = [t for t in tasks if t.image is not None]

        if not valid_tasks:
            return []

        # 1. 批量预处理
        t1 = time.perf_counter()
        preprocessed = self._preprocess_batch(valid_tasks)
        pre_time = (time.perf_counter() - t1) * 1000
        logger.debug(f"[cpu_processor] Preprocess: {pre_time:.1f}ms for {len(valid_tasks)} images")

        # 2. 批量推理
        t2 = time.perf_counter()
        scores = self._model.predict(preprocessed, verbose=0)
        infer_time = (time.perf_counter() - t2) * 1000
        logger.info(f"[cpu_processor] Inference: {infer_time:.1f}ms for {len(valid_tasks)} images")

        # 3. 后处理
        t3 = time.perf_counter()
        results = self._postprocess_batch(scores, valid_tasks)
        post_time = (time.perf_counter() - t3) * 1000
        logger.debug(f"[cpu_processor] Postprocess: {post_time:.1f}ms")

        total_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[cpu_processor] Total: {total_time:.1f}ms, per-image: {total_time/len(tasks):.1f}ms")

        return results

    def _preprocess_batch(self, tasks: list[ImageTask]) -> np.ndarray:
        """批量预处理 - 使用 TensorFlow 向量化操作"""
        batch = []

        for task in tasks:
            if task.image is None:
                continue

            # 转换为 TF tensor
            img = tf.convert_to_tensor(task.image, dtype=tf.float32)

            # Resize + Pad (XLA 编译优化)
            img = self._resize_fn(img)

            # 归一化
            img = img / 255.0

            batch.append(img.numpy())

        if not batch:
            return np.array([])

        return np.stack(batch)

    def _resize_with_pad(self, image: tf.Tensor) -> tf.Tensor:
        """保持比例 resize + 中心填充"""
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)
        target_h = float(self.input_height)
        target_w = float(self.input_width)

        # 计算缩放比例
        scale = tf.minimum(target_h / height, target_w / width)
        new_h = tf.cast(tf.round(height * scale), tf.int32)
        new_w = tf.cast(tf.round(width * scale), tf.int32)

        # Resize
        image = tf.image.resize(image, [new_h, new_w], method=tf.image.ResizeMethod.AREA)

        # 计算填充
        pad_top = (self.input_height - new_h) // 2
        pad_bottom = self.input_height - new_h - pad_top
        pad_left = (self.input_width - new_w) // 2
        pad_right = self.input_width - new_w - pad_left

        # 填充
        image = tf.pad(
            image,
            [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode='SYMMETRIC',
        )

        return image

    def _postprocess_batch(
        self,
        scores: np.ndarray,
        tasks: list[ImageTask],
    ) -> list[EvaluationResult]:
        """批量后处理"""
        from infrastructure.evaluations.filter import filter_tags, weighted_result

        results = []
        for i, task in enumerate(tasks):
            score = scores[i]
            if score.ndim == 2:
                score = score.squeeze()

            raw = filter_tags(score, self._tags, self.threshold)
            result = weighted_result(raw, self._tags, self._tags)

            results.append(EvaluationResult.from_raw(
                uid=task.uid,
                rating=result.rating,
                tags=result.tags,
                path=task.path,
            ))

        return results


# =============================================================================
# 工厂函数
# =============================================================================

_cpu_processor: Optional[CPUOptimizedProcessor] = None


def create_cpu_processor(batch_size: int = 16) -> CPUOptimizedProcessor:
    """创建 CPU 优化处理器实例"""
    from configs import Config
    from infrastructure.evaluations.model_loader import SharedModelLoader

    config = Config()
    loader = SharedModelLoader(config.model_path / "v3-20211112-sgd-e28")

    if loader._model is None:
        loader.load()

    return CPUOptimizedProcessor(
        model=loader.model,
        tags=loader.tags,
        threshold=0.5,
        batch_size=batch_size,
    )


def get_cpu_processor(batch_size: int = 16) -> CPUOptimizedProcessor:
    """获取全局 CPU 优化处理器实例"""
    global _cpu_processor
    if _cpu_processor is None:
        _cpu_processor = create_cpu_processor(batch_size)
    return _cpu_processor
