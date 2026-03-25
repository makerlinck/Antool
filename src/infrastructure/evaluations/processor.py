"""图像评估处理器 - tf.data 版本

使用 tf.data.Dataset 构建高效流水线：
1. 并行预处理（num_parallel_calls=AUTOTUNE）
2. 自动批处理（batch）
3. 预取重叠（prefetch）
"""

import logging
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor

logger = logging.getLogger(__name__)


class ImageEvaluationProcessor(ImageProcessor):
    """图像评估处理器 - tf.data 流水线版本

    职责：构建 tf.data pipeline → 批量推理 → 后处理
    """

    def __init__(
        self,
        model: tf.keras.Model,
        tags: list[str],
        threshold: float = 0.5,
        batch_size: int = 16,
        num_parallel: int = tf.data.AUTOTUNE,
    ):
        self._model = model
        self._tags = tags
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_parallel = num_parallel

        # 获取模型输入尺寸
        self.input_height = model.input_shape[1]
        self.input_width = model.input_shape[2]

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        """批量处理图像任务"""
        if not tasks:
            return []

        t0 = time.perf_counter()

        # 分离有效任务和无效任务
        valid_tasks = []
        for task in tasks:
            if task.image is not None:
                valid_tasks.append(task)

        if not valid_tasks:
            return []

        # 1. 构建 tf.data pipeline
        t1 = time.perf_counter()
        dataset = self._build_pipeline(valid_tasks)
        pipeline_time = (time.perf_counter() - t1) * 1000
        logger.debug(f"[processor] Pipeline build: {pipeline_time:.1f}ms")

        # 2. 批量推理（dataset 自动处理 batching）
        all_scores = []
        batch_count = 0
        for batch_images in dataset:
            batch_count += 1
            batch_size_actual = batch_images.shape[0]
            logger.debug(f"[processor] Batch {batch_count}: shape={batch_images.shape}")
            scores = self._model.predict(batch_images, verbose=0)
            all_scores.append(scores)

        logger.info(f"[processor] Processed {len(valid_tasks)} images in {batch_count} batches")

        inference_time = (time.perf_counter() - t1) * 1000
        logger.info(f"[processor] Inference: {inference_time:.1f}ms for {len(valid_tasks)} images")

        # 3. 合并并后处理
        if not all_scores:
            return []

        all_scores = np.concatenate(all_scores, axis=0)

        # 只取有效的分数（可能少于 batch_size）
        all_scores = all_scores[:len(valid_tasks)]

        t2 = time.perf_counter()
        results = self._postprocess_batch(all_scores, valid_tasks)
        post_time = (time.perf_counter() - t2) * 1000
        logger.debug(f"[processor] Postprocess: {post_time:.1f}ms")

        total_time = (time.perf_counter() - t0) * 1000
        logger.info(f"[processor] Total: {total_time:.1f}ms, per-image: {total_time/len(tasks):.1f}ms")

        return results

    def _build_pipeline(self, tasks: list[ImageTask]) -> tf.data.Dataset:
        """构建 tf.data 处理流水线"""
        # 提取原始图像数据（过滤 None）
        images = []
        for task in tasks:
            if task.image is not None:
                images.append(task.image)

        if not images:
            # 返回空数据集
            return tf.data.Dataset.from_tensor_slices(
                tf.zeros([0, self.input_height, self.input_width, 3], dtype=tf.float32)
            )

        # 使用 generator 避免需要相同尺寸
        def image_generator():
            for img in images:
                yield img

        # 创建数据集（从 generator，支持不同尺寸）
        dataset = tf.data.Dataset.from_generator(
            image_generator,
            output_signature=tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8)
        )

        # 并行预处理
        dataset = dataset.map(
            self._preprocess_fn,
            num_parallel_calls=self.num_parallel,
            deterministic=False,
        )

        # 批处理
        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=False,
        )

        # 预取：在 GPU 推理时，CPU 预处理下一批
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def _preprocess_fn(self, image: tf.Tensor) -> tf.Tensor:
        """单张图像预处理（在 tf.data map 中并行执行）"""
        # 转换为 float32
        image = tf.cast(image, tf.float32)

        # Resize (保持比例，边缘填充)
        image = self._resize_with_pad(image)

        # 归一化
        image = image / 255.0

        return image

    def _resize_with_pad(self, image: tf.Tensor) -> tf.Tensor:
        """保持比例 resize + 中心填充"""
        # 获取原始尺寸
        height = tf.cast(tf.shape(image)[0], tf.float32)
        width = tf.cast(tf.shape(image)[1], tf.float32)
        target_h = float(self.input_height)
        target_w = float(self.input_width)

        # 计算缩放比例
        scale = tf.minimum(target_h / height, target_w / width)
        new_h = tf.cast(tf.round(height * scale), tf.int32)
        new_w = tf.cast(tf.round(width * scale), tf.int32)

        # Resize
        image = tf.image.resize(
            image,
            [new_h, new_w],
            method=tf.image.ResizeMethod.AREA,
        )

        # 计算填充
        pad_top = (self.input_height - new_h) // 2
        pad_bottom = self.input_height - new_h - pad_top
        pad_left = (self.input_width - new_w) // 2
        pad_right = self.input_width - new_w - pad_left

        # 填充到目标尺寸（使用边缘值）
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

_processor: Optional[ImageEvaluationProcessor] = None


def create_processor(batch_size: int = 16) -> ImageEvaluationProcessor:
    """创建处理器实例"""
    from configs import Config
    from infrastructure.evaluations.model_loader import SharedModelLoader

    config = Config()
    loader = SharedModelLoader(config.model_path / "v3-20211112-sgd-e28")

    if loader._model is None:
        loader.load()

    return ImageEvaluationProcessor(
        model=loader.model,
        tags=loader.tags,
        threshold=0.5,
        batch_size=batch_size,
    )


def get_processor(batch_size: int = 16) -> ImageEvaluationProcessor:
    """获取全局处理器实例"""
    global _processor
    if _processor is None:
        _processor = create_processor(batch_size)
    return _processor
