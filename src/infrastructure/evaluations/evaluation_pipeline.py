"""图像标签评估模块

最小计算单元结构:
┌─────────────────────────────────────────────────────────────┐
│  preprocess_image(path) → np.ndarray                        │  图像预处理
│  predict_scores(image, model) → np.ndarray                  │  模型推理
│  filter_tags(scores, tags, threshold) → RawTagResult        │  纯标签过滤
│  weighted_result(raw_result, zero_tags) → TagResult         │  加权处理
└─────────────────────────────────────────────────────────────┘
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import skimage.transform
import tensorflow as tf

tf.config.optimizer.set_jit(True)  # 启用XLA加速

# 模块级常量
CENSORED_KEYS = frozenset("nude anus pussy ejaculation penis nipples naked fellatio urethra".split())


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RawTagResult:
    """原始标签过滤结果（未加权）"""
    tags: list[tuple[str, float]]      # [(tag_name, score), ...]
    tag_indices: list[int]              # 激活的标签索引
    rating_scores: np.ndarray           # rating 原始分数 [safe, questionable, nsfw]


@dataclass
class TagResult:
    """最终标签评估结果（加权后）"""
    tags: list[tuple[str, float]]       # [(tag_name, score), ...]
    rating: tuple[str, float]           # (rating_tag, score)


# ============================================================================
# 图像预处理 (最小单元 1)
# ============================================================================

@tf.function
def _decode_image(image_path: str) -> tf.Tensor:
    """TF Graph 模式解码图像"""
    image_raw = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    return image


def transform_and_pad_image(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        scale: float | None = None,
        rotation: float | None = None,
        shift: tuple[float, float] | None = None,
        order: int = 1,
        mode: str = "edge",
) -> np.ndarray:
    """应用仿射变换处理图像，并通过边缘像素扩展填充至目标尺寸"""
    image_height, image_width = image.shape[:2]

    # 构建变换矩阵
    tx = -image_width * 0.5
    ty = -image_height * 0.5

    matrix = np.eye(3)
    matrix[0, 2] = tx
    matrix[1, 2] = ty

    if scale:
        s_matrix = np.eye(3)
        s_matrix[0, 0] = scale
        s_matrix[1, 1] = scale
        matrix = s_matrix @ matrix

    if rotation:
        rad = rotation * math.pi / 180
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        r_matrix = np.eye(3)
        r_matrix[0, 0] = cos_r
        r_matrix[0, 1] = -sin_r
        r_matrix[1, 0] = sin_r
        r_matrix[1, 1] = cos_r
        matrix = r_matrix @ matrix

    t2_matrix = np.eye(3)
    t2_matrix[0, 2] = target_width * 0.5
    t2_matrix[1, 2] = target_height * 0.5
    matrix = t2_matrix @ matrix

    if shift:
        sft_matrix = np.eye(3)
        sft_matrix[0, 2] = target_width * shift[0]
        sft_matrix[1, 2] = target_height * shift[1]
        matrix = sft_matrix @ matrix

    t = skimage.transform.AffineTransform(matrix=np.linalg.inv(matrix))
    image = skimage.transform.warp(
        image, t, output_shape=(target_height, target_width), order=order, mode=mode
    )
    return image


def preprocess_image(
        image_input: Path | str | np.ndarray,
        target_size: tuple[int, int],
        normalize: bool = True,
) -> np.ndarray | None:
    """
    图像预处理 (最小计算单元 1)

    Args:
        image_input: 图像路径或 numpy 数组
        target_size: (height, width) 目标尺寸
        normalize: 是否归一化到 [0, 1]

    Returns:
        预处理后的图像数组 (H, W, C)，失败返回 None
    """
    target_height, target_width = target_size

    # 从文件加载
    if isinstance(image_input, (Path, str)):
        try:
            image = _decode_image(str(image_input))
            image = image.numpy()
        except Exception as e:
            print(f"Decode image failed: {e}")
            return None
    else:
        image = image_input

    # 调整尺寸
    image = tf.image.resize(
        image,
        size=target_size,
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    ).numpy()

    # 仿射变换填充
    image = transform_and_pad_image(image, target_width, target_height)

    # 归一化
    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


# ============================================================================
# 模型推理 (最小计算单元 2)
# ============================================================================

def predict_scores(
        image: np.ndarray,
        model: Any,
        add_batch_dim: bool = True,
) -> np.ndarray:
    """
    模型推理 (最小计算单元 2)

    Args:
        image: 预处理后的图像 (H, W, C) 或 (N, H, W, C)
        model: TensorFlow 模型
        add_batch_dim: 是否自动添加 batch 维度

    Returns:
        预测分数数组 (num_tags,) 或 (N, num_tags)
    """
    if add_batch_dim and image.ndim == 3:
        image = image.reshape((1, *image.shape))

    return model.predict(image, verbose=0)


def predict_scores_batch(
        images: list[np.ndarray],
        model: Any,
) -> np.ndarray:
    """
    批量模型推理

    Args:
        images: 预处理后的图像列表
        model: TensorFlow 模型

    Returns:
        预测分数数组 (N, num_tags)
    """
    if not images:
        return np.array([])

    batch = np.stack(images)
    return model.predict(batch, verbose=0)


# ============================================================================
# 结果过滤 (最小单元 3) - 纯过滤，无加权逻辑
# ============================================================================

def filter_tags(
        scores: np.ndarray,
        lang_tags: list[str],
        threshold: float,
) -> RawTagResult:
    """
    纯标签过滤 (最小计算单元 3)

    仅根据阈值过滤标签，不包含任何加权逻辑。

    Args:
        scores: 模型预测分数 (num_tags,)
        lang_tags: 语言标签列表
        threshold: 置信度阈值

    Returns:
        RawTagResult: 原始过滤结果，包含 tag_indices 和 rating_scores
    """
    len_tags = len(lang_tags)
    rating_start = len_tags - 3

    # 获取 rating 原始分数
    rating_scores = scores[rating_start:]

    # 向量化过滤普通标签
    mask = scores[:rating_start] >= threshold
    activated_indices = np.where(mask)[0].tolist()

    # 收集标签
    tags = [(lang_tags[idx], float(scores[idx])) for idx in activated_indices]

    return RawTagResult(
        tags=tags,
        tag_indices=activated_indices,
        rating_scores=rating_scores.copy()
    )


# ============================================================================
# 加权处理 (最小单元 4) - 敏感词检测等加权逻辑
# ============================================================================

def weighted_result(
        raw_result: RawTagResult,
        lang_tags: list[str],
        zero_tags: list[str],
) -> TagResult:
    """
    加权处理 (最小计算单元 4)

    对原始过滤结果进行加权处理，包括敏感词检测等。

    Args:
        raw_result: 原始过滤结果
        lang_tags: 语言标签列表（用于获取 rating 标签名）
        zero_tags: 零标签列表（用于敏感词检测）

    Returns:
        TagResult: 加权后的最终结果
    """
    t_safe, t_sus, t_nsfw = lang_tags[-3], lang_tags[-2], lang_tags[-1]
    rating_scores = raw_result.rating_scores

    # 获取激活标签对应的 zero_tags（用于敏感词检测）
    activated_zero_tags = [zero_tags[idx] for idx in raw_result.tag_indices]

    # 敏感词检测加权
    if any(tag in CENSORED_KEYS for tag in activated_zero_tags):
        rating = (t_nsfw, float(rating_scores[2]))
    else:
        # 默认 rating 选择逻辑
        max_idx = int(np.argmax(rating_scores))
        if max_idx == 0:
            rating = (t_safe, float(rating_scores[0]))
        elif max_idx == 1:
            if rating_scores[0] > rating_scores[2]:
                rating = (t_sus, float(rating_scores[1]))
            else:
                rating = (t_nsfw, float(rating_scores[2]))
        else:
            rating = (t_nsfw, float(rating_scores[2]))

    return TagResult(
        tags=raw_result.tags.copy(),
        rating=rating
    )


# ============================================================================
# 高层 API (组合最小单元)
# ============================================================================

def evaluate_image(
        image_input: Path | str,
        model: Any,
        lang_tags: list[str],
        zero_tags: list[str],
        threshold: float,
        normalize: bool = True
) -> Iterator[tuple[str, float]] | None:
    """
    评估单张图像的标签

    流程: preprocess_image → predict_scores → filter_tags → weighted_result
    """
    target_size = (model.input_shape[1], model.input_shape[2])

    # 1. 预处理
    image = preprocess_image(image_input, target_size, normalize)
    if image is None:
        return None

    # 2. 推理
    scores = predict_scores(image, model)

    # 3. 纯过滤
    raw_result = filter_tags(scores, lang_tags, threshold)

    # 4. 加权处理
    result = weighted_result(raw_result, lang_tags, zero_tags)

    # 输出
    yield from result.tags
    yield result.rating


def evaluate_batch(
        image_inputs: list[Path | str],
        model: Any,
        lang_tags: list[str],
        zero_tags: list[str],
        threshold: float,
        batch_size: int = 32,
        normalize: bool = True
) -> Iterator[TagResult | None]:
    """
    批量评估图像标签

    流程: preprocess_image (N次) → predict_scores_batch → filter_tags → weighted_result (N次)
    """
    if not image_inputs:
        return

    target_size = (model.input_shape[1], model.input_shape[2])

    # 分批处理
    for batch_start in range(0, len(image_inputs), batch_size):
        batch_paths = image_inputs[batch_start:batch_start + batch_size]

        # 1. 批量预处理
        valid_images = []
        valid_indices = []

        for i, image_input in enumerate(batch_paths):
            image = preprocess_image(image_input, target_size, normalize)
            if image is not None:
                valid_images.append(image)
                valid_indices.append(i)
            else:
                valid_indices.append(-1)

        # 2. 批量推理
        predictions = predict_scores_batch(valid_images, model) if valid_images else np.array([])

        # 3. 批量过滤 + 加权
        pred_idx = 0
        for i in range(len(batch_paths)):
            if valid_indices[i] == -1:
                yield None
            else:
                raw_result = filter_tags(predictions[pred_idx], lang_tags, threshold)
                result = weighted_result(raw_result, lang_tags, zero_tags)
                pred_idx += 1
                yield result
