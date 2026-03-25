# ============================================================================
# 模型推理
# ============================================================================

import numpy as np


def predict_scores(
    image: np.ndarray,
    model,
    add_batch_dim: bool = True,
) -> np.ndarray:
    """
    模型推理

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
    model,
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
