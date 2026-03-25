# ============================================================================
# 模型推理
# ============================================================================

import numpy as np


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
