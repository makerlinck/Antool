"""
The module Evaluations is based on [DeepDanbooru]( https://github.com/KichangKim/DeepDanbooru )Make modifications。
"""


from typing import Iterator
import numpy as np

from .preprocess import preprocess_image
from .prediction import predict_scores
from .filter import filter_tags, weighted_result

def evaluate_image(
        image_input: np.ndarray,
        model,
        lang_tags: list[str],
        zero_tags: list[str],
        threshold: float,
        normalize: bool = True
) -> tuple[tuple[str,float],list[tuple[str,float]]] | None:
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
    # squeeze 将 (1, num_tags) 转为 (num_tags,)
    if scores.ndim == 2:
        scores = scores.squeeze(0)
    # 3. 纯过滤
    raw_result = filter_tags(scores, lang_tags, threshold)
    # 4. 加权处理
    result = weighted_result(raw_result, lang_tags, zero_tags)
    # 输出
    return result.rating, result.tags

__all__ = ['evaluate_image']