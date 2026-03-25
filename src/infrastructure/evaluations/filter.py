# ============================================================================
# 结果过滤 (最小单元 3) - 纯过滤，无加权逻辑
# ============================================================================
from dataclasses import dataclass

import numpy as np

CENSORED_KEYS = frozenset(
    "nude anus pussy ejaculation penis nipples naked fellatio urethra".split()
)


@dataclass
class RawTagResult:
    """原始标签过滤结果（未加权）"""

    tags: list[tuple[str, float]]  # [(tag_name, score), ...]
    tag_indices: list[int]  # 激活的标签索引
    rating_scores: np.ndarray  # rating 原始分数 [safe, questionable, nsfw]


@dataclass
class TagResult:
    """最终标签评估结果（加权后）"""

    tags: list[tuple[str, float]]  # [(tag_name, score), ...]
    rating: tuple[str, float]  # (rating_tag, score)


def filter_tags(
    scores: np.ndarray,
    lang_tags: list[str],
    threshold: float,
) -> RawTagResult:
    """
    纯标签过滤

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
        tags=tags, tag_indices=activated_indices, rating_scores=rating_scores.copy()
    )


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
    # TODO 待优化
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

    return TagResult(tags=raw_result.tags.copy(), rating=rating)
