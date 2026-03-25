import math, skimage, numpy as np


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


import tensorflow as tf


def preprocess_image(
    image: np.ndarray,
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

    # 调整尺寸
    image = tf.image.resize(
        image,
        size=target_size,
        # TODO 加入自定义配置
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    ).numpy()

    # 仿射变换填充
    image = transform_and_pad_image(image, target_width, target_height)

    # 归一化
    if normalize:
        image = image.astype(np.float32) / 255.0

    return image
