"""图像编码器模块

提供图像加载和编码功能，将图像文件转换为 numpy 数组。
"""
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


def load_image_from_path(image_path: Path | str) -> np.ndarray | None:
    """从文件路径加载图像

    Args:
        image_path: 图像文件路径

    Returns:
        RGB 图像数组 (H, W, 3)，失败返回 None
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        print(f"Load image failed: {e}")
        return None


def load_image_from_bytes(data: bytes) -> np.ndarray | None:
    """从字节数据加载图像

    Args:
        data: 图像字节数据

    Returns:
        RGB 图像数组 (H, W, 3)，失败返回 None
    """
    try:
        with Image.open(BytesIO(data)) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        print(f"Load image from bytes failed: {e}")
        return None


def load_image_from_fileobj(fo: BinaryIO) -> np.ndarray | None:
    """从文件对象加载图像

    Args:
        fo: 二进制文件对象

    Returns:
        RGB 图像数组 (H, W, 3)，失败返回 None
    """
    try:
        with Image.open(fo) as img:
            img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        print(f"Load image from file object failed: {e}")
        return None


def encode_image(image: np.ndarray, format: str = "PNG") -> bytes:
    """将图像数组编码为字节

    Args:
        image: RGB 图像数组 (H, W, 3)
        format: 输出格式 (PNG, JPEG, etc.)

    Returns:
        编码后的图像字节
    """
    img = Image.fromarray(image.astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def decode_image(data: bytes) -> np.ndarray | None:
    """解码图像字节数据（load_image_from_bytes 的别名）

    Args:
        data: 图像字节数据

    Returns:
        RGB 图像数组 (H, W, 3)，失败返回 None
    """
    return load_image_from_bytes(data)
