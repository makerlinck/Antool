"""evaluate_image 模块测试"""
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
SRC_PATH = Path(__file__).parent.parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import tensorflow as tf

import pytest


# ============================================================================
# 测试资源路径
# ============================================================================

TEST_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = TEST_DIR.parent

TEST_IMAGE_PATH = TEST_DIR / "resources" / "test_images" / "b2de9ead7ada094107d95cc8d4883062.jpeg"
MODEL_PATH = PROJECT_ROOT / "resources" / "models" / "v3-20211112-sgd-e28" / "model-resnet_custom_v3.h5"
TAGS_PATH = PROJECT_ROOT / "resources" / "models" / "v3-20211112-sgd-e28" / "tags.txt"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def model():
    """加载 TensorFlow 模型"""
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


@pytest.fixture(scope="module")
def tags() -> list[str]:
    """加载标签文件"""
    with open(TAGS_PATH, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@pytest.fixture
def image_array() -> np.ndarray:
    """加载测试图像"""
    from infrastructure.evaluations.image_encoder import load_image_from_path
    img = load_image_from_path(TEST_IMAGE_PATH)
    assert img is not None, "测试图像加载失败"
    return img


# ============================================================================
# 测试用例
# ============================================================================

class TestImageEncoder:
    """图像编码器测试"""

    def test_load_image_from_path(self):
        """测试从路径加载图像"""
        from infrastructure.evaluations.image_encoder import load_image_from_path

        image = load_image_from_path(TEST_IMAGE_PATH)

        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3  # RGB

    def test_load_nonexistent_image(self):
        """测试加载不存在的图像"""
        from infrastructure.evaluations.image_encoder import load_image_from_path

        image = load_image_from_path("/nonexistent/path.jpg")
        assert image is None

    def test_encode_decode_cycle(self):
        """测试编码解码循环"""
        from infrastructure.evaluations.image_encoder import (
            load_image_from_path,
            encode_image,
            decode_image,
        )

        original = load_image_from_path(TEST_IMAGE_PATH)
        assert original is not None

        encoded = encode_image(original, format="PNG")
        assert isinstance(encoded, bytes)

        decoded = decode_image(encoded)
        assert decoded is not None
        assert np.array_equal(original, decoded)


class TestPreprocess:
    """预处理测试"""

    def test_preprocess_image(self, image_array: np.ndarray, model):
        """测试图像预处理"""
        from infrastructure.evaluations.preprocess import preprocess_image

        target_size = (model.input_shape[1], model.input_shape[2])
        processed = preprocess_image(image_array, target_size)

        assert processed is not None
        assert processed.shape == (*target_size, 3)
        assert processed.dtype == np.float32
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0


class TestPrediction:
    """模型推理测试"""

    def test_predict_scores(self, image_array: np.ndarray, model):
        """测试模型推理"""
        from infrastructure.evaluations.preprocess import preprocess_image
        from infrastructure.evaluations.prediction import predict_scores

        target_size = (model.input_shape[1], model.input_shape[2])
        processed = preprocess_image(image_array, target_size)
        assert processed is not None

        scores = predict_scores(processed, model)

        assert scores is not None
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 2  # (1, num_tags)
        assert scores.shape[0] == 1


class TestFilter:
    """结果过滤测试"""

    def test_filter_tags(self, tags: list[str]):
        """测试标签过滤"""
        from infrastructure.evaluations.filter import filter_tags

        # 模拟分数
        scores = np.random.rand(len(tags))

        result = filter_tags(scores, tags, threshold=0.5)

        assert result is not None
        assert hasattr(result, 'tags')
        assert hasattr(result, 'tag_indices')
        assert hasattr(result, 'rating_scores')

    def test_weighted_result(self, tags: list[str]):
        """测试加权结果"""
        from infrastructure.evaluations.filter import filter_tags, weighted_result

        scores = np.random.rand(len(tags))
        raw = filter_tags(scores, tags, threshold=0.5)

        # 使用 tags 同时作为 lang_tags 和 zero_tags
        result = weighted_result(raw, tags, tags)

        assert result is not None
        assert hasattr(result, 'tags')
        assert hasattr(result, 'rating')
        assert isinstance(result.rating, tuple)
        assert len(result.rating) == 2


class TestEvaluateImage:
    """evaluate_image 集成测试"""

    def test_evaluate_image_full(self, image_array: np.ndarray, model, tags: list[str]):
        """完整流程测试"""
        from infrastructure.evaluations import evaluate_image

        result = evaluate_image(
            image_input=image_array,
            model=model,
            lang_tags=tags,
            zero_tags=tags,
            threshold=0.5,
        )

        assert result is not None
        rating, tags_result = result

        assert isinstance(rating, tuple)
        assert len(rating) == 2
        assert isinstance(rating[0], str)
        assert isinstance(rating[1], float)

        assert isinstance(tags_result, list)
        for tag_name, score in tags_result:
            assert isinstance(tag_name, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
