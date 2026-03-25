"""核心实体测试"""
import pytest
import numpy as np

from core.entities import Tag, Rating, ImageTask, EvaluationResult


class TestTag:
    """Tag 值对象测试"""

    def test_create_tag(self):
        """测试创建标签"""
        tag = Tag(name="1girl", score=0.95)
        assert tag.name == "1girl"
        assert tag.score == 0.95

    def test_tag_frozen(self):
        """测试标签不可变"""
        tag = Tag(name="1girl", score=0.95)
        with pytest.raises(AttributeError):
            tag.name = "modified"

    def test_tag_score_validation_low(self):
        """测试分数下界验证"""
        with pytest.raises(ValueError, match="Score must be in"):
            Tag(name="test", score=-0.1)

    def test_tag_score_validation_high(self):
        """测试分数上界验证"""
        with pytest.raises(ValueError, match="Score must be in"):
            Tag(name="test", score=1.1)

    def test_tag_score_boundary(self):
        """测试边界值"""
        tag_low = Tag(name="test", score=0.0)
        tag_high = Tag(name="test", score=1.0)
        assert tag_low.score == 0.0
        assert tag_high.score == 1.0


class TestRating:
    """Rating 值对象测试"""

    def test_create_rating(self):
        """测试创建评级"""
        rating = Rating(label="safe", score=0.9)
        assert rating.label == "safe"
        assert rating.score == 0.9

    def test_rating_frozen(self):
        """测试评级不可变"""
        rating = Rating(label="safe", score=0.9)
        with pytest.raises(AttributeError):
            rating.label = "nsfw"

    def test_rating_score_validation(self):
        """测试分数验证"""
        with pytest.raises(ValueError):
            Rating(label="safe", score=1.5)


class TestImageTask:
    """ImageTask 实体测试"""

    def test_create_task_with_auto_uid(self):
        """测试自动生成 UID"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        task = ImageTask.from_image(img)

        assert task.image is img
        assert task.uid is not None
        assert len(task.uid) == 32  # UUID hex 长度

    def test_create_task_with_custom_uid(self):
        """测试自定义 UID"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        task = ImageTask.from_image(img, uid="custom-uid-123")

        assert task.uid == "custom-uid-123"

    def test_task_frozen(self):
        """测试任务不可变"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        task = ImageTask.from_image(img)

        with pytest.raises(AttributeError):
            task.uid = "modified"

    def test_unique_uids(self):
        """测试 UID 唯一性"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        task1 = ImageTask.from_image(img)
        task2 = ImageTask.from_image(img)

        assert task1.uid != task2.uid


class TestEvaluationResult:
    """EvaluationResult 实体测试"""

    def test_create_from_raw(self):
        """测试从原始数据创建"""
        result = EvaluationResult.from_raw(
            uid="test-uid-123",
            rating=("safe", 0.85),
            tags=[("1girl", 0.95), ("solo", 0.8)],
        )

        assert result.uid == "test-uid-123"
        assert result.rating.label == "safe"
        assert result.rating.score == 0.85
        assert len(result.tags) == 2
        assert result.tags[0].name == "1girl"
        assert result.tags[0].score == 0.95

    def test_to_raw(self):
        """测试转换为原始数据"""
        result = EvaluationResult.from_raw(
            uid="test-uid-456",
            rating=("questionable", 0.6),
            tags=[("1girl", 0.9)],
        )

        uid, rating, tags = result.to_raw()
        assert uid == "test-uid-456"
        assert rating == ("questionable", 0.6)
        assert tags == [("1girl", 0.9)]

    def test_tag_names(self):
        """测试获取标签名列表"""
        result = EvaluationResult.from_raw(
            uid="test-uid",
            rating=("safe", 0.9),
            tags=[("1girl", 0.95), ("solo", 0.8), ("blue_eyes", 0.7)],
        )

        assert result.tag_names == ["1girl", "solo", "blue_eyes"]

    def test_empty_tags(self):
        """测试空标签"""
        result = EvaluationResult.from_raw(
            uid="test-uid",
            rating=("safe", 0.5),
            tags=[],
        )
        assert result.tags == ()
        assert result.tag_names == []

    def test_result_frozen(self):
        """测试结果不可变"""
        result = EvaluationResult.from_raw(
            uid="test-uid",
            rating=("safe", 0.9),
            tags=[("1girl", 0.95)],
        )
        with pytest.raises(AttributeError):
            result.rating = Rating(label="nsfw", score=0.8)
