"""用例层测试"""
import pytest
import numpy as np

from core.entities import ImageTask, EvaluationResult
from interactors import EvaluateImageInteractor


class MockProcessor:
    """模拟处理器"""

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        results = []
        for task in tasks:
            result = EvaluationResult.from_raw(
                uid=task.uid,
                rating=("safe", 0.9),
                tags=[("1girl", 0.95), ("solo", 0.8)],
            )
            results.append(result)
        return results


class MockScheduler:
    """模拟调度器"""

    def __init__(self, processor: MockProcessor):
        self._processor = processor

    def submit(
        self,
        tasks: list[ImageTask],
        processor,  # noqa: ARG002
    ) -> list[EvaluationResult]:
        # 模拟乱序返回
        results = self._processor.process(tasks)
        return results[::-1]  # 反转顺序


class TestEvaluateImageUseCase:
    """评估图像用例测试"""

    def test_evaluate_single_image(self):
        """测试单张图像评估"""
        processor = MockProcessor()
        scheduler = MockScheduler(processor)
        usecase = EvaluateImageInteractor(processor=processor, scheduler=scheduler)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = usecase.execute(img)

        assert isinstance(result, EvaluationResult)
        assert result.uid is not None
        assert result.rating.label == "safe"

    def test_evaluate_batch_images(self):
        """测试批量图像评估"""
        processor = MockProcessor()
        scheduler = MockScheduler(processor)
        usecase = EvaluateImageInteractor(processor=processor, scheduler=scheduler)

        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
            np.full((100, 100, 3), 128, dtype=np.uint8),
        ]

        results = usecase.execute(images)

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EvaluationResult)
            assert r.rating.label == "safe"

    def test_uid_preserved_through_disordered_return(self):
        """测试 UID 在乱序返回中保持对应关系

        即使调度器乱序返回结果，UID 仍能正确匹配。
        """
        processor = MockProcessor()
        scheduler = MockScheduler(processor)  # 会反转顺序
        usecase = EvaluateImageInteractor(processor=processor, scheduler=scheduler)

        # 创建带有自定义 UID 的任务
        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]

        results = usecase.execute(images)

        # 验证每个结果都有有效的 UID
        uids = [r.uid for r in results]
        assert len(set(uids)) == 2  # UID 唯一
        for r in results:
            assert r.rating.label == "safe"
            assert len(r.tags) == 2

    def test_empty_batch(self):
        """测试空批次"""
        processor = MockProcessor()
        scheduler = MockScheduler(processor)
        usecase = EvaluateImageInteractor(processor=processor, scheduler=scheduler)

        results = usecase.execute([])

        assert results == []
