"""调度器测试"""
import pytest
import numpy as np

from core.entities import ImageTask, EvaluationResult
from infrastructure.evaluations.scheduler import BatchScheduler


class MockProcessor:
    """模拟处理器（支持 UID）"""

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        results = []
        for task in tasks:
            result = EvaluationResult.from_raw(
                uid=task.uid,
                rating=("safe", 0.9),
                tags=[("tag1", 0.95)],
            )
            results.append(result)
        return results


class MockDisorderedProcessor:
    """模拟乱序返回的处理器"""

    def process(self, tasks: list[ImageTask]) -> list[EvaluationResult]:
        # 模拟乱序返回（反转结果顺序）
        results = []
        for task in reversed(tasks):
            result = EvaluationResult.from_raw(
                uid=task.uid,
                rating=("safe", 0.9),
                tags=[("tag1", 0.95)],
            )
            results.append(result)
        return results


class TestBatchScheduler:
    """BatchScheduler 测试"""

    def test_small_batch_uses_direct_processing(self):
        """测试小批量直接处理"""
        scheduler = BatchScheduler(thread_threshold=8, max_workers=4, batch_size=32)
        processor = MockProcessor()

        tasks = [
            ImageTask.from_image(np.zeros((100, 100, 3), dtype=np.uint8)),
            ImageTask.from_image(np.ones((100, 100, 3), dtype=np.uint8)),
        ]

        results = scheduler.submit(tasks, processor)

        assert len(results) == 2
        assert results[0].uid == tasks[0].uid
        assert results[1].uid == tasks[1].uid

    def test_large_batch_uses_process_pool(self):
        """测试大批量使用进程池"""
        scheduler = BatchScheduler(thread_threshold=2, max_workers=2, batch_size=2)
        processor = MockProcessor()

        # 超过阈值，触发进程池
        tasks = [
            ImageTask.from_image(np.zeros((100, 100, 3), dtype=np.uint8)),
            ImageTask.from_image(np.ones((100, 100, 3), dtype=np.uint8)),
            ImageTask.from_image(np.full((100, 100, 3), 128, dtype=np.uint8)),
        ]

        results = scheduler.submit(tasks, processor)

        assert len(results) == 3
        # 验证 UID 对应关系
        for i, task in enumerate(tasks):
            assert results[i].uid == task.uid

    def test_disordered_results_mapped_correctly(self):
        """测试乱序结果正确映射"""
        scheduler = BatchScheduler(thread_threshold=2, max_workers=2, batch_size=2)
        processor = MockDisorderedProcessor()  # 会反转顺序

        tasks = [
            ImageTask.from_image(np.zeros((100, 100, 3), dtype=np.uint8), uid="uid-1"),
            ImageTask.from_image(np.ones((100, 100, 3), dtype=np.uint8), uid="uid-2"),
            ImageTask.from_image(np.full((100, 100, 3), 128, dtype=np.uint8), uid="uid-3"),
        ]

        results = scheduler.submit(tasks, processor)

        assert len(results) == 3
        # 无论处理器返回顺序如何，最终结果顺序应与输入一致
        assert results[0].uid == "uid-1"
        assert results[1].uid == "uid-2"
        assert results[2].uid == "uid-3"

    def test_empty_tasks(self):
        """测试空任务列表"""
        scheduler = BatchScheduler()
        processor = MockProcessor()

        results = scheduler.submit([], processor)

        assert results == []

    def test_single_task(self):
        """测试单个任务"""
        scheduler = BatchScheduler(thread_threshold=2, max_workers=2, batch_size=2)
        processor = MockProcessor()

        task = ImageTask.from_image(np.zeros((100, 100, 3), dtype=np.uint8), uid="single")

        results = scheduler.submit([task], processor)

        assert len(results) == 1
        assert results[0].uid == "single"
