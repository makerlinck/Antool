"""批量图像评估交互器

业务编排层，处理批量图像评估的完整流程。
职责：
1. 解码图片
2. 并行评估（使用外部提供的调度器）
3. 流式回调结果
4. 指标收集

性能监控点：
1. 图片解码耗时
2. 调度器执行耗时
3. 总耗时
"""

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from core.entities import ImageTask
from core.interfaces.evaluation import ImageProcessor, TaskScheduler
from infrastructure.cancel import CancelledError
from infrastructure.metrics import get_metrics

logger = __import__("logging").getLogger(__name__)


@dataclass
class ImageData:
    """输入图像数据"""
    uid: str
    path: str
    image: Optional[np.ndarray]
    error: Optional[str] = None


@dataclass
class BatchResult:
    """批量处理结果"""
    uid: str
    path: str
    rating: tuple
    tags: list
    error: Optional[str] = None


@dataclass
class BatchPerformance:
    """批量处理性能数据"""
    total_images: int
    valid_images: int
    decode_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    system_info: dict = field(default_factory=dict)


class EvaluateImageInteractor:
    """批量图像评估交互器

    编排流程：
    1. 接收原始图片数据（base64）
    2. 解码图片（在 executor 中并行）
    3. 调用调度器处理（串行，等待返回）
    4. 流式回调每张图的结果
    """

    def __init__(
        self,
        processor: ImageProcessor,
        scheduler: TaskScheduler,
        enable_metrics: bool = True,
        max_workers: int = 4,
    ):
        self._processor = processor
        self._scheduler = scheduler
        self._enable_metrics = enable_metrics
        self._max_workers = max_workers

    async def execute(
        self,
        images_data: list[dict],
        on_result: Callable[[BatchResult], None],
        on_error: Callable[[str], None],
        cancellation=None,
    ) -> Optional[BatchPerformance]:
        """执行批量评估

        流程：解码 → 调度处理 → 流式回调
        """
        perf = BatchPerformance(
            total_images=len(images_data),
            valid_images=0,
            decode_time_ms=0.0,
            inference_time_ms=0.0,
            total_time_ms=0.0,
        )
        t_start = time.perf_counter()
        metrics = get_metrics()

        # 1. 并行解码
        t_decode = time.perf_counter()
        loop = asyncio.get_event_loop()
        decoded_images = await loop.run_in_executor(
            None,
            self._decode_all_images,
            images_data,
        )
        perf.decode_time_ms = (time.perf_counter() - t_decode) * 1000
        logger.info(f"[interactor] Decode: {perf.decode_time_ms:.1f}ms for {len(images_data)} images")

        # 分离有效和无效图片
        valid_images = [img for img in decoded_images if img.error is None]
        perf.valid_images = len(valid_images)

        # 报告解码失败的图片
        for img in decoded_images:
            if img.error:
                on_result(BatchResult(
                    uid=img.uid,
                    path=img.path,
                    rating=("error", 0),
                    tags=[],
                    error=img.error,
                ))

        if not valid_images:
            on_error("No valid images to process")
            perf.total_time_ms = (time.perf_counter() - t_start) * 1000
            return perf if self._enable_metrics else None

        # 2. 创建 ImageTask
        t_tasks = time.perf_counter()
        tasks = [
            ImageTask(image=img.image, uid=img.uid, path=img.path)
            for img in valid_images
        ]
        logger.debug(f"[interactor] Task creation: {(time.perf_counter()-t_tasks)*1000:.1f}ms")

        # 3. 调用调度器处理
        t_infer = time.perf_counter()
        if self._enable_metrics:
            request_uid = f"batch-{id(self)}"
            metrics.request_start(request_uid, len(valid_images))

        try:
            results = self._scheduler.submit(tasks, self._processor)
            perf.inference_time_ms = (time.perf_counter() - t_infer) * 1000
            logger.info(f"[interactor] Schedule: {perf.inference_time_ms:.1f}ms for {len(tasks)} tasks")

            # 4. 流式回调结果
            for i, (task, result) in enumerate(zip(tasks, results)):
                if cancellation and cancellation.is_cancelled:
                    break
                on_result(BatchResult(
                    uid=task.uid,
                    path=task.path,
                    rating=(result.rating.label, result.rating.score),
                    tags=[(tag.name, tag.score) for tag in result.tags],
                ))

        except CancelledError:
            raise
        except Exception as e:
            logger.error(f"[interactor] Schedule error: {e}")
            on_error(f"Batch processing failed: {e}")
        finally:
            if self._enable_metrics:
                metrics.request_end()

        perf.total_time_ms = (time.perf_counter() - t_start) * 1000
        if self._enable_metrics:
            perf.system_info = metrics.get_system_info()

        return perf if self._enable_metrics else None

    def _decode_all_images(self, images_data: list[dict]) -> list[ImageData]:
        """并行解码所有图片"""
        results = []
        for data in images_data:
            uid = data.get("uid", "")
            path = data.get("path", uid)
            b64_data = data.get("data") or data.get("bytes")

            if not b64_data:
                results.append(ImageData(uid, path, None, "No image data"))
                continue

            try:
                img = self._decode_image(b64_data)
                if img is None:
                    results.append(ImageData(uid, path, None, "Failed to decode"))
                else:
                    results.append(ImageData(uid, path, img))
            except Exception as e:
                results.append(ImageData(uid, path, None, str(e)))

        return results

    def _decode_image(self, b64_data: str) -> Optional[np.ndarray]:
        """解码 base64 图片"""
        try:
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            return np.array(img)
        except Exception:
            return None
