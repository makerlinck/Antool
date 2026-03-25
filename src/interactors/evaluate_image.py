"""批量图像评估交互器

业务编排层，处理批量图像评估的完整流程。
负责：批量处理、流式响应、指标收集、异常处理
"""
import asyncio
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from core.entities import ImageTask, EvaluationResult
from core.interfaces.evaluation import ImageProcessor, TaskScheduler
from infrastructure.cancel import CancelledError
from infrastructure.metrics import get_metrics


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

    单一职责：编排批量图像评估流程，支持流式返回。
    使用线程池并行处理，每完成一张图立即回调。
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
        """执行批量评估 - 流式返回

        Args:
            images_data: 原始图像数据列表 [{uid, path, data(base64)}]
            on_result: 每处理完一张图的回调（立即调用）
            on_error: 错误回调
            cancellation: 取消管理器

        Returns:
            性能数据（如果 enable_metrics=True）
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

        # 1. 解码所有图片
        loop = asyncio.get_event_loop()
        t_decode_start = time.perf_counter()
        decoded_images = await loop.run_in_executor(
            None,
            self._decode_all_images,
            images_data,
        )
        perf.decode_time_ms = (time.perf_counter() - t_decode_start) * 1000

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

        # 2. 并行处理，流式返回
        if self._enable_metrics:
            request_uid = f"batch-{id(self)}"
            metrics.request_start(request_uid, len(valid_images))

        # 使用 Queue 在线程和协程之间传递结果
        result_queue: asyncio.Queue = asyncio.Queue()

        def process_single(img_data: ImageData):
            """处理单张图片，结果放入队列"""
            if cancellation and cancellation.is_cancelled:
                return

            try:
                task = ImageTask(image=img_data.image, uid=img_data.uid)
                eval_results = self._processor.process([task])

                if eval_results:
                    eval_result = eval_results[0]
                    result = BatchResult(
                        uid=img_data.uid,
                        path=img_data.path,
                        rating=(eval_result.rating.label, eval_result.rating.score),
                        tags=[(tag.name, tag.score) for tag in eval_result.tags],
                    )
                else:
                    result = BatchResult(
                        uid=img_data.uid,
                        path=img_data.path,
                        rating=("error", 0),
                        tags=[],
                        error="No result returned",
                    )
            except Exception as e:
                result = BatchResult(
                    uid=img_data.uid,
                    path=img_data.path,
                    rating=("error", 0),
                    tags=[],
                    error=str(e),
                )

            # 将结果放入队列（线程安全）
            result_queue.put_nowait(result)

        t_infer_start = time.perf_counter()

        try:
            # 启动线程池并行处理
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(process_single, img) for img in valid_images]

                # 异步消费队列，实时回调
                completed = 0

                while completed < len(valid_images):
                    if cancellation and cancellation.is_cancelled:
                        break

                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=0.5)
                        on_result(result)
                        completed += 1
                    except asyncio.TimeoutError:
                        # 检查是否所有任务都完成了
                        if all(f.done() for f in futures):
                            break
                        continue

                # 如果提前退出，确保队列清空（防止 sender 卡住）
                if completed < len(valid_images):
                    while not result_queue.empty():
                        try:
                            result = result_queue.get_nowait()
                            on_result(result)
                            completed += 1
                        except Exception:
                            break

            perf.inference_time_ms = (time.perf_counter() - t_infer_start) * 1000

        except CancelledError:
            # 清空队列，防止 sender 卡住
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    on_result(result)
                except Exception:
                    break
            raise
        except Exception as e:
            on_error(f"Batch processing failed: {e}")
        finally:
            if self._enable_metrics:
                metrics.request_end()

        perf.total_time_ms = (time.perf_counter() - t_start) * 1000
        if self._enable_metrics:
            perf.system_info = metrics.get_system_info()

        return perf if self._enable_metrics else None

    def _collect_futures(self, futures, on_result, cancellation):
        """收集 future 结果并立即回调"""
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            if cancellation and cancellation.is_cancelled:
                break
            try:
                result = future.result()
                if result:
                    on_result(result)
            except Exception:
                pass
            yield future

    def _decode_all_images(self, images_data: list[dict]) -> list[ImageData]:
        """解码所有图片"""
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
