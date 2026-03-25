"""指标系统

提供服务运行指标收集，包括：
- 模型加载耗时
- 请求延迟
- 每张图片处理延迟
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class MetricRecord:
    """指标记录"""

    uid: str = field(default_factory=lambda: uuid4().hex)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """耗时（毫秒）"""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def stop(self) -> "MetricRecord":
        """停止计时"""
        self.end_time = time.perf_counter()
        return self


@dataclass
class RequestMetrics:
    """请求指标"""

    uid: str
    num_images: int
    total_latency_ms: float
    avg_latency_ms: float
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """指标收集器

    线程安全，支持指标记录和聚合。
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._lock = threading.RLock()
        self._verbose: bool = False

        # 模型加载指标
        self._model_load_start: Optional[float] = None
        self._model_load_end: Optional[float] = None

        # 当前请求指标
        self._current_request: Optional[MetricRecord] = None
        self._current_num_images: int = 0
        self._image_records: list[MetricRecord] = []

        # 历史请求指标
        self._request_history: list[RequestMetrics] = []
        self._max_history: int = 1000

        self._initialized = True

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._model_load_start = None
            self._model_load_end = None
            self._current_request = None
            self._current_num_images = 0
            self._image_records.clear()
            self._request_history.clear()

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    # =========================================================================
    # 模型加载指标
    # =========================================================================

    def model_load_start(self) -> None:
        """标记模型开始加载"""
        self._model_load_start = time.perf_counter()

    def model_load_end(self) -> None:
        """标记模型加载完成"""
        self._model_load_end = time.perf_counter()

    @property
    def model_load_duration_ms(self) -> Optional[float]:
        """模型加载耗时（毫秒）"""
        if self._model_load_start is None or self._model_load_end is None:
            return None
        return (self._model_load_end - self._model_load_start) * 1000

    # =========================================================================
    # 请求指标
    # =========================================================================

    def request_start(self, uid: str, num_images: int) -> MetricRecord:
        """标记请求开始"""
        record = MetricRecord(uid=uid)
        self._current_request = record
        self._current_num_images = num_images
        self._image_records.clear()
        return record

    def request_end(self) -> Optional[RequestMetrics]:
        """标记请求结束并记录指标"""
        if self._current_request is None:
            return None

        self._current_request.stop()
        request = RequestMetrics(
            uid=self._current_request.uid,
            num_images=self._current_num_images,
            total_latency_ms=self._current_request.duration_ms or 0,
            avg_latency_ms=(self._current_request.duration_ms or 0)
            / max(self._current_num_images, 1),
        )

        with self._lock:
            self._request_history.append(request)
            # 限制历史记录数量
            if len(self._request_history) > self._max_history:
                self._request_history = self._request_history[-self._max_history:]

        self._current_request = None
        return request

    # =========================================================================
    # 图片处理指标
    # =========================================================================

    def image_start(self) -> MetricRecord:
        """标记图片开始处理"""
        record = MetricRecord()
        self._image_records.append(record)
        return record

    def image_end(self, record: MetricRecord) -> MetricRecord:
        """标记图片处理结束"""
        return record.stop()

    # =========================================================================
    # 聚合指标
    # =========================================================================

    @property
    def recent_requests(self) -> list[RequestMetrics]:
        """获取最近的请求指标"""
        with self._lock:
            return list(self._request_history)

    @property
    def avg_image_latency_ms(self) -> Optional[float]:
        """平均图片延迟（毫秒）"""
        with self._lock:
            if not self._request_history:
                return None
            total = sum(r.avg_latency_ms * r.num_images for r in self._request_history)
            count = sum(r.num_images for r in self._request_history)
            return total / count if count > 0 else None

    def get_summary(self) -> dict:
        """获取指标摘要"""
        with self._lock:
            return {
                "model_load_duration_ms": self.model_load_duration_ms,
                "avg_image_latency_ms": self.avg_image_latency_ms,
                "total_requests": len(self._request_history),
                "verbose": self._verbose,
                "system": self.get_system_info(),
            }

    def get_system_info(self) -> dict:
        """获取系统性能信息"""
        import os
        import psutil

        process = psutil.Process(os.getpid())
        mem = process.memory_info()

        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0),
            "memory_rss_mb": round(mem.rss / 1024 / 1024, 2),
            "memory_vms_mb": round(mem.vms / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
        }

    def get_image_latencies(self) -> list[dict]:
        """获取所有图片延迟（verbose 模式）"""
        if not self._verbose:
            return []

        with self._lock:
            return [
                {"uid": r.uid, "latency_ms": r.duration_ms}
                for r in self._image_records
                if r.duration_ms is not None
            ]


# 全局指标收集器实例
def get_metrics() -> MetricsCollector:
    """获取指标收集器实例"""
    return MetricsCollector()
