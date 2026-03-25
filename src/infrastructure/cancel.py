"""任务取消管理器

支持优雅取消所有正在处理的任务。
"""
import threading
from enum import Enum, auto
from typing import Optional

from dataclasses import dataclass, field


class CancelReason(Enum):
    """取消原因"""
    SHUTDOWN = auto()
    USER_REQUEST = auto()
    TIMEOUT = auto()
    ERROR = auto()


@dataclass
class CancelScope:
    """取消作用域"""
    is_cancelled: bool = False
    reason: Optional[CancelReason] = None


class CancellationManager:
    """任务取消管理器

    支持全局取消和作用域级取消。
    """

    _instance: Optional["CancellationManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CancellationManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._lock = threading.Lock()
        self._cancelled = False
        self._reason: Optional[CancelReason] = None
        self._scopes: list[CancelScope] = []
        self._listeners: list[callable] = []

        self._initialized = True

    def cancel(self, reason: CancelReason = CancelReason.USER_REQUEST) -> None:
        """触发取消"""
        with self._lock:
            self._cancelled = True
            self._reason = reason
            # 通知所有作用域
            for scope in self._scopes:
                scope.is_cancelled = True
                scope.reason = reason
            # 通知监听器
            for listener in self._listeners:
                try:
                    listener(reason)
                except Exception:
                    pass

    def reset(self) -> None:
        """重置取消状态"""
        with self._lock:
            self._cancelled = False
            self._reason = None
            for scope in self._scopes:
                scope.is_cancelled = False
                scope.reason = None

    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> Optional[CancelReason]:
        """获取取消原因"""
        with self._lock:
            return self._reason

    def create_scope(self) -> CancelScope:
        """创建取消作用域"""
        scope = CancelScope()
        with self._lock:
            self._scopes.append(scope)
        return scope

    def remove_scope(self, scope: CancelScope) -> None:
        """移除取消作用域"""
        with self._lock:
            if scope in self._scopes:
                self._scopes.remove(scope)

    def add_listener(self, listener: callable) -> None:
        """添加取消监听器"""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: callable) -> None:
        """移除取消监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)


# 全局取消管理器
def get_cancellation() -> CancellationManager:
    """获取取消管理器实例"""
    return CancellationManager()


class CancelledError(Exception):
    """任务被取消异常"""

    def __init__(self, reason: CancelReason):
        self.reason = reason
        super().__init__(f"Task cancelled: {reason.name}")
