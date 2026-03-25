"""应用构建器

"""
from contextlib import asynccontextmanager
from typing import Callable, TypeVar
from litestar import Litestar
from configs import Config
T = TypeVar("T")


class ServiceContainer:
    """服务容器"""

    def __init__(self):
        self._services: dict[str, object] = {}
        self._factories: dict[str, Callable[[], object]] = {}

    def register(self, name: str, instance: object) -> None:
        """注册服务实例"""
        self._services[name] = instance

    def register_factory(self, name: str, factory: Callable[[], object]) -> None:
        """注册服务工厂（延迟创建）"""
        self._factories[name] = factory

    def get(self, name: str) -> object | None:
        """获取服务"""
        if name in self._services:
            return self._services[name]

        if name in self._factories:
            instance = self._factories[name]()
            self._services[name] = instance
            return instance

        return None

    def get_or_raise(self, name: str) -> object:
        """获取服务，不存在则抛出异常"""
        service = self.get(name)
        if service is None:
            raise KeyError(f"Service '{name}' not found")
        return service


class AppBuilder:
    """应用构建器

    Usage:
        app = (
            AppBuilder(config)
            .with_service("evaluation", EvaluationService)
            .with_router(api_router)
            .on_startup(init_model)
            .build()
        )
    """

    def __init__(self, config: Config):
        self.config = config
        self.container = ServiceContainer()
        self.routers: list = []
        self.route_handlers: list = []
        self._startup_hooks: list[Callable] = []
        self._shutdown_hooks: list[Callable] = []

    def with_service(self, name: str, instance: object) -> "AppBuilder":
        """注册服务实例"""
        self.container.register(name, instance)
        return self

    def with_service_factory(self, name: str, factory: Callable[[], object]) -> "AppBuilder":
        """注册服务工厂"""
        self.container.register_factory(name, factory)
        return self

    def with_router(self, router) -> "AppBuilder":
        """添加路由"""
        self.routers.append(router)
        return self

    def with_route_handler(self, handler) -> "AppBuilder":
        """添加路由处理器"""
        self.route_handlers.append(handler)
        return self

    def on_startup(self, hook: Callable) -> "AppBuilder":
        """注册启动钩子"""
        self._startup_hooks.append(hook)
        return self

    def on_shutdown(self, hook: Callable) -> "AppBuilder":
        """注册关闭钩子"""
        self._shutdown_hooks.append(hook)
        return self

    @asynccontextmanager
    async def _lifespan(self, app: Litestar):
        """应用生命周期管理"""
        import time
        import logging

        # 启动
        for hook in self._startup_hooks:
            result = hook()
            if hasattr(result, "__await__"):
                await result

        yield

        # 关闭（异步执行，不阻塞 event loop）
        logging.info("[lifespan] === SHUTDOWN PHASE START ===")
        t0 = time.perf_counter()
        for i, hook in enumerate(self._shutdown_hooks):
            name = hook.__name__ if hasattr(hook, '__name__') else str(hook)
            logging.info(f"[lifespan] Running shutdown hook {i+1}/{len(self._shutdown_hooks)}: {name}")
            result = hook()
            if hasattr(result, "__await__"):
                await result
            logging.info(f"[lifespan] Hook {i+1} done in {time.perf_counter()-t0:.3f}s")

        logging.info(f"[lifespan] All shutdown hooks done. Total: {time.perf_counter()-t0:.3f}s")

    def build(self) -> Litestar:
        """构建 Litestar 应用"""
        # 合并所有路由处理器
        all_handlers = [*self.routers, *self.route_handlers]

        return Litestar(
            path="/",
            route_handlers=all_handlers,
            lifespan=[self._lifespan],
            debug=self.config.verbose_enabled,
        )


def create_app(config: Config | None = None) -> Litestar:
    """创建应用实例（便捷函数）"""
    config = config or Config()

    from api import router as api_router
    return (
        AppBuilder(config)
        .with_router(api_router)
        .build()
    )
