"""应用入口

组装所有依赖并启动应用。
"""

import logging

from litestar import Litestar, get

from configs import Config
from core.interfaces.evaluation import ImageProcessor, TaskScheduler
from interactors import ImageEvaluationInteractor
from infrastructure.evaluations.model_loader import SharedModelLoader
from infrastructure.evaluations.scheduler import BatchScheduler
from infrastructure.evaluations.processor import ImageEvaluationProcessor
from infrastructure.evaluations.processor_cpu import CPUOptimizedProcessor
from infrastructure.cancel import CancelReason, get_cancellation
from infrastructure.metrics import get_metrics

from .builder import AppBuilder

# 模块级日志记录器
logger: logging.Logger
_app_config: Config | None = None
_evaluation_interactor: ImageEvaluationInteractor | None = None


# ============================================================================
# HTTP 端点
# ============================================================================


@get("/health")
async def health() -> dict:
    """健康检查"""
    return {"status": "OK"}


# ============================================================================
# 依赖组装
# ============================================================================


def create_evaluation_interactor(config: Config) -> ImageEvaluationInteractor:
    """创建评估交互器

    按照依赖注入原则组装：
    1. 基础设施层：模型加载器（共享内存）
    2. 适配器层：处理器、调度器
    3. 用例层：评估用例
    """
    global logger

    # 初始化指标收集器
    metrics = get_metrics()
    metrics.verbose = config.verbose_enabled

    logger.info("Initializing evaluation service...")

    # 1. 模型加载（共享内存、同步加载）
    metrics.model_load_start()
    logger.info(f"Loading model from: {config.model_path / 'v3-20211112-sgd-e28'}")
    model_loader = SharedModelLoader(config.model_path / "v3-20211112-sgd-e28")
    model_loader.load()
    metrics.model_load_end()

    load_duration = metrics.model_load_duration_ms
    logger.info(
        f"Model loaded. Tags count: {len(model_loader.tags)}, Duration: {load_duration:.2f}ms"
    )

    # 2. 处理器（预处理 + 推理 + 后处理）
    # 使用 CPU 优化版本（单张推理 + TF 预处理 + XLA）
    processor: ImageProcessor = CPUOptimizedProcessor(
        model=model_loader.model,
        tags=model_loader.tags,
        threshold=0.5,
        batch_size=config.batch_size,
    )
    logger.debug("CPU optimized processor initialized")

    # 3. 模型预热（触发 XLA 编译）
    logger.info("Warming up model...")
    from core.entities import ImageTask
    import numpy as np

    dummy_input = np.random.rand(256, 256, 3).astype(np.float32)
    processor.process([ImageTask(image=dummy_input, uid="warmup")])
    logger.info("Model warmup complete")

    # 4. 调度器（线程池 / 进程池）
    scheduler: TaskScheduler = BatchScheduler(
        thread_threshold=config.batch_thread_threshold,
        max_workers=config.max_concurrent,
        batch_size=config.batch_size,
    )
    logger.debug(
        f"Batch scheduler initialized: workers={config.max_concurrent}, batch_size={config.batch_size}"
    )

    # 5. 用例（复用 processor，避免重复创建）
    logger.info("Evaluation service initialized successfully")
    return ImageEvaluationInteractor(
        processor=processor,
        scheduler=scheduler,
        enable_metrics=True,
        max_workers=config.max_concurrent,
    )


def build_app(config: Config | None = None) -> Litestar:
    """构建应用"""
    global logger, _app_config, _evaluation_interactor

    # 1. 初始化配置
    if config is None:
        config = Config()

    _app_config = config

    # 2. 初始化日志系统
    logger = config.init_logging()
    logger.info("=" * 50)
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    logger.info(f"Log level: {config.min_log_level.name}")
    logger.info(f"Verbose: {config.verbose_enabled}")
    logger.info("=" * 50)

    # 3. 导入路由
    # from apps.api import router as api_router
    from apps.ws import ws_router

    # 4. 创建用例
    _evaluation_interactor = create_evaluation_interactor(config)

    # 6. 构建应用
    app = (
        AppBuilder(config)
        .with_service("evaluation", _evaluation_interactor)
        # .with_router(api_router)
        .with_router(ws_router)
        .with_route_handler(health)
        .on_startup(lambda: logger.info(f"✅ {config.app_name} HTTP server ready"))
        .on_shutdown(_on_shutdown)
        .build()
    )

    logger.info(f"Application built successfully")
    return app


async def _on_shutdown() -> None:
    """应用关闭时的回调

    输出指标摘要并取消所有正在处理的任务。
    """
    global logger
    import time

    t0 = time.perf_counter()

    cancellation = get_cancellation()
    metrics = get_metrics()

    logger.info("=" * 50)
    logger.info("Application shutting down...")

    # 触发取消
    cancellation.cancel(CancelReason.SHUTDOWN)
    t1 = time.perf_counter()
    logger.info(f"[{t1-t0:.3f}s] All pending tasks cancelled")

    # 输出指标摘要
    summary = metrics.get_summary()
    logger.info("--- Metrics Summary ---")
    if summary.get("model_load_duration_ms"):
        logger.info(f"  Model load duration: {summary['model_load_duration_ms']:.2f}ms")
    if summary.get("avg_image_latency_ms"):
        logger.info(f"  Avg image latency: {summary['avg_image_latency_ms']:.2f}ms")
    logger.info(f"  Total requests: {summary.get('total_requests', 0)}")
    logger.info("=" * 50)

    # 重置取消状态
    cancellation.reset()
    logger.info(f"[{time.perf_counter()-t0:.3f}s] Shutdown complete")


# 应用实例（懒加载）
_app_instance: Litestar | None = None


def get_app() -> Litestar:
    global _app_instance
    if _app_instance is None:
        _app_instance = build_app()
    return _app_instance


# 别名供 uvicorn 使用
app = get_app()


if __name__ == "__main__":
    import uvicorn

    _app = get_app()
    print("Antool API server ready")
    uvicorn.run(_app, host="127.0.0.1", port=8000)
    import uvicorn

    # 注册关闭时取消所有任务
    cancellation = get_cancellation()
    cancellation.add_listener(
        lambda reason: logger.warning(f"Cancellation triggered: {reason}")
    )

    uvicorn.run("apps.main:app", host="127.0.0.1", port=8000)
