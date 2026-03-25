"""WebSocket 图像评估服务

纯协议层，只处理 WebSocket 收发。
所有业务逻辑委托给 interactors。
"""

import asyncio
import json
import logging
from typing import Optional

from litestar import websocket, WebSocket
from litestar.exceptions import WebSocketException

from infrastructure.cancel import CancelReason, get_cancellation, CancelledError
from interactors import EvaluateImageInteractor

# 延迟导入避免循环依赖
# from apps.main import _evaluation_interactor 会在函数内使用

logger = logging.getLogger(__name__)


@websocket(path="/evaluate")
async def evaluate(socket: WebSocket) -> None:
    """WebSocket 图像评估端点

    协议:
    - 客户端发送: {"type": "submit", "images": [...], "verbose": bool}
    - 服务端发送: {"type": "result", "uid": "...", "path": "...", "rating": [...], "tags": [...]}
    - 服务端发送: {"type": "pong"}
    - 服务端发送: {"type": "stopped"} | {"type": "error", "message": "..."}
    - 客户端发送: {"type": "stop"}
    """
    ws = socket
    try:
        await ws.accept()
        logger.info(f"[ws] Connected: {ws.client}")
    except Exception as e:
        logger.error(f"WebSocket accept failed: {e}")
        return

    cancellation = get_cancellation()
    heartbeat_task: Optional[asyncio.Task] = None

    async def send_heartbeat():
        while True:
            await asyncio.sleep(5)
            try:
                await ws.send_json({"type": "pong"})
            except Exception:
                break

    try:
        while True:
            if cancellation.is_cancelled:
                await ws.send_json({"type": "stopped", "reason": "CANCELLED"})
                break

            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "submit":
                    heartbeat_task = asyncio.create_task(send_heartbeat())

                    images = message.get("images", [])
                    verbose = message.get("verbose", False)
                    logger.info(f"[ws] Received {len(images)} images")

                    # 委托给 interactor 处理
                    await _handle_batch(ws, images, verbose, cancellation)

                    await ws.send_json({"type": "complete"})
                    logger.info("[ws] Batch complete")

                elif msg_type == "stop":
                    logger.info("[ws] Stop requested")
                    cancellation.cancel(CancelReason.USER_REQUEST)
                    await ws.send_json({"type": "stopped"})
                    break

            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})

    except WebSocketException as e:
        logger.warning(f"[ws] Connection error: {e}")
    except CancelledError:
        await ws.send_json({"type": "stopped", "reason": "CANCELLED"})
    except Exception as e:
        logger.error(f"[ws] Unexpected error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        logger.info(f"[ws] Disconnected: {ws.client}")


async def _handle_batch(
    ws: WebSocket,
    images: list[dict],
    verbose: bool,
    cancellation,
) -> None:
    """处理批量请求

    从容器中获取 interactor，执行业务逻辑。
    实时流式返回每张图的结果。
    """
    from apps.main import _evaluation_interactor

    logger.info(f"[ws] _handle_batch start, images={len(images)}")

    # 检查 interactor 是否已初始化
    if _evaluation_interactor is None:
        logger.error("[ws] _evaluation_interactor is None")
        raise RuntimeError("Evaluation service not initialized")

    # 创建 batch interactor，复用已有的 processor 和 scheduler
    interactor = EvaluateImageInteractor(
        processor=_evaluation_interactor._processor,
        scheduler=_evaluation_interactor._scheduler,
        enable_metrics=True,
    )
    logger.info("[ws] Interactor created")

    results_sent = 0
    connection_closed = False
    send_queue: asyncio.Queue = asyncio.Queue()

    async def sender():
        """后台发送协程"""
        nonlocal results_sent, connection_closed
        while True:
            try:
                msg = await asyncio.wait_for(send_queue.get(), timeout=1.0)
                if msg is None:  # 终止信号
                    send_queue.task_done()
                    break
                try:
                    await ws.send_json(msg)
                    if msg.get("type") == "result":
                        results_sent += 1
                        if verbose:
                            logger.info(
                                f"[ws] Sent result {results_sent}: {msg.get('path')}"
                            )
                except Exception as e:
                    logger.warning(f"[ws] Send failed: {e}")
                    connection_closed = True
                finally:
                    send_queue.task_done()
            except asyncio.TimeoutError:
                continue

    def on_result(result):
        """同步回调 - 将结果放入发送队列"""
        if connection_closed:
            return
        send_queue.put_nowait(
            {
                "type": "result",
                "uid": result.uid,
                "path": result.path,
                "rating": result.rating,
                "tags": result.tags,
            }
        )

    def on_error(msg: str):
        """同步错误回调"""
        logger.warning(f"[ws] on_error: {msg}")
        send_queue.put_nowait({"type": "error", "message": msg})

    # 启动发送协程
    sender_task = asyncio.create_task(sender())

    try:
        # 执行批量处理
        logger.info("[ws] Calling interactor.execute...")
        perf = await interactor.execute(images, on_result, on_error, cancellation)
        logger.info(f"[ws] Interactor.execute returned, results_sent={results_sent}")

        # 等待队列清空
        await send_queue.join()

        # 发送性能信息
        if perf and not connection_closed:
            logger.info(
                f"[ws] Batch perf: {perf.total_time_ms:.1f}ms for {perf.valid_images} images"
            )
            try:
                await ws.send_json(
                    {
                        "type": "performance",
                        "data": {
                            "total_images": perf.total_images,
                            "valid_images": perf.valid_images,
                            "decode_time_ms": perf.decode_time_ms,
                            "inference_time_ms": perf.inference_time_ms,
                            "total_time_ms": perf.total_time_ms,
                            "system": perf.system_info,
                        },
                    }
                )
            except Exception as e:
                logger.warning(f"[ws] Failed to send performance: {e}")

    except Exception as e:
        logger.error(f"[ws] _handle_batch error: {e}", exc_info=True)
        raise
    finally:
        # 终止发送协程
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        logger.info(f"[ws] _handle_batch done, total_sent={results_sent}")
