#!/usr/bin/env python
"""批量图像标签请求 CLI 工具 (WebSocket 版本)

用法:
    python -m cli.tagging_request <image_paths...>

示例:
    python -m cli.tagging_request image1.jpg image2.png
    python -m cli.tagging_request ./images/*.jpg
"""
import argparse
import asyncio
import base64
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import websockets


class HeartbeatMonitor:
    """心跳监控器"""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_pong = time.time()
        self._timeout_triggered = False

    def start(self, on_timeout: callable):
        """启动心跳监控"""
        self._running = True
        self._timeout_triggered = False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor,
            args=(on_timeout,),
            daemon=True,
        )
        self._thread.start()

    def _monitor(self, on_timeout: callable):
        """监控循环"""
        while self._running and not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if self._running and not self._stop_event.is_set():
                elapsed = time.time() - self._last_pong
                if elapsed > self.interval * 3 and not self._timeout_triggered:
                    self._timeout_triggered = True
                    on_timeout()

    def pong_received(self):
        """收到 pong"""
        self._last_pong = time.time()
        self._timeout_triggered = False

    def stop(self):
        """停止监控"""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)


def load_image_as_base64(path: Path) -> Optional[str]:
    """将图片加载为 base64 编码"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error loading image {path}: {e}", file=sys.stderr)
        return None


def parse_image_paths(paths: list[str]) -> list[Path]:
    """解析图片路径"""
    image_paths: list[Path] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    for p in paths:
        # 对于 glob 模式，提取目录和模式
        if "*" in p:
            # glob 模式
            glob_path = Path(p)
            parent = glob_path.parent if glob_path.parent.name else Path(".")
            pattern = glob_path.name

            # 解析父目录
            if not parent.is_absolute():
                parent = Path.cwd() / parent
            parent = parent.resolve()

            if parent.exists():
                matched = list(parent.glob(pattern))
                for m in matched:
                    if m.is_file() and m.suffix.lower() in image_extensions:
                        image_paths.append(m)
            else:
                print(f"Warning: Directory not found: {parent}", file=sys.stderr)
        else:
            # 普通路径
            path = Path(p)
            if not path.is_absolute():
                path = Path.cwd() / path
            path = path.resolve()

            if path.is_file():
                if path.suffix.lower() in image_extensions:
                    image_paths.append(path)
            elif path.is_dir():
                for ext in image_extensions:
                    image_paths.extend(path.rglob(f"*{ext}"))
            else:
                print(f"Warning: {p} is not a valid file or directory", file=sys.stderr)

    return sorted(set(image_paths))


def print_result(result: dict) -> None:
    """打印评估结果"""
    # 处理批量结果
    if "results" in result:
        for item in result["results"]:
            path = item.get("path", item.get("uid", "?"))
            uid = item.get("uid", "?")
            rating = item.get("rating", ("?", 0))
            tags = item.get("tags", [])
            error = item.get("error")

            print(f"\n{'=' * 50}")
            print(f"Path: {path}")
            print(f"UID: {uid}")

            if error:
                print(f"Error: {error}")
                continue

            print(f"Rating: {rating[0]} ({rating[1]:.4f})")
            print(f"Tags ({len(tags)}):")
            for tag, score in sorted(tags, key=lambda x: -x[1])[:20]:
                print(f"  {score:.4f} {tag}")
            if len(tags) > 20:
                print(f"  ... and {len(tags) - 20} more tags")

    # 处理流式结果 (type: result) 或包含 rating 键的数据
    elif result.get("type") == "result" or "rating" in result:
        path = result.get("path", result.get("uid", "?"))
        uid = result.get("uid", "?")
        rating = result.get("rating", ("?", 0))
        tags = result.get("tags", [])

        print(f"\n{'=' * 50}")
        print(f"Path: {path}")
        print(f"UID: {uid}")
        print(f"Rating: {rating[0]} ({rating[1]:.4f})")
        print(f"Tags ({len(tags)}):")
        for tag, score in sorted(tags, key=lambda x: -x[1])[:20]:
            print(f"  {score:.4f} {tag}")
        if len(tags) > 20:
            print(f"  ... and {len(tags) - 20} more tags")

    elif "error" in result:
        print(f"\nError: {result['error']}", file=sys.stderr)


async def send_stop_signal(ws_url: str) -> None:
    """发送停止信号"""
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "stop"}))
            await ws.close()
        print("\n[WS] Stop signal sent")
    except Exception as e:
        print(f"\n[WS] Failed to send stop signal: {e}", file=sys.stderr)


async def run_client(
    ws_url: str,
    images: list[dict],
    verbose: bool = False,
) -> int:
    """运行 WebSocket 客户端"""
    heartbeat = HeartbeatMonitor(interval=5.0)
    stop_requested = threading.Event()
    start_time = time.perf_counter()

    def on_heartbeat_timeout():
        """心跳超时，设置停止标志"""
        if not stop_requested.is_set():
            print("\n[WS] Heartbeat timeout! Will stop after current batch...")
            stop_requested.set()

    try:
        async with websockets.connect(ws_url) as ws:
            print(f"[WS] Connected to {ws_url}")

            # 启动心跳监控
            heartbeat.start(on_heartbeat_timeout)

            # 发送图片数据
            print(f"[WS] Sending {len(images)} images...")
            await ws.send(json.dumps({
                "type": "submit",
                "images": images,
                "verbose": verbose,
            }))
            print("[WS] Images sent, waiting for results...")

            # 接收结果
            received_count = 0
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30.0)  # 增加超时时间
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "pong":
                        heartbeat.pong_received()
                        print("[WS] Heartbeat received")

                    elif msg_type == "result":
                        received_count += 1
                        elapsed = time.perf_counter() - start_time
                        print(f"[WS] Received result {received_count} (elapsed: {elapsed:.2f}s)")
                        print_result(data)

                    elif msg_type == "performance":
                        print(f"\n[WS] Performance: {data.get('data', data)}")

                    elif msg_type == "complete":
                        print(f"\n[WS] Server reports complete (received {received_count}/{len(images)} results)")
                        break

                    elif msg_type == "error":
                        print(f"\n[WS] Error: {data.get('message')}", file=sys.stderr)

                    elif msg_type == "stopped":
                        print("\n[WS] Task stopped by server")
                        break

                except asyncio.TimeoutError:
                    print(f"[WS] Waiting... (received: {received_count}/{len(images)})")
                    if stop_requested.is_set():
                        break
                    continue

            heartbeat.stop()
            elapsed = time.perf_counter() - start_time
            print(f"\n[WS] Completed. Total time: {elapsed:.2f}s")

    except websockets.exceptions.ConnectionClosed:
        print("\n[WS] Connection closed")
    except Exception as e:
        print(f"\n[WS] Error: {e}", file=sys.stderr)
        heartbeat.stop()
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量图像标签请求 CLI 工具 (WebSocket)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m cli.tagging_request image1.jpg image2.png
  python -m cli.tagging_request ./images/
  python -m cli.tagging_request "*.jpg" --url ws://localhost:8000/ws/evaluate
        """,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="图片路径、目录或 glob 模式",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/evaluate",
        help="WebSocket 服务器地址 (默认: ws://localhost:8000/ws/evaluate)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式",
    )

    args = parser.parse_args()

    # 解析图片路径
    image_paths = parse_image_paths(args.paths)
    if not image_paths:
        print("Error: No valid images found", file=sys.stderr)
        return 1

    if not image_paths:
        print(f"Error: No valid images found", file=sys.stderr)
        print(f"  Tried paths: {args.paths}", file=sys.stderr)
        print(f"  CWD: {Path.cwd()}", file=sys.stderr)
        return 1

    print(f"Found {len(image_paths)} images")
    print(f"WebSocket URL: {args.url}")

    # 加载图片
    print("\nLoading images...")
    images = []
    for i, path in enumerate(image_paths):
        b64 = load_image_as_base64(path)
        if b64:
            images.append({
                "path": str(path),
                "data": b64,
            })
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(image_paths)} images")

    if not images:
        print("Error: No valid images could be loaded", file=sys.stderr)
        return 1

    print(f"\nTotal: {len(images)} images ready")

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n[CLI] Interrupt received, sending stop signal...")
        asyncio.get_event_loop().run_until_complete(send_stop_signal(args.url))
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行客户端
    return asyncio.run(run_client(args.url, images, verbose=args.verbose))


if __name__ == "__main__":
    sys.exit(main())
