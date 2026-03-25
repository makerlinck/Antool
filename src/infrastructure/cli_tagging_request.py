#!/usr/bin/env python
"""批量图像标签请求 CLI 工具

用法:
    python -m infrastructure.cli_tagging_request <image_paths...>

示例:
    python -m infrastructure.cli_tagging_request image1.jpg image2.png
    python -m infrastructure.cli_tagging_request ./images/*.jpg
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests


def load_image_as_base64(path: Path) -> Optional[str]:
    """将图片加载为 base64 编码"""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error loading image {path}: {e}", file=sys.stderr)
        return None


def parse_image_paths(paths: list[str]) -> list[Path]:
    """解析图片路径

    支持：
    - 单个文件路径
    - 目录路径（递归查找图片）
    - glob 模式（如 *.jpg）
    """
    image_paths: list[Path] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                image_paths.append(path)
        elif path.is_dir():
            for ext in image_extensions:
                image_paths.extend(path.rglob(f"*{ext}"))
        elif "*" in str(path):
            # glob 模式
            image_paths.extend(Path(".").glob(str(path)))
        else:
            print(f"Warning: {p} is not a valid file or directory", file=sys.stderr)

    return sorted(set(image_paths))


def send_request(
    url: str,
    images: list[dict],
    timeout: int = 300,
) -> Optional[dict]:
    """发送请求到 API"""
    try:
        response = requests.post(
            f"{url}/api/v1/evaluate",
            json={"images": images},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Error: Request timeout", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON response", file=sys.stderr)
        return None


def print_result(result: dict, verbose: bool = False) -> None:
    """打印评估结果"""
    if "results" in result:
        for item in result["results"]:
            uid = item.get("uid", "?")
            rating = item.get("rating", ("?", 0))
            tags = item.get("tags", [])

            print(f"\n{'=' * 50}")
            print(f"UID: {uid}")
            print(f"Rating: {rating[0]} ({rating[1]:.4f})")
            print(f"Tags ({len(tags)}):")
            for tag, score in sorted(tags, key=lambda x: -x[1])[:20]:
                print(f"  {score:.4f} {tag}")
            if len(tags) > 20:
                print(f"  ... and {len(tags) - 20} more tags")

    elif "error" in result:
        print(f"\nError: {result['error']}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量图像标签请求 CLI 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m infrastructure.cli_tagging_request image1.jpg image2.png
  python -m infrastructure.cli_tagging_request ./images/
  python -m infrastructure.cli_tagging_request "*.jpg" --url http://localhost:8000
        """,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="图片路径、目录或 glob 模式",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API 服务器地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="标签置信度阈值 (默认: 0.5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间，秒 (默认: 300)",
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

    print(f"Found {len(image_paths)} images")
    print(f"API URL: {args.url}")

    # 加载图片
    print("\nLoading images...")
    images = []
    for i, path in enumerate(image_paths):
        b64 = load_image_as_base64(path)
        if b64:
            images.append(
                {
                    "path": str(path),
                    "data": b64,
                }
            )
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(image_paths)} images")

    if not images:
        print("Error: No valid images could be loaded", file=sys.stderr)
        return 1

    print(f"\nSending request with {len(images)} images...")

    # 发送请求
    start_time = time.perf_counter()
    result = send_request(
        args.url,
        images,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - start_time

    if result is None:
        return 1

    # 打印结果
    print(f"\nRequest completed in {elapsed:.2f}s")
    print_result(result, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
