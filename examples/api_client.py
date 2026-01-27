#!/usr/bin/env python3
"""
调用 API 服务的示例（需要先启动 ddddocr api）

用法:
  python -m ddddocr api
  python examples/api_client.py <图片路径>
  python examples/api_client.py <图片路径> --endpoint http://127.0.0.1:8000/ocr
  python examples/api_client.py <图片路径> --probability
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DdddOcr API 示例客户端")
    parser.add_argument("image", type=Path, help="待识别图片路径")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/ocr",
        help="API 接口地址，默认 http://127.0.0.1:8000/ocr",
    )
    parser.add_argument(
        "--probability",
        action="store_true",
        help="是否请求概率输出",
    )
    return parser


def payload_from_image(path: Path, probability: bool) -> Dict[str, Any]:
    data = base64.b64encode(path.read_bytes()).decode()
    return {
        "image": data,
        "probability": probability,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"文件 {args.image} 不存在")

    payload = payload_from_image(args.image, args.probability)
    try:
        response = requests.post(args.endpoint, json=payload, timeout=15)
    except requests.RequestException as exc:
        parser.error(f"请求 API 失败: {exc}")
        return 1

    if response.status_code != 200:
        parser.error(f"API 返回错误: {response.status_code} {response.text}")
        return 1

    print("API 返回结果:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
