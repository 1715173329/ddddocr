#!/usr/bin/env python3
"""
目标检测示例（det=True）

用法:
  python examples/detector.py <图片路径>
  python examples/detector.py <图片路径> --use-gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddddocr import DdddOcr, DdddOcrInputError, InvalidImageError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DdddOcr 目标检测示例")
    parser.add_argument("image", type=Path, help="待检测图片路径")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="如果环境支持，启用 GPU 推理",
    )
    return parser


def format_boxes(boxes: List[List[int]]) -> str:
    if not boxes:
        return "未检测到目标"
    lines = ["检测到的矩形框 (x_min, y_min, x_max, y_max):"]
    for idx, (x_min, y_min, x_max, y_max) in enumerate(boxes, 1):
        lines.append(f"  #{idx}: ({x_min}, {y_min}, {x_max}, {y_max})")
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"文件 {args.image} 不存在")

    detector = DdddOcr(ocr=False, det=True, use_gpu=args.use_gpu, show_ad=False)

    try:
        data = args.image.read_bytes()
        boxes = detector.detection(img_bytes=data)
    except (DdddOcrInputError, InvalidImageError) as exc:
        parser.error(str(exc))
        return 1

    print(format_boxes(boxes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
