#!/usr/bin/env python3
"""基本的 OCR 使用示例。"""
from __future__ import annotations

import argparse
from pathlib import Path

from ddddocr import DdddOcr, DdddOcrInputError, InvalidImageError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DdddOcr 本地识别示例")
    parser.add_argument("image", type=Path, help="待识别图片路径")
    parser.add_argument(
        "--probability",
        action="store_true",
        help="输出每一列的概率分布",
    )
    parser.add_argument(
        "--colors",
        nargs="*",
        default=(),
        help="可选：指定需要保留的颜色，例如 red green",
    )
    parser.add_argument(
        "--beta",
        action="store_true",
        help="使用新版 beta OCR 模型",
    )
    parser.add_argument(
        "--old",
        action="store_true",
        help="使用旧版 OCR 模型 (beta=False 时有效)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"文件 {args.image} 不存在")

    ocr = DdddOcr(ocr=True, det=False, beta=args.beta, old=args.old, show_ad=False)

    try:
        data = args.image.read_bytes()
        if args.probability:
            result = ocr.classification(
                data,
                probability=True,
                colors=list(args.colors),
            )
        else:
            result = ocr.classification(
                data,
                probability=False,
                colors=list(args.colors),
            )
    except (DdddOcrInputError, InvalidImageError) as exc:
        parser.error(str(exc))
        return 1

    print("识别结果:")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
