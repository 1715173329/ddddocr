# coding=utf-8
"""
工具函数模块
提供图像处理、异常处理、输入验证等工具函数
"""

import importlib.util
from pathlib import Path

# 从新的模块化结构导入
from .image_io import png_rgba_black_preprocess
from .exceptions import DDDDOCRError, ModelLoadError, ImageProcessError
from .validators import validate_image_input, validate_model_config

# 从父目录的 utils.py 文件导入（保持向后兼容）
_utils_legacy_file = Path(__file__).parent.parent / "utils.py"
_spec = importlib.util.spec_from_file_location("ddddocr._utils_legacy_compat", _utils_legacy_file)
_utils_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_legacy)

# 导入旧版 utils.py 中的常量和函数
ALLOWED_IMAGE_FORMATS = _utils_legacy.ALLOWED_IMAGE_FORMATS
MAX_IMAGE_BYTES = _utils_legacy.MAX_IMAGE_BYTES
MAX_IMAGE_SIDE = _utils_legacy.MAX_IMAGE_SIDE
DdddOcrInputError = _utils_legacy.DdddOcrInputError
InvalidImageError = _utils_legacy.InvalidImageError
TypeError = _utils_legacy.TypeError
base64_to_image = _utils_legacy.base64_to_image
get_img_base64 = _utils_legacy.get_img_base64

__all__ = [
    # 新模块化结构的导出
    'png_rgba_black_preprocess',
    'DDDDOCRError',
    'ModelLoadError',
    'ImageProcessError',
    'validate_image_input',
    'validate_model_config',
    # 从旧 utils.py 导入的兼容性导出
    'ALLOWED_IMAGE_FORMATS',
    'MAX_IMAGE_BYTES',
    'MAX_IMAGE_SIDE',
    'DdddOcrInputError',
    'InvalidImageError',
    'TypeError',
    'base64_to_image',
    'get_img_base64',
]
