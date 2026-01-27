# coding=utf-8
import base64
import binascii
import warnings

warnings.filterwarnings('ignore')
import io
import json
import os
import pathlib
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import onnxruntime
from PIL import Image, ImageChops
import numpy as np
import cv2

from .charsets import CHARSET_OLD, CHARSET_BETA
from .utils import (
    ALLOWED_IMAGE_FORMATS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    DdddOcrInputError,
    InvalidImageError,
    TypeError,
    base64_to_image,
    _coerce_bool,
    _coerce_int,
    _coerce_positive_int,
    _ensure_file_exists,
    png_rgba_black_preprocess,
)

onnxruntime.set_default_logger_severity(3)


class DdddOcr(object):
    def __init__(self, ocr: bool = True, det: bool = False, old: bool = False, beta: bool = False,
                 use_gpu: bool = False,
                 device_id: int = 0, show_ad=True, import_onnx_path: str = "", charsets_path: str = "",
                 max_image_bytes: Optional[Union[int, str]] = None,
                 max_image_side: Optional[Union[int, str]] = None):
        ocr = _coerce_bool(ocr, 'ocr')
        det = _coerce_bool(det, 'det')
        old = _coerce_bool(old, 'old')
        beta = _coerce_bool(beta, 'beta')
        use_gpu = _coerce_bool(use_gpu, 'use_gpu')
        show_ad = _coerce_bool(show_ad, 'show_ad')
        device_id = _coerce_int(device_id, 'device_id')
        if import_onnx_path and not isinstance(import_onnx_path, str):
            raise DdddOcrInputError("import_onnx_path 必须是字符串")
        if charsets_path and not isinstance(charsets_path, str):
            raise DdddOcrInputError("charsets_path 必须是字符串")

        if max_image_bytes is None:
            max_image_bytes = MAX_IMAGE_BYTES
        else:
            max_image_bytes = _coerce_positive_int(max_image_bytes, 'max_image_bytes')
        if max_image_side is None:
            max_image_side = MAX_IMAGE_SIDE
        else:
            max_image_side = _coerce_positive_int(max_image_side, 'max_image_side')

        self._max_image_bytes = max_image_bytes
        self._max_image_side = max_image_side
        self._allowed_formats = ALLOWED_IMAGE_FORMATS

        if show_ad:
            print("欢迎使用ddddocr，本项目专注带动行业内卷，个人博客:wenanzhe.com")
            print("训练数据支持来源于:http://146.56.204.113:19199/preview")
            print("爬虫框架feapder可快速一键接入，快速开启爬虫之旅：https://github.com/Boris-code/feapder")
            print(
                "谷歌reCaptcha验证码 / hCaptcha验证码 / funCaptcha验证码商业级识别接口：https://yescaptcha.com/i/NSwk7i")
        if not hasattr(Image, 'ANTIALIAS'):
            setattr(Image, 'ANTIALIAS', Image.LANCZOS)
        self.use_import_onnx = False
        self.__word = False
        self.__resize = []
        self.__charset_range = []
        self.__valid_charset_range_index = []  # 指定字符对应的有效索引
        self.__channel = 1
        if import_onnx_path != "":
            _ensure_file_exists(import_onnx_path, "自定义模型路径")
            _ensure_file_exists(charsets_path, "自定义字符集路径")
            det = False
            ocr = False
            self.__graph_path = import_onnx_path
            try:
                with open(charsets_path, 'r', encoding="utf-8") as f:
                    info = json.loads(f.read())
            except Exception as exc:
                raise DdddOcrInputError("读取自定义字符集文件失败") from exc
            required_keys = {'charset', 'word', 'image', 'channel'}
            if not required_keys.issubset(info.keys()):
                raise DdddOcrInputError("自定义字符集文件缺少必要字段 charset/word/image/channel")
            self.__charset = info['charset']
            self.__word = info['word']
            self.__resize = info['image']
            self.__channel = info['channel']
            self.use_import_onnx = True

        if det:
            ocr = False
            self.__graph_path = os.path.join(os.path.dirname(__file__), 'common_det.onnx')
            self.__charset = []
        if ocr:
            if not beta:
                self.__graph_path = os.path.join(os.path.dirname(__file__), 'common_old.onnx')
                self.__charset = CHARSET_OLD
            else:
                self.__graph_path = os.path.join(os.path.dirname(__file__), 'common.onnx')
                self.__charset = CHARSET_BETA
        self.det = det
        if use_gpu:
            self.__providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cuda_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
            ]
        else:
            self.__providers = [
                'CPUExecutionProvider',
            ]
        if ocr or det or self.use_import_onnx:
            self.__ort_session = onnxruntime.InferenceSession(self.__graph_path, providers=self.__providers)

    def _ensure_image_bytes(self, data: Union[bytes, bytearray], field_name: str = 'image') -> bytes:
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, (bytes, bytearray)):
            raise DdddOcrInputError(f"{field_name} 必须是二进制数据")
        if len(data) == 0:
            raise InvalidImageError(f"{field_name} 内容为空")
        if len(data) > self._max_image_bytes:
            raise InvalidImageError(
                f"{field_name} 大小超过 {self._max_image_bytes // 1024}KB 限制"
            )
        return data

    def _validate_pil_image(self, image: Image.Image) -> Image.Image:
        try:
            image.load()
        except Exception as exc:
            raise InvalidImageError("图片内容损坏") from exc
        width, height = image.size
        if width <= 0 or height <= 0:
            raise InvalidImageError("图片尺寸异常")
        if width > self._max_image_side or height > self._max_image_side:
            raise InvalidImageError(
                f"图片最长边不能超过 {self._max_image_side}px (当前 {width}x{height})"
            )
        fmt = (image.format or '').upper()
        if fmt and fmt not in self._allowed_formats:
            raise InvalidImageError(f"不支持的图片格式: {fmt}")
        return image

    def _load_image(self, img: Union[bytes, str, pathlib.PurePath, Image.Image]) -> Image.Image:
        try:
            if isinstance(img, Image.Image):
                candidate = img.copy()
            elif isinstance(img, (bytes, bytearray)):
                data = self._ensure_image_bytes(img)
                candidate = Image.open(io.BytesIO(data))
            elif isinstance(img, pathlib.PurePath):
                candidate = Image.open(str(img))
            elif isinstance(img, str):
                if os.path.exists(img):
                    candidate = Image.open(img)
                else:
                    candidate = base64_to_image(img)
            else:
                raise DdddOcrInputError("未知图片输入类型，仅支持 bytes/base64/path/Image")
        except InvalidImageError:
            raise
        except Exception as exc:
            raise InvalidImageError("无法解析图片输入") from exc
        return self._validate_pil_image(candidate)

    def _normalize_colors(self, colors: Optional[Iterable[str]]) -> Sequence[str]:
        if colors is None:
            return []
        if isinstance(colors, str):
            colors_iterable = [colors]
        elif isinstance(colors, Iterable):
            colors_iterable = colors
        else:
            raise DdddOcrInputError("colors 需要是可迭代对象")
        normalized = []
        for color in colors_iterable:
            if not isinstance(color, str):
                raise DdddOcrInputError("colors 中的元素必须是字符串")
            stripped = color.strip()
            if stripped:
                normalized.append(stripped)
        return normalized

    def _normalize_custom_ranges(self, custom_ranges: Optional[Any]) -> Optional[Mapping[str, Sequence[Sequence[int]]]]:
        if custom_ranges is None:
            return None
        if not isinstance(custom_ranges, Mapping):
            raise DdddOcrInputError("custom_color_ranges 必须是 dict 类型")
        normalized = {}
        for key, value in custom_ranges.items():
            if not isinstance(key, str):
                raise DdddOcrInputError("custom_color_ranges 的键必须是字符串")
            if not isinstance(value, Iterable):
                raise DdddOcrInputError("custom_color_ranges 的值必须是可迭代对象")
            ranges_list = []
            for segment in value:
                if not isinstance(segment, Iterable):
                    raise DdddOcrInputError("custom_color_ranges 的区间需要为长度为3的列表")
                segment_list = list(segment)
                if len(segment_list) != 3:
                    raise DdddOcrInputError("HSV 区间需要包含 3 个整数")
                int_segment = []
                for idx, item in enumerate(segment_list):
                    try:
                        int_value = int(item)
                    except (TypeError, ValueError):
                        raise DdddOcrInputError("HSV 区间值必须是整数") from None
                    if not 0 <= int_value <= 255:
                        raise DdddOcrInputError("HSV 区间的取值需要介于 0-255")
                    int_segment.append(int_value)
                ranges_list.append(int_segment)
            normalized[key.strip()] = ranges_list
        return normalized

    def _decode_base64_bytes(self, img_base64: str, field_name: str = 'image') -> bytes:
        if not isinstance(img_base64, str) or not img_base64.strip():
            raise DdddOcrInputError(f"{field_name} base64 内容不能为空")
        try:
            img_bytes = base64.b64decode(img_base64, validate=True)
        except binascii.Error as exc:
            raise DdddOcrInputError(f"{field_name} base64 内容非法") from exc
        return self._ensure_image_bytes(img_bytes, field_name)

    def color_filter(self, pil_image, colors, custom_ranges=None):
        # 预定义颜色阈值范围（HSV颜色空间）
        color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)],
            'blue': [(90, 50, 50), (130, 255, 255)],
            'yellow': [(20, 100, 100), (35, 255, 255)],
            'orange': [(10, 100, 100), (20, 255, 255)],
            'purple': [(130, 50, 50), (170, 255, 255)],
            'pink': [(140, 50, 50), (170, 255, 255)],
            'brown': [(0, 50, 50), (20, 255, 150)]
        }

        # 将Pillow图像转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 合并自定义颜色范围
        if custom_ranges:
            color_ranges.update(custom_ranges)

        # 将图像转换为HSV颜色空间
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 创建掩码
        final_mask = np.zeros_like(hsv_image[:, :, 0])

        # 处理每种颜色
        for color in colors:
            if color.lower() in color_ranges:
                color_range = color_ranges[color.lower()]

                # 处理多个颜色范围（主要针对红色）
                if len(color_range) > 2:
                    for lower, upper in zip(color_range[::2], color_range[1::2]):
                        lower_bound = np.array(lower)
                        upper_bound = np.array(upper)
                        temp_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
                        final_mask = cv2.bitwise_or(final_mask, temp_mask)
                else:
                    lower_bound = np.array(color_range[0])
                    upper_bound = np.array(color_range[1])
                    temp_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
                    final_mask = cv2.bitwise_or(final_mask, temp_mask)

        # 应用掩码
        result = cv2.bitwise_and(cv_image, cv_image, mask=final_mask)

        # 将背景设为白色
        white_background = np.ones_like(cv_image) * 255
        result = cv2.bitwise_or(result, white_background, mask=cv2.bitwise_not(final_mask))

        # 将OpenCV图像转换回Pillow图像
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)

        return result_pil

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        return self.multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr)

    def get_bbox(self, image_bytes):
        image_bytes = self._ensure_image_bytes(image_bytes, 'det_image')
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("无法解析检测图片数据")

        im, ratio = self.preproc(img, (416, 416))
        ort_inputs = {self.__ort_session.get_inputs()[0].name: im[None, :, :, :]}
        output = self.__ort_session.run(None, ort_inputs)
        predictions = self.demo_postprocess(output[0], (416, 416))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        pred = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        try:
            final_boxes = pred[:, :4].tolist()
            result = []
            for b in final_boxes:
                if b[0] < 0:
                    x_min = 0
                else:
                    x_min = int(b[0])
                if b[1] < 0:
                    y_min = 0
                else:
                    y_min = int(b[1])
                if b[2] > img.shape[1]:
                    x_max = int(img.shape[1])
                else:
                    x_max = int(b[2])
                if b[3] > img.shape[0]:
                    y_max = int(img.shape[0])
                else:
                    y_max = int(b[3])
                result.append([x_min, y_min, x_max, y_max])
        except Exception as e:
            return []
        return result

    def set_ranges(self, charset_range):
        if isinstance(charset_range, int):
            if charset_range == 0:
                # 数字
                self.__charset_range = list("0123456789")
            elif charset_range == 1:
                # 小写英文
                self.__charset_range = list("abcdefghijklmnopqrstuvwxyz")
            elif charset_range == 2:
                # 大写英文
                self.__charset_range = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif charset_range == 3:
                # 混合英文
                self.__charset_range = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif charset_range == 4:
                # 小写英文+数字
                self.__charset_range = list("abcdefghijklmnopqrstuvwxyz") + list(
                    "0123456789")
            elif charset_range == 5:
                # 大写英文+数字
                self.__charset_range = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list(
                    "0123456789")
            elif charset_range == 6:
                # 混合大小写+数字
                self.__charset_range = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + list(
                    "0123456789")
            elif charset_range == 7:
                # 除去英文，数字
                delete_range = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
                self.__charset_range = [item for item in self.__charset if item not in delete_range]
        elif isinstance(charset_range, str):
            charset_range_list = list(charset_range)
            self.__charset_range = charset_range_list
        elif isinstance(charset_range, (list, tuple, set)):
            charset_range_list = []
            for item in charset_range:
                if not isinstance(item, str) or not item:
                    raise DdddOcrInputError("charset_range 列表中只能包含非空字符串")
                charset_range_list.append(item)
            self.__charset_range = charset_range_list
        else:
            raise TypeError("暂时不支持该类型数据的输入")

        # 去重
        self.__charset_range = list(set(self.__charset_range)) + [""]
        # 根据指定字符获取对应的索引 
        valid_charset_range_index = []
        if len(self.__charset_range) > 0:
            for item in self.__charset_range:
                if item in self.__charset:
                    valid_charset_range_index.append(self.__charset.index(item))
                else:
                    # 未知字符没有索引，直接忽略
                    pass
        self.__valid_charset_range_index = valid_charset_range_index

    def classification(self, img, png_fix: bool = False, probability: bool = False,
                      colors: Optional[Iterable[str]] = None,
                      custom_color_ranges: Optional[Mapping[str, Sequence[Sequence[int]]]] = None):
        if self.det:
            raise TypeError("当前识别类型为目标检测")
        png_fix = _coerce_bool(png_fix, 'png_fix')
        probability = _coerce_bool(probability, 'probability')
        image = self._load_image(img)
        normalized_colors = self._normalize_colors(colors)
        normalized_custom_ranges = self._normalize_custom_ranges(custom_color_ranges)
        if normalized_colors or normalized_custom_ranges:

            image = self.color_filter(image, normalized_colors, normalized_custom_ranges)
        if png_fix:
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                image = png_rgba_black_preprocess(image.convert("RGBA"))
        if not self.use_import_onnx:
            image = image.resize((int(image.size[0] * (64 / image.size[1])), 64), Image.ANTIALIAS).convert('L')
        else:
            if self.__resize[0] == -1:
                if self.__word:
                    image = image.resize((self.__resize[1], self.__resize[1]), Image.ANTIALIAS)
                else:
                    image = image.resize((int(image.size[0] * (self.__resize[1] / image.size[1])), self.__resize[1]),
                                         Image.ANTIALIAS)
            else:
                image = image.resize((self.__resize[0], self.__resize[1]), Image.ANTIALIAS)
            if self.__channel == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=0) / 255.
        if not self.use_import_onnx:
            image = (image - 0.5) / 0.5
        else:
            if self.__channel == 1:
                image = (image - 0.456) / 0.224
            else:
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image = image[0]
                image = image.transpose((2, 0, 1))

        ort_inputs = {'input1': np.array([image]).astype(np.float32)}
        ort_outs = self.__ort_session.run(None, ort_inputs)
        result = []

        last_item = 0

        if self.__word:
            for item in ort_outs[1]:
                result.append(self.__charset[item])
            return ''.join(result)
        else:
            if not self.use_import_onnx:
                # 概率输出仅限于使用官方模型
                if probability:
                    ort_outs = ort_outs[0]
                    ort_outs = np.exp(ort_outs) / np.sum(np.exp(ort_outs))
                    ort_outs_sum = np.sum(ort_outs, axis=2)
                    ort_outs_probability = np.empty_like(ort_outs)
                    for i in range(ort_outs.shape[0]):
                        ort_outs_probability[i] = ort_outs[i] / ort_outs_sum[i]
                    ort_outs_probability = np.squeeze(ort_outs_probability).tolist()
                    result = {}
                    if len(self.__charset_range) == 0:
                        # 返回全部
                        result['charsets'] = self.__charset
                        result['probability'] = ort_outs_probability
                    else:
                        result['charsets'] = self.__charset_range
                        valid_charset_range_index = self.__valid_charset_range_index
                        probability_result = []
                        for item in ort_outs_probability:
                            probability_result.append([item[i] for i in valid_charset_range_index])
                        result['probability'] = probability_result
                    return result
                else:
                    if len(self.__charset_range) == 0:
                        # 没有指定特定的字符集合，直接获取结果
                        last_item = 0
                        argmax_result = np.squeeze(np.argmax(ort_outs[0], axis=2))
                        for item in argmax_result:
                            if item == last_item:
                                continue
                            else:
                                last_item = item
                            if item != 0:
                                result.append(self.__charset[item])
                    else:
                        # 指定了特定的字符集合
                        last_item = 0
                        valid_charset_range_index = self.__valid_charset_range_index
                        for row in np.squeeze(ort_outs[0]):
                            # 仅在指定字符集合中寻找最大值
                            idx = np.argmax(row[list(valid_charset_range_index)])
                            if idx == last_item:
                                continue
                            else:
                                last_item = idx
                            result.append(self.__charset[valid_charset_range_index[idx]])
                    return ''.join(result)

            else:
                last_item = 0
                for item in ort_outs[0][0]:
                    if item == last_item:
                        continue
                    else:
                        last_item = item
                    if item != 0:
                        result.append(self.__charset[item])
                return ''.join(result)

    def detection(self, img_bytes: bytes = None, img_base64: str = None):
        if not self.det:
            raise TypeError("当前识别类型为文字识别")
        if img_bytes is not None:
            data = self._ensure_image_bytes(img_bytes, 'img_bytes')
        elif img_base64:
            data = self._decode_base64_bytes(img_base64, 'img_base64')
        else:
            raise DdddOcrInputError("需要提供 img_bytes 或 img_base64")
        result = self.get_bbox(data)
        return result

    def get_target(self, img_bytes: bytes = None):
        img_bytes = self._ensure_image_bytes(img_bytes, 'target_bytes')
        image = Image.open(io.BytesIO(img_bytes))
        w, h = image.size
        starttx = 0
        startty = 0
        end_x = 0
        end_y = 0
        for x in range(w):
            for y in range(h):
                p = image.getpixel((x, y))
                if p[-1] == 0:
                    if startty != 0 and end_y == 0:
                        end_y = y

                    if starttx != 0 and end_x == 0:
                        end_x = x
                else:
                    if startty == 0:
                        startty = y
                        end_y = 0
                    else:
                        if y < startty:
                            startty = y
                            end_y = 0
            if starttx == 0 and startty != 0:
                starttx = x
            if end_y != 0:
                end_x = x
        return image.crop([starttx, startty, end_x, end_y]), starttx, startty

    def slide_match(self, target_bytes: bytes = None, background_bytes: bytes = None, simple_target: bool = False,
                    flag: bool = False):
        if target_bytes is None or background_bytes is None:
            raise DdddOcrInputError("需要提供 target_bytes 和 background_bytes")
        target_bytes = self._ensure_image_bytes(target_bytes, 'target_bytes')
        background_bytes = self._ensure_image_bytes(background_bytes, 'background_bytes')
        if not simple_target:
            try:
                target, target_x, target_y = self.get_target(target_bytes)
                target = cv2.cvtColor(np.asarray(target), cv2.IMREAD_ANYCOLOR)
            except SystemError as e:
                # SystemError: tile cannot extend outside image
                if flag:
                    raise e
                return self.slide_match(target_bytes=target_bytes, background_bytes=background_bytes,
                                        simple_target=True, flag=True)
        else:
            target = cv2.imdecode(np.frombuffer(target_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
            target_y = 0
            target_x = 0

        if target is None:
            raise InvalidImageError("无法解析滑块目标图片")

        background = cv2.imdecode(np.frombuffer(background_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
        if background is None:
            raise InvalidImageError("无法解析滑块背景图片")

        background = cv2.Canny(background, 100, 200)
        target = cv2.Canny(target, 100, 200)

        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)

        res = cv2.matchTemplate(background, target, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w = target.shape[:2]
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        return {"target_x": target_x,
                "target_y": target_y,
                "target": [int(max_loc[0]), int(max_loc[1]), int(bottom_right[0]), int(bottom_right[1])]}

    def slide_comparison(self, target_bytes: bytes = None, background_bytes: bytes = None):
        if target_bytes is None or background_bytes is None:
            raise DdddOcrInputError("需要提供 target_bytes 和 background_bytes")
        target_bytes = self._ensure_image_bytes(target_bytes, 'target_bytes')
        background_bytes = self._ensure_image_bytes(background_bytes, 'background_bytes')
        target = Image.open(io.BytesIO(target_bytes)).convert("RGB")
        background = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        image = ImageChops.difference(background, target)
        background.close()
        target.close()
        image = image.point(lambda x: 255 if x > 80 else 0)
        start_y = 0
        start_x = 0
        for i in range(0, image.width):
            count = 0
            for j in range(0, image.height):
                pixel = image.getpixel((i, j))
                if pixel != (0, 0, 0):
                    count += 1
                if count >= 5 and start_y == 0:
                    start_y = j - 5

            if count >= 5:
                start_x = i + 2
                break
        return {
            "target": [start_x, start_y]
        }
