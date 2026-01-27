# 示例库

本目录收录了常见场景下的最小可运行示例，方便快速对照使用。

| 示例 | 说明 |
| --- | --- |
| `basic_ocr.py` | 本地直接调用 `DdddOcr` 进行字符识别，可开启概率输出与颜色过滤。 |
| `detector.py` | 使用 `det=True` 的模型做目标检测，输出所有检测框。 |
| `api_client.py` | 调用 `python -m ddddocr api` 启动的 HTTP 服务，演示如何发送 JSON 请求。 |

## 使用方式

1. 安装项目依赖：`pip install -e .` 或 `pip install ddddocr`。
2. 在仓库根目录运行示例，例如：

```bash
python examples/basic_ocr.py ./path/to/captcha.png --probability
python examples/detector.py ./path/to/captcha.png
python examples/api_client.py ./path/to/captcha.png --endpoint http://127.0.0.1:8000/ocr
```

每个脚本都带有 `-h/--help` 选项，可查看全部参数。欢迎在 issue 中补充更多场景。
