

# DdddOcr 带带弟弟OCR通用验证码离线本地识别SDK免费开源版

DdddOcr，其由 [本作者](https://github.com/sml2h3) 与 [kerlomz](https://github.com/kerlomz) 共同合作完成，通过大批量生成随机数据后进行深度网络训练，本身并非针对任何一家验证码厂商而制作，本库使用效果完全靠玄学，可能可以识别，可能不能识别。

DdddOcr、最简依赖的理念，尽量减少用户的配置和使用成本，希望给每一位测试者带来舒适的体验

项目地址： [点我传送](https://github.com/sml2h3/ddddocr) 

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/sml2h3/ddddocr/">
    <img src="https://cdn.wenanzhe.com/img/logo.png!/crop/700x500a400a500" alt="Logo">
  </a>
  <p align="center">
    一个容易使用的通用验证码识别python库
    <br />
    <a href="https://github.com/sml2h3/ddddocr/"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/sml2h3/ddddocr/issues">报告Bug</a>
    ·
    <a href="https://github.com/sml2h3/ddddocr/issues">提出新特性</a>
  </p>

</p>

 
## 目录

- [赞助合作商](#赞助合作商)
- [上手指南](#上手指南)
  - [环境支持](#环境支持)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [项目底层支持](#项目底层支持)
- [使用文档](#使用文档)
  - [基础ocr识别能力](#i-基础ocr识别能力)
  - [图片颜色过滤功能](#ii-图片颜色过滤功能)
  - [目标检测能力](#iii-目标检测能力)
  - [滑块检测](#ⅳ-滑块检测)
  - [OCR概率输出](#ⅴ-ocr概率输出)
  - [自定义OCR训练模型导入](#ⅵ-自定义ocr训练模型导入)
  - [HTTP API服务](#ⅶ-http-api服务)
  - [MCP协议支持](#ⅷ-mcp协议支持)
- [版本控制](#版本控制)
- [常见问题解决方案](#常见问题解决方案)
- [相关推荐文章or项目](#相关推荐文章or项目)
- [作者](#作者)
- [捐赠](#捐赠)
- [Star历史](#Star历史)



### 赞助合作商

|                                                            | 赞助合作商 | 推荐理由                                                                                             |
|------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------|
| ![YesCaptcha](https://cdn.wenanzhe.com/img/yescaptcha.png) | [YesCaptcha](https://yescaptcha.com/i/NSwk7i) | 谷歌reCaptcha验证码 / hCaptcha验证码 / funCaptcha验证码商业级识别接口 [点我](https://yescaptcha.com/i/NSwk7i) 直达VIP4 |
| ![超级鹰](https://cdn.wenanzhe.com/img/logo.gif) | [超级鹰](https://www.chaojiying.com/) | 全球领先的智能图片分类及识别商家，安全、准确、高效、稳定、开放，强大的技术及校验团队，支持大并发。7*24h作业进度管理 |
| ![Malenia](https://cdn.wenanzhe.com/img/malenia.png!/scale/50)    | [Malenia](https://malenia.iinti.cn/malenia-doc/) | Malenia企业级代理IP网关平台/代理IP分销软件 |
| 雨云VPS    | [注册首月5折](https://www.rainyun.com/ddddocr_) | 浙江节点低价大带宽，100M每月30元 |


### 上手指南

###### 环境支持



| 系统               | CPU | GPU | 最大支持py版本 | 备注                                                                 |
|------------------|-----|------|----------|--------------------------------------------------------------------|
| Windows 64位      | √   | √ | 3.12     | 部分版本windows需要安装<a href="https://www.ghxi.com/yxkhj.html">vc运行库</a> |
| Windows 32位      | ×   | × | -        |                                                                    |
| Linux 64 / ARM64 | √   | √ | 3.12     |                                                                    |
| Linux 32         | ×   | × | -        |                                                                    |
| Macos  X64       | √   | √ | 3.12     | M1/M2/M3...芯片参考<a href="https://github.com/sml2h3/ddddocr/issues/67">#67</a>         |

###### **安装步骤**

**i. 从pypi安装** 
```sh
pip install ddddocr
```

**ii. 安装API服务支持**
```sh
pip install ddddocr[api]
```

**iii. 从源码安装**
```sh
git clone https://github.com/sml2h3/ddddocr.git
cd ddddocr
python setup.py install
```

**请勿直接在ddddocr项目的根目录内直接import ddddocr**，请确保你的开发项目目录名称不为ddddocr，此为基础常识。

### 文件目录说明
eg:

```
ddddocr 
├── MANIFEST.in
├── LICENSE
├── README.md
├── /ddddocr/
│  │── __init__.py            主代码库文件
│  │── common.onnx            新ocr模型
│  │── common_det.onnx        目标检测模型
│  │── common_old.onnx        老ocr模型
│  │── logo.png
│  │── README.md
│  │── requirements.txt
├── logo.png
└── setup.py

```

### 项目底层支持 

本项目基于[dddd_trainer](https://github.com/sml2h3/dddd_trainer) 训练所得，训练底层框架位pytorch，ddddocr推理底层抵赖于[onnxruntime](https://pypi.org/project/onnxruntime/)，故本项目的最大兼容性与python版本支持主要取决于[onnxruntime](https://pypi.org/project/onnxruntime/)。

### 使用文档

##### i. 基础ocr识别能力

主要用于识别单行文字，即文字部分占据图片的主体部分，例如常见的英数验证码等，本项目可以对中文、英文（随机大小写or通过设置结果范围圈定大小写）、数字以及部分特殊字符。

```python
# example.py
import ddddocr

ocr = ddddocr.DdddOcr()

image = open("example.jpg", "rb").read()
result = ocr.classification(image)
print(result)
```

本库内置有两套ocr模型，默认情况下不会自动切换，需要在初始化ddddocr的时候通过参数进行切换

```python
# example.py
import ddddocr

ocr = ddddocr.DdddOcr(beta=True)  # 切换为第二套ocr模型

image = open("example.jpg", "rb").read()
result = ocr.classification(image)
print(result)
```

**提示**
对于部分透明黑色png格式图片得识别支持: `classification` 方法 使用 `png_fix` 参数，默认为False

```python
 ocr.classification(image, png_fix=True)
```

**注意**

之前发现很多人喜欢在每次ocr识别的时候都重新初始化ddddocr，即每次都执行```ocr = ddddocr.DdddOcr()```，这是错误的，通常来说只需要初始化一次即可，因为每次初始化和初始化后的第一次识别速度都非常慢


**参考例图**

包括且不限于以下图片

<img src="https://cdn.wenanzhe.com/img/20210715211733855.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/78b7f57d-371d-4b65-afb2-d19608ae1892.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142305.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142325.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2AMLyA_fd83e1f1800e829033417ae6dd0e0ae0.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/aabd_181ae81dd5526b8b89f987d1179266ce.jpg" alt="captcha" width="150">
<br />
<img src="https://cdn.wenanzhe.com/img/2bghz_b504e9f9de1ed7070102d21c6481e0cf.png" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/0000_z4ecc2p65rxc610x.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2acd_0586b6b36858a4e8a9939db8a7ec07b7.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/2a8r_79074e311d573d31e1630978fe04b990.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/aftf_C2vHZlk8540y3qAmCM.bmp" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20211226144057.png" alt="captcha" width="150">

##### ii. 图片颜色过滤功能

本功能支持HSV颜色空间的颜色范围过滤，可以有效提高特定颜色文字的识别准确率。

**内置颜色预设**

支持以下预设颜色：red（红色）、blue（蓝色）、green（绿色）、yellow（黄色）、orange（橙色）、purple（紫色）、cyan（青色）、black（黑色）、white（白色）、gray（灰色）

**使用方式1：通过预设颜色名称**

```python
import ddddocr

ocr = ddddocr.DdddOcr()

with open("captcha.jpg", "rb") as f:
    image = f.read()

# 只保留红色和蓝色的文字
result = ocr.classification(image, color_filter_colors=['red', 'blue'])
print(result)
```

**使用方式2：通过自定义HSV范围**

```python
import ddddocr

ocr = ddddocr.DdddOcr()

with open("captcha.jpg", "rb") as f:
    image = f.read()

# 自定义HSV颜色范围 (H, S, V)
custom_ranges = [
    ((0, 50, 50), (10, 255, 255)),    # 红色范围1
    ((170, 50, 50), (180, 255, 255))  # 红色范围2
]

result = ocr.classification(image, color_filter_custom_ranges=custom_ranges)
print(result)
```

**查看可用颜色**

```python
from ddddocr import ColorFilter

# 获取所有可用的预设颜色
colors = ColorFilter.get_available_colors()
print(colors)

# 查看颜色的HSV范围
print(ColorFilter.COLOR_PRESETS['red'])
```

**命令行查看颜色**

```sh
python -m ddddocr colors
```

##### iii. 目标检测能力

主要用于快速检测出图像中可能的目标主体位置，由于被检测出的目标不一定为文字，所以本功能仅提供目标的bbox位置 **（在⽬标检测⾥，我们通常使⽤bbox（bounding box，缩写是 bbox）来描述⽬标位置。bbox是⼀个矩形框，可以由矩形左上⻆的 x 和 y 轴坐标与右下⻆的 x 和 y 轴坐标确定）** 

如果使用过程中无需调用ocr功能，可以在初始化时通过传参`ocr=False`关闭ocr功能，开启目标检测需要传入参数`det=True`

```python
import ddddocr
import cv2

det = ddddocr.DdddOcr(det=True)

with open("test.jpg", 'rb') as f:
    image = f.read()

bboxes = det.detection(image)
print(bboxes)

im = cv2.imread("test.jpg")

for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

cv2.imwrite("result.jpg", im)

```



**参考例图**

包括且不限于以下图片

<img src="https://cdn.wenanzhe.com/img/page1_1.jpg" alt="captcha" width="200">
<img src="https://cdn.wenanzhe.com/img/page1_2.jpg" alt="captcha" width="200">
<img src="https://cdn.wenanzhe.com/img/page1_3.jpg" alt="captcha" width="200">
<img src="https://cdn.wenanzhe.com/img/page1_4.jpg" alt="captcha" width="200">
<br />
<img src="https://cdn.wenanzhe.com/img/result.jpg" alt="captcha" width="200">
<img src="https://cdn.wenanzhe.com/img/result2.jpg" alt="captcha" width="200">
<img src="https://cdn.wenanzhe.com/img/result4.jpg" alt="captcha" width="200">

##### Ⅳ. 滑块检测

本项目的滑块检测功能并非AI识别实现，均为opencv内置算法实现。可能对于截图党用户没那么友好~，如果使用过程中无需调用ocr功能或目标检测功能，可以在初始化时通过传参`ocr=False`关闭ocr功能或`det=False`来关闭目标检测功能

本功能内置两套算法实现，适用于两种不同情况，具体请参考以下说明

**a.算法1**

算法1原理是通过滑块图像的边缘在背景图中计算找到相对应的坑位，可以分别获取到滑块图和背景图，滑块图为透明背景图

滑块图

<img src="https://cdn.wenanzhe.com/img/b.png" alt="captcha" width="50">

背景图

<img src="https://cdn.wenanzhe.com/img/a.png" alt="captcha" width="350">

```python
    det = ddddocr.DdddOcr(det=False, ocr=False)
    
    with open('target.png', 'rb') as f:
        target_bytes = f.read()
    
    with open('background.png', 'rb') as f:
        background_bytes = f.read()
    
    res = det.slide_match(target_bytes, background_bytes)
    
    print(res)
  ```
  由于滑块图可能存在透明边框的问题，导致计算结果不一定准确，需要自行估算滑块图透明边框的宽度用于修正得出的bbox

  *提示：如果滑块无过多背景部分，则可以添加simple_target参数， 通常为jpg或者bmp格式的图片*

```python
    slide = ddddocr.DdddOcr(det=False, ocr=False)
    
    with open('target.jpg', 'rb') as f:
        target_bytes = f.read()
    
    with open('background.jpg', 'rb') as f:
        background_bytes = f.read()
    
    res = slide.slide_match(target_bytes, background_bytes, simple_target=True)
    
    print(res)
  ```

**a.算法2**

算法2是通过比较两张图的不同之处进行判断滑块目标坑位的位置

参考图a，带有目标坑位阴影的全图

<img src="https://cdn.wenanzhe.com/img/bg.jpg" alt="captcha" width="350">

参考图b，全图

<img src="https://cdn.wenanzhe.com/img/fullpage.jpg" alt="captcha" width="350">

```python
    slide = ddddocr.DdddOcr(det=False, ocr=False)

    with open('bg.jpg', 'rb') as f:
        target_bytes = f.read()
    
    with open('fullpage.jpg', 'rb') as f:
        background_bytes = f.read()
    
    img = cv2.imread("bg.jpg")
    
    res = slide.slide_comparison(target_bytes, background_bytes)

    print(res)
  ```

##### Ⅴ. OCR概率输出

为了提供更灵活的ocr结果控制与范围限定，项目支持对ocr结果进行范围限定。

可以通过在调用`classification`方法的时候传参`probability=True`，此时`classification`方法将返回全字符表的概率
当然也可以通过`set_ranges`方法设置输出字符范围来限定返回的结果。

Ⅰ. `set_ranges` 方法限定返回字符返回

本方法接受1个参数，如果输入为int类型为内置的字符集限制，string类型则为自定义的字符集

如果为int类型，请参考下表

| 参数值 | 意义                                |
|-----|-----------------------------------|
| 0   | 纯整数0-9                            |
| 1   | 纯小写英文a-z                          |
| 2   | 纯大写英文A-Z                          |
| 3   | 小写英文a-z + 大写英文A-Z                 |
| 4   | 小写英文a-z + 整数0-9                   |
| 5   | 大写英文A-Z + 整数0-9                   |
| 6   | 小写英文a-z + 大写英文A-Z + 整数0-9         |
| 7   | 默认字符库 - 小写英文a-z - 大写英文A-Z - 整数0-9 |

如果为string类型请传入一段不包含空格的文本，其中的每个字符均为一个待选词
如：`"0123456789+-x/=""`

```python
import ddddocr

ocr = ddddocr.DdddOcr()

image = open("test.jpg", "rb").read()
ocr.set_ranges("0123456789+-x/=")
result = ocr.classification(image, probability=True)
s = ""
for i in result['probability']:
    s += result['charsets'][i.index(max(i))]

print(s)

```

##### Ⅵ. 自定义OCR训练模型导入

本项目支持导入来自于 [dddd_trainer](https://github.com/sml2h3/dddd_trainer) 进行自定义训练后的模型，参考导入代码为

```python
import ddddocr

ocr = ddddocr.DdddOcr(det=False, ocr=False, import_onnx_path="myproject_0.984375_139_13000_2022-02-26-15-34-13.onnx", charsets_path="charsets.json")

with open('test.jpg', 'rb') as f:
    image_bytes = f.read()

res = ocr.classification(image_bytes)
print(res)

```

##### Ⅶ. HTTP API服务

本项目支持通过HTTP API的方式提供服务，方便集成到各种应用中。

**启动API服务**

```sh
# 基础启动
python -m ddddocr api

# 指定端口和主机
python -m ddddocr api --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
python -m ddddocr api --reload

# 查看所有选项
python -m ddddocr api --help
```

**API端点说明**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/initialize` | POST | 初始化并选择加载的模型类型 |
| `/switch-model` | POST | 运行时切换模型配置 |
| `/toggle-feature` | POST | 开启/关闭特定功能 |
| `/ocr` | POST | 执行OCR识别 |
| `/detect` | POST | 执行目标检测 |
| `/slide-match` | POST | 滑块匹配算法 |
| `/slide-comparison` | POST | 滑块比较算法 |
| `/status` | GET | 获取当前服务状态 |
| `/docs` | GET | Swagger UI文档 |

**使用示例**

1. 初始化服务
```bash
curl -X POST "http://localhost:8000/initialize" \
     -H "Content-Type: application/json" \
     -d '{"ocr": true, "det": false}'
```

2. OCR识别（支持颜色过滤）
```bash
curl -X POST "http://localhost:8000/ocr" \
     -H "Content-Type: application/json" \
     -d '{
       "image": "base64_encoded_image_data",
       "color_filter_colors": ["red", "blue"],
       "png_fix": false,
       "probability": false
     }'
```

3. 目标检测
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_data"}'
```

4. 查看服务状态
```bash
curl "http://localhost:8000/status"
```

**Python客户端示例**

```python
import requests
import base64

# 读取图片并转换为base64
with open("captcha.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# 初始化服务
response = requests.post("http://localhost:8000/initialize",
                        json={"ocr": True, "det": False})
print(response.json())

# OCR识别
response = requests.post("http://localhost:8000/ocr",
                        json={
                            "image": image_data,
                            "color_filter_colors": ["red", "blue"]
                        })
result = response.json()
print(result["data"]["text"])
```

##### Ⅷ. MCP协议支持

本项目支持MCP（Model Context Protocol）协议，使AI Agent能够直接调用ddddocr服务。

**MCP端点**

- 能力声明：`GET /mcp/capabilities`
- 工具调用：`POST /mcp/call`

**可用工具**

1. `ddddocr_initialize` - 初始化服务
2. `ddddocr_ocr` - OCR文字识别（支持颜色过滤）
3. `ddddocr_detection` - 目标检测
4. `ddddocr_slide_match` - 滑块匹配
5. `ddddocr_slide_comparison` - 滑块比较
6. `ddddocr_status` - 获取服务状态

**MCP调用示例**

```python
import requests

# 获取MCP能力
response = requests.get("http://localhost:8000/mcp/capabilities")
print(response.json())

# 调用OCR工具
mcp_request = {
    "method": "ddddocr_ocr",
    "params": {
        "image": "base64_encoded_image",
        "color_filter_colors": ["red", "blue"]
    },
    "id": "1"
}

response = requests.post("http://localhost:8000/mcp/call", json=mcp_request)
print(response.json())
```

### 版本控制

该项目使用Git进行版本管理。您可以在repository参看当前可用版本。

### 常见问题解决方案

#### OpenCV相关问题

**问题1：ImportError: No module named 'cv2'**

解决方案：
```bash
# 卸载可能冲突的opencv包
pip uninstall opencv-python opencv-python-headless

# 重新安装
pip install opencv-python-headless
```

**问题2：Linux系统OpenCV运行时错误**

Ubuntu/Debian系统：
```bash
sudo apt-get update
sudo apt-get install build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

CentOS/RHEL系统：
```bash
sudo yum install gcc gcc-c++ make glib2-devel libSM libXext libXrender mesa-libGL
```

**问题3：Windows系统缺少VC运行库**

下载并安装Visual C++运行库：
- [Microsoft Visual C++ 运行库下载](https://www.ghxi.com/yxkhj.html)

**问题4：macOS M1/M2芯片兼容性问题**

参考解决方案：
- [GitHub Issue #67](https://github.com/sml2h3/ddddocr/issues/67)

#### API服务相关问题

**问题1：启动API服务时提示缺少依赖**

解决方案：
```bash
pip install ddddocr[api]
# 或者
pip install fastapi uvicorn pydantic
```

**问题2：API服务端口被占用**

解决方案：
```bash
# 指定其他端口
python -m ddddocr api --port 8001

# 或查找并终止占用进程
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/macOS
```

**问题3：颜色过滤效果不理想**

解决方案：
1. 查看可用颜色预设：`python -m ddddocr colors`
2. 使用自定义HSV范围进行精确控制
3. 可以使用图像处理工具先分析图片的颜色分布

#### 性能优化建议

1. **避免重复初始化**：只初始化一次DdddOcr实例
2. **GPU加速**：如有NVIDIA GPU，可设置`use_gpu=True`
3. **批量处理**：对于大量图片，建议使用API服务模式
4. **内存管理**：处理大图片时注意内存使用

#### 识别准确率优化

1. **图片预处理**：确保图片清晰，对比度适中
2. **颜色过滤**：对于彩色验证码，使用颜色过滤功能
3. **字符集限制**：使用`set_ranges`方法限制字符范围
4. **模型选择**：尝试不同的模型（old、beta）

如果遇到其他问题，请在[GitHub Issues](https://github.com/sml2h3/ddddocr/issues)中提交问题报告。

### 相关推荐文章or项目

[带带弟弟OCR，纯VBA本地获取网络验证码整体解决方案](https://club.excelhome.net/thread-1666823-1-1.html)

[ddddocr nodejs 版本](https://github.com/rhy3h/ddddocr-node)

[ddddocr rust 版本](https://github.com/86maid/ddddocr)

[captcha-killer的修改版](https://github.com/f0ng/captcha-killer-modified)

[通过ddddocr训练字母数字验证码模型并识别部署调用](https://www.bilibili.com/video/BV1ez421C7dB)

...

欢迎更多优秀案例或教程等进行投稿，可直接新建issue标题以【投稿】开头，附上公开教程站点链接，我会选择根据文章内容选择相对不重复或者有重点内容等进行readme展示，感谢各位朋友~

### 作者

sml2h3@gamil.com
 
<img src="https://cdn.wenanzhe.com/img/mmqrcode1640418911274.png!/scale/50" alt="wechat" width="150">

 *好友数过多不一定通过，有问题可以在issue进行交流*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE](https://github.com/sml2h3/ddddocr/blob/master/LICENSE)

### 捐赠 （如果项目有帮助到您，可以选择捐赠一些费用用于ddddocr的后续版本维护，本项目长期维护）

<img src="https://cdn.wenanzhe.com/img/zhifubao.jpg" alt="captcha" width="150">
<img src="https://cdn.wenanzhe.com/img/weixin.jpg" alt="captcha" width="150">


<!-- links -->
[your-project-path]:sml2h3/ddddocr
[contributors-shield]: https://img.shields.io/github/contributors/sml2h3/ddddocr?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sml2h3/ddddocr?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/sml2h3/ddddocr?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/sml2h3/ddddocr?style=flat-square
[issues-url]: https://img.shields.io/github/issues/sml2h3/ddddocr.svg
[license-shield]: https://img.shields.io/github/license/sml2h3/ddddocr?style=flat-square
[license-url]: https://github.com/sml2h3/ddddocr/blob/master/LICENSE


### Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=sml2h3/ddddocr&type=Date)](https://star-history.com/#sml2h3/ddddocr&Date)


