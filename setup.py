#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 15:41
# @Author  : sml2h3
# @Site    :
# @File    : setup.py
# @Software: PyCharm
# @Description:

import platform
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 根据onnxruntime的支持情况设置Python版本限制
python_requires = ">=3.10"

setup(
    name="ddddocr",
    version="1.5.8",
    author="sml2h3",
    description="带带弟弟OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sml2h3/ddddocr",
    packages=find_packages(where='.', exclude=(), include=('*',)),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'onnxruntime', 'Pillow'],
    extras_require={
        ':sys_platform == "linux"': ['opencv-python-headless'],
        ':sys_platform == "win32" or sys_platform == "darwin"': ['opencv-python'],
    },
    python_requires=python_requires,
    include_package_data=True,
    install_package_data=True,
)
