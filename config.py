#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2022/7
  @author  : WYX
  @file    : config.py
"""

# 算法名称
algName = "NYQH_NZW_QH"

# 算法关键步骤
keySteps = ["开始数据准备",
            "开始批量计算指标",
            "开始区划计算",
            "开始输出结果",
            "区划处理完成",
            "算法执行结束"
]

# 入口json标签
labels = [
    "startDate",
    "endDate",
    "regionName",
    "regionCode",
    "RSP",
    "inPutFileList",
    "resultPath",
    "resultJsonPath",
    "resultLogPath",
    "resultFlowPath",
    "taskId"
]
