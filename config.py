#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2022/7
  @author  : WYX
  @file    : config.py
"""

# 算法名称
algName = "NYQH_NZW_CORN"

# 算法关键步骤
keySteps = ["气候区划算法执行",
            "数据预处理完成"
            "参数获取完成",
            "指标计算完成",
            "区划计算完成",
            "输出结果完成",
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
