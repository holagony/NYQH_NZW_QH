#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/5/13 10:14
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : time_match.py
"""

import os
import sys
from datetime import datetime
from pytz import timezone
import numpy as np
from arspy import glo
import re


def fmt2pattern(priDateFmt):
    """
将时间日期格式化符号替换为正则表达式格式
    :param priDateFmt:
    :return:
    """
    priDateFmt = re.sub("%Y", r"\\d\\d\\d\\d", priDateFmt)  # 年
    priDateFmt = re.sub("%m", r"\\d\\d", priDateFmt)  # 月
    priDateFmt = re.sub("%d", r"\\d\\d", priDateFmt)  # 日
    priDateFmt = re.sub("%H", r"\\d\\d", priDateFmt)  # 时
    priDateFmt = re.sub("%M", r"\\d\\d", priDateFmt)  # 分
    priDateFmt = re.sub("%S", r"\\d\\d", priDateFmt)  # 秒
    priDateFmt = re.sub("%j", r"\\d\\d\\d", priDateFmt)  # 年内的一天
    return priDateFmt


def time_nearest_match(priFile, priDateFmt, auxFiles, auxDateFmt, priTZ="UTC", auxTZ="UTC", threshold="NUL"):
    """
主文件与辅助文件最邻近时间匹配
    :param priFile: 主文件
    :param priDateFmt: 主文件中时间字符格式
    :param auxFiles: 辅助文件
    :param auxDateFmt: 辅助文件时间字符格式
    :param priTZ: 主文件文件名时间所属时区，默认是世界时，若是北京时，则 'Asia/Shanghai'
    :param auxTZ: 辅助文件文件名时间所属时区，默认是世界时
    :param threshold: 时间阈值（单位：分钟）：默认为NUL，即在平台所传所有文件中找出最近的时间；
                     若设置阈值，则会判断最邻近是否在阈值范围内，超出阈值则会抛出时间异常
    :return:时间最邻近的文件全路径,主文件与辅助文件的时间差,辅助文件的时间（世界时）
    """
    # 提取主文件时间
    filename = os.path.basename(priFile)
    priStr = re.findall(fmt2pattern(priDateFmt), filename)[0]
    pridatef = timezone(priTZ).localize(datetime.strptime(priStr, priDateFmt))

    # 提取辅文件时间
    auxlst = auxFiles.split(',')
    auxtimesf = np.array(
        [timezone(auxTZ).localize(datetime.strptime(re.findall(fmt2pattern(auxDateFmt), auxFile)[0], auxDateFmt))
         for auxFile in auxlst])

    auxtimesf = auxtimesf.tolist()

    # 在辅文件时间中找到距离pridatef最邻近时间
    nsttime = min(auxtimesf, key=lambda x: abs(x - pridatef))
    nstauxfile = auxlst[auxtimesf.index(nsttime)]
    difftime = (pridatef - nsttime).total_seconds() / 60  # 时间差多少分钟
    if threshold != "NUL":
        if np.abs(difftime) > float(threshold):
            rjson = glo.get_value("rjson")
            fjson = glo.get_value("fjson")

            fjson.info("最邻近时间超出设定阈值!")
            rjson.info('status', ['3', "最邻近时间超出设定阈值!"])
            sys.exit()

    return nstauxfile, difftime, nsttime
