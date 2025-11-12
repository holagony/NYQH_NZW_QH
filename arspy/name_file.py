#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/4/27 13:54
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : name_file.py
"""

import os
import datetime
from arspy.id import product, proj, period, fmt, region


def prodname(filename, resultPath, region, level, prod, proj, res, period, fmt, version="V0001"):
    """
    文件名命名原则：尽量与输入的文件名命名保持一致
    :param filename: 原始数据文件名（无路径）
    :param resultPath: 输出目录
    :param region: 数据区域
    :param level: 数据级别
    :param prod: 产品名称
    :param proj: 投影
    :param res: 分辨率
    :param period: 周期
    :param fmt: 格式
    :param version: 版本
    """
    # 原始文件名分割
    items = filename.split("_")

    # 定义文件名组成(文件名中固定部分从原始文件名中获取，其余可变部分通过传参获取)
    sat = items[0]  # 卫星名称
    ins = items[1]  # 仪器名称
    model = items[2]  # 观测模式
    subpoint = items[4]  # 星下点经度
    chl = items[7]  # 仪器通道
    stime = items[9]  # 开始时间
    etime = items[10]  # 结束时间

    # 文件名规则：
    # 卫星名称_仪器名称_观测模式_数据区域_星下点经度_数据级别_产品名称_仪器通道_投影方式_观测起始时间_观测结束时间_空间分辨率_产品周期_产品版本.数据格式
    prodname = f'{sat}_{ins}_{model}_{region}_{subpoint}_{level}_{prod["std"]}_{chl}_{proj}_{stime}_{etime}_{res}_{period}_{version}.{fmt}'

    # 目录规则：
    # 根目录/卫星类别/仪器类别/数据级别/数据区域/时间属性/产品属性/投影方式/年/年月日
    prodDir = os.path.join(resultPath, sat, ins, level, region, period, prod["var"], proj, stime[:4], stime[:8])
    prodDir = prodDir.replace("-", "")  # 路径名称不包含中划线 - 字符
    if not os.path.exists(prodDir):
        os.makedirs(prodDir)
    # 产品路径
    prodpath = os.path.join(prodDir, prodname)
    return prodpath


def createName(priFile, resultPath):
    """
    函数实现两种定义：一是产品输出路径的定义；二是产品输出文件名的定义；
    其中输出的文件名定义包括NC文件、TIFF文件、缩略图文件的文件名，以及缩略图标题
    @param inputfile: 输入的文件
    @param resultPath:输出文件根目录
    @return:定义的名称（字典形式）
    """
    # 获取文件名
    priname = os.path.basename(priFile)

    # 产品路径
    ncpath = prodname(priname, resultPath, region.DISK, "L2-", product.CTP, proj.GLL, "2000M", period.DAY1, fmt.NC)
    tiffpath = prodname(priname, resultPath, region.DISK, "L2-", product.CTP, proj.GLL, "2000M", period.DAY1, fmt.TIFF)
    pngpath = prodname(priname, resultPath, region.DISK, "L2-", product.CTP, proj.GLL, "2000M", period.DAY1, fmt.PNG)

    timestr = priname.split("_")[9]
    timef = datetime.datetime.strptime(timestr, "%Y%m%d%H%M%S")  # 时间字符串,datetime格式
    sat = priname.split("_")[0]

    # 输出缩略图标题
    prodtitle = "%s_%s_%s %s" % (sat, product.CTP["var"], proj.GLL, timef.__format__("%Y-%m-%d %H:%M:%S UTC"))

    # 定义的文件名和文件标题以字典形式返回
    names = {
        "ncpath": ncpath,
        "tiffpath": tiffpath,
        "pngpath": pngpath,
        "prodtitle": prodtitle
    }
    return names
