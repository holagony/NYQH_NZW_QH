#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/3/18 10:49
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : glo.py
"""

"""
为了方便调用接口文件中的变量，因此定义为全局变量
"""


def _init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    """ 定义一个全局变量 """
    _global_dict[key] = value


def get_value(key):
    """ 获得一个全局变量,不存在则返回默认值 """
    try:
        return _global_dict[key]
    except KeyError as ke:
        print(ke)
