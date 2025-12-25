#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/4/27 13:25
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : read_data.py
"""

import xarray as xr
import numpy as np
import h5py


def read_tiff(filepath):
    """
读取TIFF数据
    :param filepath:
    :return:
    """
    ds = xr.open_rasterio(filepath)
    val = np.array(ds.sel(band=1))
    lon = np.array(ds.x)
    lat = np.array(ds.y)
    return val, lon, lat


def read_nc(filepath):
    """
读取NC格式数据
    @param filepath:
    @return:
    """
    with h5py.File(filepath, 'r') as fid:
        val = fid["LST"][:]

    val[(val < 250) | (val > 350)] = 65535
    val = val.astype(np.float)
    print(val)
    return val
