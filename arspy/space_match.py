#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/4/28 16:21
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : space_match.py
"""

from osgeo import gdal, osr


def nom2Gll(val_arr, res_in, res_out, out_bounds, srcnodata=65535, dstnodata=65535, dtype=gdal.GDT_Float32):
    """
将标称投影数据转换为等经纬投影(以FY4A卫星为例),支持多维度数据的转换，前提条件是第一维是层数，shape=[dim,x,y]
    :param val_arr: 待转换数据，shape=[dim,x,y]
    :param res_in: 待转换数据分辨率,单位：米
    :param res_out: 转换后的数据分辨率,单位：度
    :param out_bounds: 输出数据的范围,[minLon,minLat,maxLon,maxLat]
    :param srcnodata: 源数据的无效值，默认为65535
    :param dstnodata: 目标数据的无效值，默认为65535
    :param dtype: 数据类型，gdal.GDT_*
    :return:
    """
    # 判读数组维数
    if len(val_arr.shape) == 3:
        im_bands, im_height, im_width = val_arr.shape
    else:
        im_bands, (im_height, im_width) = 1, val_arr.shape
    # 构建MEM文件框架
    ds = gdal.GetDriverByName('MEM').Create('', im_width, im_height, im_bands, dtype)  # GTiff,MEM

    # 设置影像的范围:[左上角x坐标,东西方向分辨率,旋转角度,左上角y坐标,旋转角度,南北方向分辨率]
    ds.SetGeoTransform([-0.5 * ds.RasterXSize * res_in, res_in, 0, 0.5 * ds.RasterYSize * res_in, 0, -res_in])

    # 地理坐标系统信息
    srs = osr.SpatialReference()  # 获取地理坐标系统信息，用于选取需要的地理坐标系统
    # 定义地球长半轴a=6378137.0m，地球短半轴b=6356752.3m，FY4A卫星高度h=35786000，卫星星下点所在经度104.7，目标空间参考
    srs.ImportFromProj4('+proj=geos +h=35786000 +a=6378137.0 +b=6356752.3 +lon_0=104.7 +no_defs')  # 定义输出的坐标系
    ds.SetProjection(srs.ExportToWkt())  # 给新建图层赋予投影信息
    # 数据写出
    if im_bands == 1:
        ds.GetRasterBand(1).WriteArray(val_arr)  # 写入数组数据
    else:
        for i in range(im_bands):
            ds.GetRasterBand(i + 1).WriteArray(val_arr[i])
    ds.FlushCache()

    # 投影转换,指定了最近邻采样方法，当然也可以指定其他重采样方法:resampleAlg = gdal.GRA_*
    dst_ds = gdal.Warp('', ds, dstSRS='EPSG:4326', format='MEM', outputType=dtype,
                       outputBounds=out_bounds, xRes=res_out, yRes=res_out, dstNodata=dstnodata,
                       srcNodata=srcnodata, resampleAlg=gdal.GRA_NearestNeighbour)
    im_data_warp = dst_ds.ReadAsArray(0, 0, dst_ds.RasterXSize, dst_ds.RasterYSize)  # 获取数据,xoff/yoff设定为0表示获取整个数据
    del ds
    del dst_ds

    return im_data_warp
