#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/4/27 13:28
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : write_data.py
"""

import os
import h5py
import numpy as np
import datetime
from osgeo import gdal, osr
from qgis.core import *
from PyQt5.QtGui import QColor


def createNetCDFNom(filepath, val_arr, varname):
    """
标称投影数据的输出
    :param filepath:文件路径
    :param val_arr:数据,numpy数据类型
    :param varname:变量名
    """
    h5fid = h5py.File(filepath, 'w')

    # 创建变量
    h5fid.create_dataset(varname, data=val_arr, dtype=np.float32, chunks=True, compression='gzip')

    # 添加变量属性
    h5fid[varname].attrs['long_name'] = ""
    h5fid[varname].attrs['standard_name'] = ""
    h5fid[varname].attrs['_FillValue'] = ""
    h5fid[varname].attrs['valid_range'] = ""
    h5fid[varname].attrs['scale_factor'] = ""
    h5fid[varname].attrs['add_offset'] = ""
    h5fid[varname].attrs['units'] = ""
    h5fid[varname].attrs['resolution'] = ""

    # 星下点纬度
    h5fid.create_dataset('nominal_satellite_subpoint_lat', data=0, dtype=np.float32)
    h5fid['nominal_satellite_subpoint_lat'].attrs[
        'long_name'] = 'nominal satellite subpoint latitude (platform latitude)'
    h5fid['nominal_satellite_subpoint_lat'].attrs['standard_name'] = 'Latitude'
    h5fid['nominal_satellite_subpoint_lat'].attrs['units'] = 'degrees_north'

    # 星下点经度
    h5fid.create_dataset('nominal_satellite_subpoint_lon ', data=140.7, dtype=np.float32)
    h5fid['nominal_satellite_subpoint_lon '].attrs[
        'long_name'] = 'nominal satellite subpoint longitude (platformlongitude)'
    h5fid['nominal_satellite_subpoint_lon '].attrs['standard_name'] = 'Longitude'
    h5fid['nominal_satellite_subpoint_lon '].attrs['units'] = 'degrees_east'

    # 卫星高度
    h5fid.create_dataset('nominal_satellite_height', data=35785863, dtype=np.float32)
    h5fid['nominal_satellite_height'].attrs[
        'long_name'] = 'nominal satellite height above GRS 80 ellipsoid(platform altitude)'
    h5fid['nominal_satellite_height'].attrs['standard_name'] = 'height_above_reference_ellipsoid'
    h5fid['nominal_satellite_height'].attrs['units'] = 'm'

    # 全局属性
    h5fid.attrs['dataset_name'] = 'FY4A AGRI L2 Data'
    h5fid.attrs['Project'] = "NOM"
    h5fid.attrs['platform_ID'] = 'FY4A'
    h5fid.attrs['instrument_ID'] = 'AGRI'
    h5fid.attrs['processing_level'] = 'L2'
    h5fid.attrs['date_created'] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    h5fid.attrs['Version_Of_Software'] = 'V0.1'

    h5fid.close()


def createNetCDFGll(filepath, val_arr, lon_arr, lat_arr, varname):
    """
等经纬投影数据的输出
    :param filepath:文件路径
    :param val_arr:数据，numpy数据类型
    :param lon:经度，一维数组，numpy数据类型，由小到大排列
    :param lat:纬度，一维数组，numpy数据类型，由大到小排列
    :param varname:变量名
    :return:
    """
    h5fid = h5py.File(filepath, 'w')

    # 添加纬度
    lat_ds = h5fid.create_dataset("latitude", data=lat_arr)
    lat_ds.attrs["long_name"] = 'Latitude'
    lat_ds.attrs["units"] = 'degrees_north'

    # 添加经度
    lon_ds = h5fid.create_dataset("longitude", data=lon_arr)
    lon_ds.attrs["long_name"] = 'Longitude'
    lon_ds.attrs["units"] = 'degrees_east'

    # 创建变量
    h5fid.create_dataset(varname, data=val_arr, dtype=np.float32, chunks=True, compression='gzip')

    # 添加变量属性
    h5fid[varname].attrs['long_name'] = ""
    h5fid[varname].attrs['standard_name'] = ""
    h5fid[varname].attrs['_FillValue'] = "65535"
    h5fid[varname].attrs['valid_range'] = ""
    h5fid[varname].attrs['scale_factor'] = "1.0"
    h5fid[varname].attrs['add_offset'] = "0"
    h5fid[varname].attrs['units'] = ""
    h5fid[varname].attrs['resolution'] = ""

    # Attach scales to units
    h5fid[varname].dims[0].attach_scale(lat_ds)
    h5fid[varname].dims[1].attach_scale(lon_ds)

    # 星下点纬度
    h5fid.create_dataset('nominal_satellite_subpoint_lat', data=0, dtype=np.float32)
    h5fid['nominal_satellite_subpoint_lat'].attrs[
        'long_name'] = 'nominal satellite subpoint latitude (platform latitude)'
    h5fid['nominal_satellite_subpoint_lat'].attrs['standard_name'] = 'Latitude'
    h5fid['nominal_satellite_subpoint_lat'].attrs['units'] = 'degrees_north'

    # 星下点经度
    h5fid.create_dataset('nominal_satellite_subpoint_lon ', data=140.7, dtype=np.float32)
    h5fid['nominal_satellite_subpoint_lon '].attrs[
        'long_name'] = 'nominal satellite subpoint longitude (platformlongitude)'
    h5fid['nominal_satellite_subpoint_lon '].attrs['standard_name'] = 'Longitude'
    h5fid['nominal_satellite_subpoint_lon '].attrs['units'] = 'degrees_east'

    # 卫星高度
    h5fid.create_dataset('nominal_satellite_height', data=35785863, dtype=np.float32)
    h5fid['nominal_satellite_height'].attrs[
        'long_name'] = 'nominal satellite height above GRS 80 ellipsoid(platform altitude)'
    h5fid['nominal_satellite_height'].attrs['standard_name'] = 'height_above_reference_ellipsoid'
    h5fid['nominal_satellite_height'].attrs['units'] = 'm'

    # 全局属性
    h5fid.attrs['dataset_name'] = 'FY4A AGRI L2 Data'
    h5fid.attrs['Project'] = "GLL"
    h5fid.attrs['platform_ID'] = 'FY4A'
    h5fid.attrs['instrument_ID'] = 'AGRI'
    h5fid.attrs['processing_level'] = 'L2'
    h5fid.attrs['date_created'] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    h5fid.attrs['Version_Of_Software'] = 'V0.1'

    h5fid.close()


def createGeoTIFF(filepath, val_arr, lon_arr, lat_arr, resolution, datatype):
    """
将数据输出为GeoTIFF格式的数据（等经纬投影）
    :param filepath: 数据路径
    :param val_arr: 数据，numpy数据类型，shape=[dim,x,y]
    :param lon_arr: 经度，一维数组，numpy数据类型，由小到大排列
    :param lat_arr: 纬度，一维数组，numpy数据类型，由大到小排列
    :param resolution: 分辨率，单位：度
    :param datatype: 数据类型，[gdal.GDT_*]
    :return:
    """
    # 判读数组维数
    if len(val_arr.shape) == 3:
        im_bands, im_height, im_width = val_arr.shape
    else:
        im_bands, (im_height, im_width) = 1, val_arr.shape

    # 构建文件框架
    ds = gdal.GetDriverByName("GTiff").Create(filepath, im_width, im_height, im_bands, datatype)
    # 设置影像的范围:[左上角x坐标,东西方向分辨率,旋转角度,左上角y坐标,旋转角度,南北方向分辨率]
    ds.SetGeoTransform([lon_arr[0], float(resolution), 0, lat_arr[0], 0, -float(resolution)])

    # 地理坐标系统信息,等经纬投影
    ssr = osr.SpatialReference()
    ssr.ImportFromEPSG(4326)
    ds.SetProjection(ssr.ExportToWkt())
    # 数据写出
    if im_bands == 1:
        ds.GetRasterBand(1).WriteArray(val_arr)  # 写入数组数据
    else:
        for i in range(im_bands):
            ds.GetRasterBand(i + 1).WriteArray(val_arr[i])

    ds.FlushCache()
    del ds


def createQGISMap(filenames, sensor, resolution, levels, colors, auxdata):
    tiffpath = filenames["tifpath"]
    pngpath = filenames["jpgpath"]
    datestr = filenames["timestr"]
    maintitle = filenames["title"]
    qgs_template = os.path.join(auxdata, "sps_cth.qgs")

    # 解决ERROR 6: The PNG driver does not support update access to existing datasets.
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

    # create a reference to the QgsApplication, setting the # second argument to False disables the GUI
    qgs = QgsApplication([], False)
    # load providers
    qgs.initQgis()

    # p1：读取qgs模板文件
    project = QgsProject.instance()
    project.read(qgs_template)

    # 加载栅格图层
    rlayer = QgsRasterLayer(tiffpath, "rlayer")
    project.addMapLayer(rlayer)

    # 修改图层顺序
    root = project.layerTreeRoot()

    myrlayer = root.findLayer(rlayer.id())
    rlayclone = myrlayer.clone()
    parent = myrlayer.parent()
    parent.insertChildNode(4, rlayclone)
    parent.removeChildNode(myrlayer)

    # 颜色渲染
    lst = [QgsColorRampShader.ColorRampItem(level, QColor(color)) for level, color in zip(levels, colors)]

    myRasterShader = QgsRasterShader()
    myColorRamp = QgsColorRampShader()

    myColorRamp.setColorRampItemList(lst)
    myColorRamp.setColorRampType(QgsColorRampShader.Interpolated)
    myRasterShader.setRasterShaderFunction(myColorRamp)

    myPseudoRenderer = QgsSingleBandPseudoColorRenderer(rlayer.dataProvider(), rlayer.type(),
                                                        myRasterShader)
    rlayer.setRenderer(myPseudoRenderer)
    rlayer.triggerRepaint()

    # 获取qgs模板文件中的名字为jilin的制图模板
    layout = project.layoutManager().layoutByName("layout")

    mapitem = layout.itemById("Map1")
    extent = rlayer.extent()
    extent.scale(1.1)
    mapitem.zoomToExtent(extent)

    # 设置标题
    title = layout.itemById("title")
    title.setText(maintitle)

    # 添加日期
    datelabel = layout.itemById("datelabel")
    datelabel.setText(datestr)

    # 添加卫星载荷
    sensorlabel = layout.itemById("sensor")
    sensorlabel.setText(sensor)

    # 添加分辨率
    reslabel = layout.itemById("resolution")
    reslabel.setText(resolution)

    layout.refresh()

    # 输出图片
    exporter = QgsLayoutExporter(layout)
    exporter.exportToImage(pngpath, QgsLayoutExporter.ImageExportSettings())
    
    qgs.exitQgis()
