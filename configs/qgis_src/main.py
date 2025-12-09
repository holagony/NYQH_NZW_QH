#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QGIS绘图主程序 - 调用qgis_plot生成专题图
支持9位areaCode（前6位县级，后3位乡镇）
"""

import os
import sys
import argparse
import os
from qgis.core import QgsProject, QgsRasterLayer,QgsVectorLayer,QgsCoordinateReferenceSystem,QgsMapLayer,QgsWkbTypes,QgsSymbol
from qgis.core import QgsExpressionContextUtils,QgsCoordinateTransform,QgsLayoutExporter,QgsApplication,QgsRuleBasedRenderer
from osgeo import gdal
import shutil
from PyQt5.QtGui import QColor
import numpy as np
from osgeo import gdal,osr
import pandas as pd
import rioxarray as rxr
import geopandas as gpd  
from qgis_src.qgis_plot import drawmaps,drawmaps_

def main(json):
    """主函数"""
    # 解析命令行参数

    areacode = json["areacode"]
    areaname = json["areaname"]
    startdate = json["startdate"]
    enddate = json["enddate"]
    cropname = json["cropname"]
    resultfile = json["resultfile"]
    # 设置路径
    auxPath = os.path.join(os.path.dirname(__file__), "auxpath")
    rasterStyle = os.path.join(auxPath, "colorbar", "CRA_AL.qml")
    
    # 区域代码字典 - 9位代码
    codesdict = {
        "610802000": '榆阳区', "610802001": '鼓楼街道', "610802002": "青山路街道", 
        "610802003": '上郡路街道', "610802004": '新明楼街道', "610802005": '驼峰路街道',
        "610802006": '崇文路街道', "610802007": '航宇路街道', "610802008": '长城路街道',
        "610802100": '鱼河镇', "610802101": '上盐湾镇', "610802102": '镇川镇',
        "610802105": '麻黄梁镇', "610802106": '牛家梁镇', "610802107": '金鸡滩镇',
        "610802108": '马合镇', "610802109": '巴拉素镇', "610802111": '鱼河峁镇',
        "610802112": '青云镇', "610802113": '古塔镇', "610802114": '大河塔镇',
        "610802115": '小纪汗镇', "610802116": '芹河镇', "610802205": '孟家湾乡',
        "610802206": '小壕兔乡', "610802207": '岔河则乡', "610802208": '补浪河乡', 
        "610802209": '红石桥乡'
    }
    
    # 获取区域名称
    area_name = codesdict.get(areacode, areaname)

    # 格式化时间
    time_str = f"{startdate[:4]}-{enddate[:4]}年"
    
    # 生成地图信息
    mapinfo = mapinfo_name(time_str, area_name, cropname)
    
    # 导入绘图模块并调用
    # try:
    # 调用绘图函数
    if areacode=="610802000":
        template = os.path.join(auxPath, "template", "template.qgs")
        drawmaps(tiffile=resultfile,
                    rasterStyle=rasterStyle,
                    template=template,
                    mapinfo=mapinfo)
    else:
        template = os.path.join(auxPath, "template", "template_xiang.qgs")
        drawmaps_(
            tiffile=resultfile,
            rasterStyle=rasterStyle,
            template=template,
            code=areacode,  # 使用9位代码
            mapinfo=mapinfo
            )
        
        # if result:
        #     print(f"专题图生成成功: {args.output}")
        # else:
        #     print("专题图生成失败")
        #     sys.exit(1)
            
    # except ImportError as e:
    #     print(f"无法导入qgis_plot模块: {e}")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"绘图过程中出错: {e}")
    #     sys.exit(1)


def mapinfo_name(time, pacname, cropname):
    if cropname == "corn":
        cropname="玉米"
    elif cropname == "potato":
        cropname="马铃薯"
    elif cropname == "sweet_potato":
        cropname="红薯"
    elif cropname == "rice":
        cropname="水稻"
        
    """生成地图信息字典"""
    mapinfo = {
        "title": f"{pacname}{cropname}种植气候区划图",
        "date": time
    }
    return mapinfo


if __name__ == "__main__":
    main()