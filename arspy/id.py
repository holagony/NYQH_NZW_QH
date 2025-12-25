#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/5/13 10:47
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : id.py
"""


# 产品名称
class Product:
    def __init__(self):
        # 大气环境 -- 气溶胶
        self.HAZE = {"var": "HAZE", "std": "HAZE--", "CHN": "灰霾气溶胶光学厚度", "EN": "Haze Aerosol Optical Depth"}
        self.AOD = {"var": "AOD", "std": "AOD---", "CHN": "气溶胶光学厚度", "EN": "Aerosol Optical Depth"}
        self.PM25 = {"var": "PM2.5", "std": "PM25--", "CHN": "PM2.5", "EN": "PM2.5"}
        self.PM10 = {"var": "PM10", "std": "PM10--", "CHN": "PM10", "EN": "PM10"}
        self.TSP = {"var": "TSP", "std": "TSP---", "CHN": "大气总悬浮颗粒物浓度", "EN": "Total Suspended Particulate"}
        self.AQI = {"var": "AQI", "std": "AQI---", "CHN": "空气质量指数", "EN": "Air Quality Index"}
        self.VIS = {"var": "VIS", "std": "VIS---", "CHN": "地面水平能见度", "EN": "Visibility"}
        self.CO = {"var": "CO", "std": "CO----", "CHN": "一氧化碳", "EN": "CO"}
        self.NO2 = {"var": "NO2", "std": "NO2---", "CHN": "二氧化氮", "EN": "NO2"}
        self.O3 = {"var": "O3", "std": "O3----", "CHN": "臭氧", "EN": "O3"}
        self.SO2 = {"var": "SO2", "std": "SO2---", "CHN": "二氧化硫", "EN": "SO2"}
        # 大气环境 -- 沙尘
        self.DSTDET = {"var": "DSTDET", "std": "DSTDET", "CHN": "沙尘判识", "EN": "Dust Detection"}
        self.IDDI = {"var": "IDDI", "std": "IDDI--", "CHN": "红外差值沙尘指数", "EN": "Infrared Difference Dust Index"}
        self.DSTSL = {"var": "DSTSL", "std": "DSTSL-", "CHN": "沙尘强度等级", "EN": "Dust Strength Level"}
        self.DSTVIS = {"var": "DSTVIS", "std": "DSTVIS", "CHN": "沙尘能见度", "EN": "Dust Visibility"}
        self.DSTAOD = {"var": "DSTAOD", "std": "DSTAOD", "CHN": "沙尘气溶胶光学厚度", "EN": "Dust Aerosol Optical Depth"}
        self.DSTLOD = {"var": "DSTLOD", "std": "DSTLOD", "CHN": "沙尘载沙量", "EN": "Dust Loading"}
        # 大气环境 -- 大雾
        self.FOGDET = {"var": "FOGDET", "std": "FOGDET", "CHN": "大雾判识", "EN": "Fog Detection"}
        self.FOGOPT = {"var": "FOGOPT", "std": "FOGOPT", "CHN": "大雾光学厚度", "EN": "Fog Optical Thickness"}
        self.FOGLWP = {"var": "FOGLWP", "std": "FOGLWP", "CHN": "大雾液态水路径", "EN": "Fog Liquid Water Path"}
        self.FOGEPR = {"var": "FOGEPR", "std": "FOGEPR", "CHN": "大雾有效粒子半径", "EN": "Fog Efficient Particle Radius"}
        self.FOGFUS = {"var": "FOGFUS", "std": "FOGFUS", "CHN": "大雾融合", "EN": "Fog Fusion"}
        self.FOGEF = {"var": "FOGEF", "std": "FOGEF-", "CHN": "大雾集合预报", "EN": "Fog Ensemble Forecast"}

        # 陆表生态 -- 地表温度
        self.LST = {"var": "LST", "std": "LST---", "CHN": "地表温度", "EN": "Landsat Surface Temperature"}
        self.UHI = {"var": "UHI", "std": "UHI---", "CHN": "城市热岛强度", "EN": "Urban Heat Island"}
        self.UTFD = {"var": "UTFD", "std": "UTFD--", "CHN": "城市热场分布", "EN": "Urban Thermal Field Distribution"}
        self.NUTFD = {"var": "NUTFD", "std": "NUTFD-", "CHN": "城市热场强度指数",
                      "EN": "Normalized Urban Thermal Field Distribution"}

        # 陆表生态 -- 干旱
        self.NDVI = {"var": "NDVI", "std": "NDVI--", "CHN": "归一化植被指数", "EN": "Normalized Differential Vegetation Index"}
        self.VFC = {"var": "VFC", "std": "VFC---", "CHN": "植被覆盖度", "EN": "Vegetation Fractional Coverage"}
        self.EVI = {"var": "EVI", "std": "EVI---", "CHN": "增强植被指数", "EN": "Enhanced Vegetation Index"}
        self.ATI = {"var": "ATI", "std": "ATI---", "CHN": "表观热惯量", "EN": "Apparent Thermal Inertia"}
        self.VSWI = {"var": "VSWI", "std": "VSWI--", "CHN": "植被供水指数", "EN": "Vegetation Supplication Water Index"}
        self.TVDI = {"var": "TVDI", "std": "TVDI--", "CHN": "温度植被干旱指数", "EN": "Temperature Vegetation Dryness Index"}
        self.VCI = {"var": "VCI", "std": "VCI---", "CHN": "植被状态指数", "EN": "Vegetation Condition Index"}
        self.TCI = {"var": "TCI", "std": "TCI---", "CHN": "温度条件指数", "EN": "Temperature Condition Index"}
        self.VHI = {"var": "VHI", "std": "VHI---", "CHN": "植被健康指数", "EN": "Vegetation Health Index"}
        self.CDI = {"var": "CDI", "std": "CDI---", "CHN": "综合干旱指数", "EN": "Composite Drought Index"}

        # 陆表生态 -- 火情

        # 天气 -- 云
        self.CLM = {"var": "CLM", "std": "CLM---", "CHN": "云检测", "EN": "Cloud Mask"}
        self.CTT = {"var": "CTT", "std": "CTT---", "CHN": "云顶温度", "EN": "Cloud Top Temperature"}
        self.CTP = {"var": "CTP", "std": "CTP---", "CHN": "云顶气压", "EN": "Cloud Top Pressure"}
        self.CTH = {"var": "CTH", "std": "CTH---", "CHN": "云顶高度", "EN": "Cloud Top Height"}
        self.CLC = {"var": "CLC", "std": "CLC---", "CHN": "云分类", "EN": "Cloud Classification"}
        self.CLP = {"var": "CLP", "std": "CLP---", "CHN": "云相态", "EN": "Cloud Phase"}
        self.COT = {"var": "COT", "std": "COT---", "CHN": "云光学厚度", "EN": "Cloud Optical Thickness"}
        self.CER = {"var": "CER", "std": "CER---", "CHN": "云有效粒子半径", "EN": "Cloud Efficient Radius"}
        # 气候


# 产品投影
class Proj:
    def __init__(self):
        self.GLL = "GLL"  # 等经纬投影
        self.NOM = "NOM"  # 标称投影
        self.NUL = "NUL"  # 无投影


# 产品周期
class Period:
    def __init__(self):
        self.HHMM = "HHMM"  # 实时
        self.HOUR = "POAH"  # 逐时
        self.DAY1 = "POAD"  # 逐日
        self.DAY5 = "POFD"  # 逐侯
        self.DAY7 = "POAW"  # 逐周
        self.DAY10 = "POTD"  # 逐旬
        self.DAY30 = "POAM"  # 逐月
        self.YEAR = "POAY"  # 逐年


# 产品格式
class DataFormat:
    def __init__(self):
        self.HDF = "HDF"
        self.NC = "NC"
        self.TIFF = "TIFF"
        self.PNG = "PNG"
        self.TXT = "TXT"
        self.CSV = "CSV"
        self.SHP = "SHP"


# 产品区域
class Region:
    def __init__(self):
        self.DISK = "DISK"
        self.NHEM = "NHEM"
        self.SHEM = "SHEM"
        self.REGC = "REGC"

class TF:
    def __init__(self):
        self.sumpre = {"evalution":70,
                        "weight":[0.04,0.16,0.33,0.47,1.0],
                        "threshold":[100,200,300,400] }

        self.maxpre = {"evalution":50,
                        "weight":[0.09,0.18,0.29,0.43,1.0],
                        "threshold":[100, 150, 200, 250] }

        self.maxwin = {"evalution":9,
                        "weight":[0.09,0.15,0.28,0.49,1.0],
                        "threshold":[10.7, 17.1, 24.4, 32.6] }

        self.preweight = 0.5

        self.winweight = 1.0 -self.preweight


# 实例化类
product = Product()
proj = Proj()
period = Period()
fmt = DataFormat()
region = Region()

TF_config = TF()
