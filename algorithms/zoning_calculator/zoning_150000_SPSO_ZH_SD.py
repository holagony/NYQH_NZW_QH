# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:12:07 2025

@author: HTHT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from algorithms.interpolation.lsm import LSMInterpolation
import os
from osgeo import gdal
from scipy.ndimage import sobel
import sys

def alignRePrjRmp(file1, file2, outfile, srs_nodata=None, dst_nodata=None,
                  resample_type=gdal.GRA_NearestNeighbour):
    '''
    根据参考影像，进行重采样，包含重投影，行列一致
    :param file1: str, 基准参考影像
    :param file2: str, 需要修改的栅格影像
    :param outfile: str, 结果输出文件
    :param nodata:float,无效值
    :param resample_type:ds，重采样方法
    :return:
    @version <1> LiYaobin 2019-10-08  : created
    '''
    try:
        try:
            refnds = gdal.Open(file1)
        except:
            refnds = file1
        base_xsize = refnds.RasterXSize
        base_ysize = refnds.RasterYSize
        base_proj = refnds.GetProjection()
        base_gt = refnds.GetGeoTransform()
        xmin = base_gt[0]
        ymax = base_gt[3]
        xmax = xmin + base_xsize * base_gt[1]
        ymin = ymax + base_ysize * base_gt[5]
        try:
            warpds = gdal.Open(file2)
        except:
            warpds = file2
        if srs_nodata is None:
            band = warpds.GetRasterBand(1)
            srs_nodata = band.GetNoDataValue()
        if outfile is None:
            dataset = gdal.Warp("",
                                warpds,
                                width=base_xsize,
                                height=base_ysize,
                                srcNodata=srs_nodata,
                                dstNodata=dst_nodata,
                                dstSRS=str(base_proj),
                                outputBounds=(xmin, ymin, xmax, ymax),
                                format="MEM",
                                resampleAlg=resample_type)
        else:
            dataset = gdal.Warp(outfile,
                                warpds,
                                width=base_xsize,
                                height=base_ysize,
                                srcNodata=srs_nodata ,
                                dstNodata=dst_nodata,
                                dstSRS=str(base_proj),
                                outputBounds=(xmin, ymin, xmax, ymax),
                                format="GTiff",
                                resampleAlg=resample_type)
        return dataset
    finally:
        warpds = None
        refnds = None
def Dv(frost):
    if len(frost) > 0:
        day_of_year = frost.index.dayofyear
        mean_value = np.mean(day_of_year)
        std_dev = np.std(day_of_year)
        Dv_value = std_dev / mean_value if mean_value != 0 else 0
    else:
        Dv_value = 0  
    return Dv_value
#转化日序
def mdd_to_day_of_year(date_str, year=2024):
    """
    将'月日日'格式转换为日序
    
    参数:
        date_str: 字符串，如'530'表示5月30日
        year: 年份，默认2024
    
    返回:
        day_of_year: 一年中的第几天
    """
    # 确保输入是字符串
    date_str = str(date_str)
    
    # 处理长度
    if len(date_str) == 3:
        month = int(date_str[0])      # 月份（1位）
        day = int(date_str[1:3])      # 日期（2位）
    elif len(date_str) == 4:
        month = int(date_str[0:2])    # 月份（2位）
        day = int(date_str[2:4])      # 日期（2位）
    else:
        raise ValueError(f"无效的日期格式: {date_str}")
    
    # 验证日期有效性
    from datetime import datetime
    try:
        date_obj = datetime(year, month, day)
        return date_obj.timetuple().tm_yday
    except ValueError as e:
        raise ValueError(f"无效的日期: {month}月{day}日 - {e}")
# 霜冻致灾危险性风险性指数
#霜冻强度频率指数计算
def calculate_Ih_Dv(daily_data,chumiao_80,chengshu_80):
    """
    输入：逐日数据，该站点80%保证率出苗期日序，80%保证率成熟期日序
    计算得到站点春季5日10日15日的Ih和Dv;得到秋季5日10日的Ih和Dv，一共10个量
    
    
    
    计算致灾因子危险性指数需要计算霜冻指标的Ih和Dv值
    80%保证率出苗期后5天、10天、15天的致灾因子危险性指数，权重0.5、0.3、0.2
    80%保证率成熟期前5天、前10天的致灾因子危险性指数，权重0.4、0.6
    输出：综合播种0.4成熟0.6致灾因子危险性指数
    """
    df = daily_data.copy()
    TMIN = df["tmin"]
    spring_temp=[-1,-2,-3,-4.5]
    autumn_temp=[0.5,0,-1,-2.5]
    SDlist=["轻霜冻","中霜冻","重霜冻"]
    years=len(daily_data.index.year.unique())
    #播种期------------------------------------------------------------------------------------
    condition_spring_5 = (TMIN.index.dayofyear>=int(chumiao_80)) & (TMIN.index.dayofyear<(int(chumiao_80)+5))
    condition_spring_10 = (TMIN.index.dayofyear>=int(chumiao_80)) & (TMIN.index.dayofyear<(int(chumiao_80)+10))
    condition_spring_15 = (TMIN.index.dayofyear>=int(chumiao_80)) & (TMIN.index.dayofyear<(int(chumiao_80)+15))
    #计算Ih
    Ih_spring_5=0
    Ih_spring_10=0
    Ih_spring_15=0
    Dv_spring_5=0
    Dv_spring_10=0
    Dv_spring_15=0
    for i in range(len(SDlist)):
        frost_spring_5=TMIN[condition_spring_5 & (TMIN < spring_temp[i]) & (TMIN >= spring_temp[i+1])] 
        frost_spring_10=TMIN[condition_spring_10 & (TMIN < spring_temp[i]) & (TMIN >= spring_temp[i+1])]
        frost_spring_15=TMIN[condition_spring_15 & (TMIN < spring_temp[i]) & (TMIN >= spring_temp[i+1])]
        if len(frost_spring_5.index.year.unique())>0:
            temp_5=len(frost_spring_5)/years*(frost_spring_5.sort_values().median())/years
            Dv_temp_5=Dv(frost_spring_5)
        else:
            temp_5=0
            Dv_temp_5=0
        if len(frost_spring_10.index.year.unique())>0:
            temp_10=len(frost_spring_10)/years*(frost_spring_10.sort_values().median())/years
            Dv_temp_10=Dv(frost_spring_10)
        else:
            temp_10=0
            Dv_temp_10=0
        if len(frost_spring_15.index.year.unique())>0:
            temp_15=len(frost_spring_15)/years*(frost_spring_15.sort_values().median())/years
            Dv_temp_15=Dv(frost_spring_15)
        else:
            temp_15=0
            Dv_temp_15=0
        Ih_spring_5=Ih_spring_5+temp_5
        Ih_spring_10=Ih_spring_10+temp_10
        Ih_spring_15=Ih_spring_15+temp_15
        Dv_spring_5=Dv_spring_5+Dv_temp_5
        Dv_spring_10=Dv_spring_10+Dv_temp_10
        Dv_spring_15=Dv_spring_15+Dv_temp_15

    #然后春季霜冻变异系数需要分别求三种霜冻类型的变异系数，求平均得到春季霜冻变异系数
    Dv_spring_5=Dv_spring_5/3
    Dv_spring_10=Dv_spring_10/3
    Dv_spring_15=Dv_spring_15/3

        # frost_spring_5_all=pd.concat([frost_spring_5_all, frost_spring_5])
        # frost_spring_10_all=pd.concat([frost_spring_10_all, frost_spring_10])
        # frost_spring_15_all=pd.concat([frost_spring_15_all, frost_spring_15])
    #需要分别对春季5日、10日、15日的霜冻强度频率指数求绝对值，然后全场做归一化
    

    #计算气候危险性风险指数
    
    # W_spring_5=0.75*Ih_spring_5+0.25*Dv(frost_spring_5_all)
    # W_spring_10=0.75*Ih_spring_10+0.25*Dv(frost_spring_10_all)
    # W_spring_15=0.75*Ih_spring_15+0.25*Dv(frost_spring_15_all)
    # W_spring=0.5*W_spring_5+0.3*W_spring_10+0.2*W_spring_15
    
    
    #成熟期--------------------------------------------------------------------------------------
    condition_autumn_5=(TMIN.index.dayofyear<=int(chengshu_80)) & (TMIN.index.dayofyear>(int(chengshu_80)-5))
    condition_autumn_10=(TMIN.index.dayofyear<=int(chengshu_80)) & (TMIN.index.dayofyear>(int(chengshu_80)-10))
    
    #计算Ih
    Ih_autumn_5=0
    Ih_autumn_10=0
    Dv_autumn_5=0
    Dv_autumn_10=0
    
    for i in range(len(SDlist)):
        frost_autumn_5=TMIN[condition_autumn_5 & (TMIN < autumn_temp[i]) & (TMIN >=autumn_temp[i+1])] 
        frost_autumn_10=TMIN[condition_autumn_10 & (TMIN < autumn_temp[i]) & (TMIN >= autumn_temp[i+1])]
        if len(frost_autumn_5.index.year.unique())>0:
            temp_5=len(frost_autumn_5)/years*(frost_autumn_5.sort_values().median())/years
            Dv_temp_5=Dv(frost_autumn_5)
            
        else:
            temp_5=0
            Dv_temp_5=0
        if len(frost_autumn_10.index.year.unique())>0:
            temp_10=len(frost_autumn_10)/years*(frost_autumn_10.sort_values().median())/years
            Dv_temp_10=Dv(frost_autumn_10)
        else:
            temp_10=0
            Dv_temp_10=0
        Ih_autumn_5=Ih_autumn_5+temp_5
        Ih_autumn_10=Ih_autumn_10+temp_10
        Dv_autumn_5=Dv_autumn_5+Dv_temp_5
        Dv_autumn_10=Dv_autumn_10+Dv_temp_10
    #然后春季霜冻变异系数需要分别求三种霜冻类型的变异数，求平均得到春季霜冻变异系数,秋季霜冻也一样
    Dv_autumn_5=Dv_autumn_5/3
    Dv_autumn_10=Dv_autumn_10/3
        
        
        
    #     frost_autumn_5_all=pd.concat([frost_autumn_5_all, frost_autumn_5])
    #     frost_autumn_10_all=pd.concat([frost_autumn_10_all, frost_autumn_10])
    # #计算气候危险性风险指数
    # W_autumn_5=0.75*Ih_autumn_5+0.25*Dv(frost_autumn_5_all)
    # W_autumn_10=0.75*Ih_autumn_10+0.25*Dv(frost_autumn_10_all)

    # W_autumn=0.4*W_autumn_5+0.6*W_autumn_10 
    
    
    # #综合致灾因子危险性指数--------------------------------------------
    # W=0.4*W_spring+0.6*W_autumn
    
    return Ih_spring_5,Ih_spring_10,Ih_spring_15,Ih_autumn_5,Ih_autumn_10,Dv_spring_5,Dv_spring_10,Dv_spring_15,Dv_autumn_5,Dv_autumn_10
    
   

def normalize_values(values: List[float]) -> List[float]:
    """
    归一化数值到0-1范围
    """
    if not values:
        return []

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_values:
        return [0.0] * len(values)

    min_val = min(valid_values)
    max_val = max(valid_values)

    # 如果所有值都相同，归一化到0.5
    if max_val == min_val:
        return [0.5 if v is not None and not np.isnan(v) else 0.0 for v in values]

    normalized = []
    for v in values:
        if v is None or np.isnan(v):
            normalized.append(0.0)
        else:
            norm_val = (v - min_val) / (max_val - min_val)
            normalized.append(norm_val)

    return normalized


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    归一化数组到0-1范围
    """
    if array.size == 0:
        return array

    # 创建一个掩码来标识非NaN值
    mask = ~np.isnan(array)

    if not np.any(mask):
        return np.zeros_like(array)

    valid_values = array[mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    normalized_array = (array - min_val) / (max_val - min_val)
    normalized_array[~mask] = np.nan

    return normalized_array


class SPSO_ZH:
    '''
    内蒙古-大豆-灾害区划
    干旱和霜冻
    '''

    def _calculate_frost(self, params):
        """霜冻灾害风险指数模型"""
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')

        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())

        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')

        # 第一步：收集所有站点的W值
        
        station_Ih_spring_5_values = []
        station_Ih_spring_10_values = []
        station_Ih_spring_15_values = []
        station_Ih_autumn_5_values = []
        station_Ih_autumn_10_values = []
        station_Dv_spring_5_values = []
        station_Dv_spring_10_values = []
        station_Dv_spring_15_values = []
        station_Dv_autumn_5_values = []
        station_Dv_autumn_10_values = []        
        
        #-------------------------------------------

        station_data_map = {}  # 存储站点数据以便后续使用

        print("收集所有站点的Ih和Dv值...")
        #7大气候区站点分布
        station_=os.path.join(cfg.get('dependDir'),"7大气候区站点分布_内蒙古.csv")
        print(station_)
        if os.path.exists(station_):
            # 读取CSV文件
            station_qihou = pd.read_csv(station_,encoding="gbk")
        else:
            print("7大气候区站点分布_内蒙古.csv文件不存在")
            sys.exit(1)

        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            station_data_map[sid] = daily
            #80%出苗期，80%成熟期日序
            chumiao_80=station_qihou[station_qihou["站号"]==int(sid)]["80%出苗期"].iloc[0]
            chengshu_80=station_qihou[station_qihou["站号"]==int(sid)]["80%成熟期"].iloc[0]
            chumiao_80=mdd_to_day_of_year(str(chumiao_80), year=2024)
            chengshu_80=mdd_to_day_of_year(str(chengshu_80), year=2024)
            
            Ih_spring_5,Ih_spring_10,Ih_spring_15,Ih_autumn_5,Ih_autumn_10,Dv_spring_5,Dv_spring_10,Dv_spring_15,Dv_autumn_5,Dv_autumn_10 = calculate_Ih_Dv(daily,chumiao_80,chengshu_80)
            station_Ih_spring_5_values.append(Ih_spring_5)
            station_Ih_spring_10_values.append(Ih_spring_10)
            station_Ih_spring_15_values.append(Ih_spring_15)
            station_Ih_autumn_5_values.append(Ih_autumn_5)
            station_Ih_autumn_10_values.append(Ih_autumn_10)
            
            station_Dv_spring_5_values.append(Dv_spring_5)
            station_Dv_spring_10_values.append(Dv_spring_10)
            station_Dv_spring_15_values.append(Dv_spring_15)
            station_Dv_autumn_5_values.append(Dv_autumn_5)
            station_Dv_autumn_10_values.append(Dv_autumn_10) 
            print("站点"+str(sid)+"的Ih和Dv计算完成")
        #将各站点的5日、10日、15日Ih进行求绝对值
        station_Ih_spring_5_values_abs = [abs(x) for x in station_Ih_spring_5_values]
        station_Ih_spring_10_values_abs = [abs(x) for x in station_Ih_spring_10_values]
        station_Ih_spring_15_values_abs = [abs(x) for x in station_Ih_spring_15_values]
        station_Ih_autumn_5_values_abs = [abs(x) for x in station_Ih_autumn_5_values]
        station_Ih_autumn_10_values_abs = [abs(x) for x in station_Ih_autumn_10_values]
        ##Ih归一化
        
        Ih_spring_5_nor=np.array(normalize_values(station_Ih_spring_5_values_abs))
        Ih_spring_10_nor=np.array(normalize_values(station_Ih_spring_10_values_abs))
        Ih_spring_15_nor=np.array(normalize_values(station_Ih_spring_15_values_abs))
        Ih_autumn_5_nor=np.array(normalize_values(station_Ih_autumn_5_values_abs))
        Ih_autumn_10_nor=np.array(normalize_values(station_Ih_autumn_10_values_abs))

        print("Ih归一化完成")
        ##Dv归一化
        #归一化省略
#        Dv_spring_5_nor=np.array(normalize_values(station_Dv_spring_5_values))
#        Dv_spring_10_nor=np.array(normalize_values(station_Dv_spring_10_values))
#        Dv_spring_15_nor=np.array(normalize_values(station_Dv_spring_15_values))
#        Dv_autumn_5_nor=np.array(normalize_values(station_Dv_autumn_5_values))
#        Dv_autumn_10_nor=np.array(normalize_values(station_Dv_autumn_10_values)) 
        Dv_spring_5_nor=np.array(station_Dv_spring_5_values)*100
        Dv_spring_10_nor=np.array(station_Dv_spring_10_values)*100
        Dv_spring_15_nor=np.array(station_Dv_spring_15_values)*100
        Dv_autumn_5_nor=np.array(station_Dv_autumn_5_values)*100
        Dv_autumn_10_nor=np.array(station_Dv_autumn_10_values)*100
        print("Dv归一化完成")       
        
        #计算危险性指数W
        W_spring_5=0.75*Ih_spring_5_nor+0.25*Dv_spring_5_nor
        W_spring_10=0.75*Ih_spring_10_nor+0.25*Dv_spring_10_nor
        W_spring_15=0.75*Ih_spring_15_nor+0.25*Dv_spring_15_nor
        W_autumn_5=0.75*Ih_autumn_5_nor+0.25*Dv_autumn_5_nor
        W_autumn_10=0.75*Ih_autumn_10_nor+0.25*Dv_autumn_10_nor
        W_spring_5=normalize_array(W_spring_5)
        W_spring_10=normalize_array(W_spring_10)
        W_spring_15=normalize_array(W_spring_15)
        W_autumn_5=normalize_array(W_autumn_5)
        W_autumn_10=normalize_array(W_autumn_10)
        
        W_spring=0.5*W_spring_5+0.3*W_spring_10+0.2*W_spring_15
        W_autumn=0.4*W_autumn_5+0.6*W_autumn_10
        
        W=W_spring*0.4+W_autumn*0.6
        
#        df = pd.DataFrame(list(zip(station_ids,list(W),list(W_spring),list(W_autumn),station_Ih_spring_5_values_abs,list(Dv_spring_5_nor))),
#                  columns=['stationid', 'W', 'W_spring', 'W_autumn','station_Ih_spring_5_values_abs','Dv_spring_5_nor'])
#        df.to_csv( os.path.join(cfg.get("resultPath"), "intermediate", "W_Ih_Dv.csv"))
        
            
        # 第二步：对所有站点的Ih和Dv进行归一化
        normalized_W = W
        #这一步省略

        # 第三步：使用归一化后的Ih和Dv计算每个站点的W值
        station_values: Dict[str, float] = {}
        for i, sid in enumerate(station_ids):
            W_value= normalized_W[i]
            station_values[sid] = float(W_value) if np.isfinite(W_value) else 0

        # 第四步：插值计算
        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})
        iparams["min_value"]=0
        iparams["radius_dist"]=1.0
        iparams["min_num"]=5
        iparams["first_size"]=50

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath'),
            "CZT_CRX":cfg.get('vulFilePath'),     ######承灾体脆弱性
            "FZJZNL":cfg.get('dpamFilePath')      #防灾减灾能力
        }

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        elif method== "LSM":
            result = LSMInterpolation().execute(interp_data, iparams)
        elif method=="idw":
            result = IDWInterpolation().execute(interp_data, iparams)
            print("进行反距离权重插值")
        W_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "归一化后的致灾危险性指数.tif")
        self._save_geotiff(result['data'], result['meta'], W_tif_path, -99999)  # 保存致灾危险性指数tif

        interpolated_W_value = result["data"]

        # 计算敏感性指数
        M_value = self.M(params)
        M_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "归一化后的孕灾环境敏感性指数.tif")
        self._save_geotiff(M_value, result['meta'], M_tif_path, -99999)  # 保存孕灾环境敏感性指数tif

        # 霜冻承灾体暴露度指数C，承载体脆弱性指标(大豆种植面积比例栅格数据)
        ZZ_percent_path = interp_data["CZT_CRX"]
        ZZ_temp_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/ZZ_temp.tif"
        ZZ_temp_path = LSMIDWInterpolation()._align_datasets(interp_data["grid_path"], ZZ_percent_path, ZZ_temp_path)
        in_ds_C = gdal.Open(ZZ_temp_path)
        C_array = in_ds_C.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_C.GetRasterBand(1).GetNoDataValue()
        C_array = np.where(C_array == Nodata, 0, C_array)
        C_array = normalize_array(C_array)

        # 霜冻防灾减灾能力指数F(灌溉面积百分比)
        GG_percent_path = interp_data["FZJZNL"]
        GG_temp_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/GG_temp.tif"
        GG_temp_path = LSMIDWInterpolation()._align_datasets(interp_data["grid_path"], GG_percent_path, GG_temp_path)
        in_ds_F = gdal.Open(GG_temp_path)
        F_array = in_ds_F.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_F.GetRasterBand(1).GetNoDataValue()
        F_array = np.where(F_array == Nodata, 0, F_array)
        F_array = normalize_array(F_array)

        # FRI计算
        FRI = interpolated_W_value * 0.593 + C_array * 0.255 + M_value * 0.106 + F_array * 0.046
        FRI=np.where(FRI<0,0,FRI)

        # 分级
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]
        # 执行
        classdata = classificator.execute(FRI, class_conf)
        os.remove(ZZ_temp_path)
        os.remove(GG_temp_path)
        return {
            'data': classdata,
            'meta': {
                'width': result['meta']['width'],
                'height': result['meta']['height'],
                'transform': result['meta']['transform'],
                'crs': result['meta']['crs']
            },
            'type': '内蒙古大豆霜冻'
        }

    def M(self, params):
        # 环境敏感性指数
        config = params['config']
        dem_path = config.get("demFilePath", "")
        grid_path = config.get("gridFilePath", "")
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp.tif')
        
        alignRePrjRmp(grid_path, dem_path, temp_path,srs_nodata=None, dst_nodata=-32768, resample_type=gdal.GRA_NearestNeighbour)
        
        in_ds_dem = gdal.Open(temp_path)
        gtt = in_ds_dem.GetGeoTransform()
        alti_array = in_ds_dem.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_dem.GetRasterBand(1).GetNoDataValue()
        
        in_ds_grid = gdal.Open(grid_path)
        grid_array = in_ds_grid.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata_grid = in_ds_grid.GetRasterBand(1).GetNoDataValue()        
        # 海拔
        alti_array = np.where(alti_array == Nodata, np.nan, alti_array)
        alti_array = np.where(grid_array == Nodata_grid, np.nan, alti_array)
        # 求纬度
        width = in_ds_dem.RasterXSize  # cols
        height = in_ds_dem.RasterYSize  # rows
        x = np.linspace(gtt[0], gtt[0] + gtt[1] * width, width)
        y = np.linspace(gtt[3], gtt[3] + gtt[5] * height, height)
        lon, lat = np.meshgrid(x, y)
        # 计算坡向aspect
        # 使用Sobel算子计算水平和垂直方向的梯度
        dz_dx = sobel(alti_array, axis=1) / (gtt[1] * 111320 * np.cos(np.radians(lat)))
        dz_dy = sobel(alti_array, axis=0) / (gtt[1] * 111320 * np.cos(np.radians(lat)))
        aspect = np.arctan2(dz_dy, dz_dx)
        aspect = np.where(alti_array == Nodata, np.nan, aspect)
        # 赋值
        alti_array_ = np.zeros_like(alti_array)
        alti_array_ = np.where(alti_array < 200, 1, alti_array_)
        alti_array_ = np.where((alti_array >= 200) & (alti_array < 400), 2, alti_array_)
        alti_array_ = np.where((alti_array >= 400) & (alti_array < 600), 3, alti_array_)
        alti_array_ = np.where((alti_array >= 600) & (alti_array < 800), 4, alti_array_)
        alti_array_ = np.where(alti_array >= 800, 5, alti_array_)
        alti_array_ = np.where(alti_array == Nodata, np.nan, alti_array_)

        aspect_ = np.zeros_like(aspect)
        aspect_ = np.where((aspect > 45) & (aspect <= 135), 1, aspect_)
        aspect_ = np.where((aspect >= 135) & (aspect < 225), 2, aspect_)
        aspect_ = np.where((aspect >= 225) & (aspect < 315), 3, aspect_)
        aspect_ = np.where((aspect >= 315) & (aspect <= 360), 4, aspect_)
        aspect_ = np.where(aspect <= 45, 5, aspect_)
        aspect_ = np.where(aspect == Nodata, np.nan, aspect_)

        # 对alti_array_和aspect_进行归一化处理
        alti_array_norm = normalize_array(alti_array_)
        aspect_norm = normalize_array(aspect_)

        # 输出归一化前后的统计信息
        print(f"海拔等级原始范围: {np.nanmin(alti_array_):.1f} ~ {np.nanmax(alti_array_):.1f}")
        print(f"坡向等级原始范围: {np.nanmin(aspect_):.1f} ~ {np.nanmax(aspect_):.1f}")
        print(f"海拔等级归一化范围: {np.nanmin(alti_array_norm):.4f} ~ {np.nanmax(alti_array_norm):.4f}")
        print(f"坡向等级归一化范围: {np.nanmin(aspect_norm):.4f} ~ {np.nanmax(aspect_norm):.4f}")

        # 使用归一化后的值计算M_value
        M_value = 0.833 * alti_array_norm + 0.167 * aspect_norm

        os.remove(temp_path)
        return M_value

    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        """保存GeoTIFF文件"""
        from osgeo import gdal

        # 根据输入数据的 dtype 确定 GDAL 数据类型
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.uint16:
            datatype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            datatype = gdal.GDT_Int16
        elif data.dtype == np.uint32:
            datatype = gdal.GDT_UInt32
        elif data.dtype == np.int32:
            datatype = gdal.GDT_Int32
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32  # 默认情况

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])

        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])

        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)

        band.FlushCache()
        dataset = None

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'GH':
            return self.calculate_drought(params)
        elif disaster_type == 'SD':
            return self._calculate_frost(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
