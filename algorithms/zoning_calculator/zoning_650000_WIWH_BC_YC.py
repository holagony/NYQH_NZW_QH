# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:12:07 2025

@author: HTHT
"""
from typing import Dict, Tuple
from algorithms.data_manager import DataManager
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from algorithms.interpolation.idw import IDWInterpolation
from typing import Dict, Any, Union, List
import os
from osgeo import gdal
import pandas as pd
import numpy as np
import time


def _mask_to_target_grid(mask_path, meta):
    src = gdal.Open(mask_path)
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', meta['width'], meta['height'], 1, gdal.GDT_Byte)
    dst.SetGeoTransform(meta['transform'])
    dst.SetProjection(meta['crs'])
    gdal.Warp(dst, src, resampleAlg=gdal.GRA_NearestNeighbour)
    arr = dst.GetRasterBand(1).ReadAsArray()
    return arr

def count_comfortable_days_YC(daily_data,start_date,end_date):
    """
    计算日平均温度为15～25℃且日平均相对湿度60%～70％的平均日数
    """
    tavg_level=[15,25]
    rhum_level=[60,70]
    # 从DataFrame中提取符合要求的数据
    mask_date=((daily_data.index.month==int(start_date[:2]))&(daily_data.index.day>=int(start_date[4:5])))|(daily_data.index.month==(int(start_date[:2])+1))|((daily_data.index.month==int(end_date[:2]))&(daily_data.index.day<=int(end_date[4:5])))
    comfortable_df = daily_data[(daily_data["tavg"]>=tavg_level[0])& (daily_data["tavg"]<=tavg_level[1])&
                    (daily_data["rhum"]>=rhum_level[0])& (daily_data["rhum"]<=rhum_level[1])&mask_date]
    if len(comfortable_df) == 0:
        print("没有找到符合条件的数据")
        return 0
    else:
    
      # 按年份统计舒适日数
      yearly_counts = comfortable_df.groupby(comfortable_df.index.year).size()
      
      # 计算年平均日数
      avg_days_per_year = yearly_counts.mean()   
      
      
      return avg_days_per_year

def update_thresholds_simple(thresholds_data,data1,data2,data3):
    """
    简单直接的修改方法
    """
    thresholds = thresholds_data.copy() if isinstance(thresholds_data, list) else thresholds_data
    
    # 定义替换映射
    replace_map = {
        0.175: data1,
        0.22: data2,
        0.28: data3  # 如果0.28也要改成0.3
    }
    
    for item in thresholds:
        # 修改min值
        if item["min"] != "":
            try:
                min_val = float(item["min"])
                if min_val in replace_map:
                    item["min"] = str(replace_map[min_val])
                else:
                    item["min"] = str(min_val)
            except ValueError:
                pass
        
        # 修改max值
        if item["max"] != "":
            try:
                max_val = float(item["max"])
                if max_val in replace_map:
                    item["max"] = str(replace_map[max_val])
                else:
                    item["max"] = str(max_val)
            except ValueError:
                pass
    
    return thresholds
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
    
    # 如果所有有效值都相同，归一化到0.5
    if max_val == min_val:
        normalized_array = np.full_like(array, 0.5, dtype=float)
        normalized_array[~mask] = np.nan
    else:
        normalized_array = (array - min_val) / (max_val - min_val)
        normalized_array[~mask] = np.nan
    
    return normalized_array        

class WIWH_BC:
    '''
    新疆-冬小麦-病虫区划
    蚜虫
    '''

    def calculate(self, params):
        config = params['config']
        pest_type = config['element']
        if pest_type == 'YC':
            return self.calculate_YC(params)
        else:
            raise ValueError(f"不支持的虫害类型: {pest_type}")
    def _get_algorithm(self, algorithm_name: str):
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def calculate_YC(self, params):
        '''
        新疆冬小麦蚜虫病虫区划
        '''
        # 获取配置参数
        self._algorithms=params.get("algorithms")
        station_indicators = params.get('station_indicators', {})
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        
        #计算指标-----------------------------------------------------------------------------------
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), 
                         multiprocess=False, num_processes=1)
        nan_nanjiang_beijiang=algorithm_config.get("nanjiang_beijiang",{})
        station_ids = [sid for sid in station_coords.keys() 
                      if sid in dm.get_all_stations()]
        # 构建地区映射关系   市：南疆
        region_mapping = {}
        for region_name, region_config in nan_nanjiang_beijiang.items():   #region_name南疆北疆，region_config：start_date\end_date\area
            for area in region_config['area']:
                region_mapping[area] = region_name
        print(region_mapping)
        #逐站点计算指标
        station_data = {}
        for sid in station_ids:
            city=dm.get_station_info(str(sid))["city"]#获取站点的市名称
            print(sid,city)
            station_region=region_mapping[city]
            region_config = nan_nanjiang_beijiang[station_region]
            start_date = region_config['start_date']  # "05-11"
            end_date = region_config['end_date']  # "07-10"   
            daily = dm.load_station_data(sid, cfg.get('startDate'), cfg.get('endDate'))
            station_value=count_comfortable_days_YC(daily,start_date,end_date)
            station_data[sid]=station_value
            
 
        #站点的值插值-------------------------------------------------------------------
        interp_data = {
            'station_values': station_data,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }
        
        if algorithm_config.get("interpolation")['method'] == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, algorithm_config.get("interpolation")['params'])
        else:
            result = IDWInterpolation().execute(interp_data, algorithm_config.get("interpolation")['params'])

        mask_path = cfg.get("maskFilePath")
        data = result["data"]
        if mask_path:
            mask_arr = _mask_to_target_grid(mask_path, result['meta'])
            data = np.where(mask_arr == 1, np.maximum(data, 0.0), np.nan)
        else:
            data = np.maximum(data, 0.0)

        #归一化-----------------------------------------------------------------------
        result_norm = normalize_array(data)
        result_path = cfg.get("resultPath")
        intermediate_dir = os.path.join(result_path, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        tif_path = os.path.join(intermediate_dir, "蚜虫病虫害未分级栅格数据.tif")
        self._save_geotiff(result_norm, result['meta'], tif_path, 0)   
        #分级------------------------------------------------------------------------------
        classification = algorithm_config['classification']
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        arr = result_norm[~np.isnan(result_norm)] 
        class_value=np.percentile(arr, [50,70,90])
        print("50%\70%\90%的阈值为",class_value)
        data = classifier.execute(result_norm, classification)
        result['data'] = data
        result["type"]="新疆冬小麦蚜虫病虫害"
        
        return result
        

    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        """保存GeoTIFF文件"""
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
        dataset = driver.Create(
            output_path,
            meta['width'],
            meta['height'],
            1,
            datatype,
            ['COMPRESS=LZW']
        )
        
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        
        band.FlushCache()
        dataset = None    
