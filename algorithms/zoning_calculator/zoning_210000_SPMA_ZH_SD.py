# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:57:35 2025

@author: HTHT
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal

def calculate_Cf(daily_data):
    """
    霜冻气候区划评估指标
    """
    df = daily_data.copy()
    TMIN = df["tmin"]
    
    # 霜冻指标计算
    condition_spring = (TMIN.index.month == 5)|(TMIN.index.month == 4)
    condition_autumn = (TMIN.index.month == 10)|(TMIN.index.month == 9)
    
    frost1_spring = TMIN[condition_spring & (TMIN <= -1) & (TMIN > -2)]  # 轻霜冻
    frost1_autumn= TMIN[condition_autumn & (TMIN <= -1) & (TMIN > -2)]
    frost2_spring = TMIN[condition_spring & (TMIN <= -2) & (TMIN > -3)]  # 中霜冻
    frost2_autumn= TMIN[condition_autumn & (TMIN <= -2) & (TMIN > -3)]
    frost3_spring = TMIN[condition_spring & (TMIN <= -3) & (TMIN > -4.5)]  # 重霜冻
    frost3_autumn= TMIN[condition_autumn & (TMIN <= -3) & (TMIN > -4)]

    #年平均霜冻天数
    frost1_spring_mean_d = len(frost1_spring) / len(frost1_spring.index.year.unique()) if len(frost1_spring.index.year.unique()) > 0 else 0
    frost1_autumn_mean_d = len(frost1_autumn) / len(frost1_autumn.index.year.unique()) if len(frost1_autumn.index.year.unique()) > 0 else 0
    frost2_spring_mean_d = len(frost2_spring) / len(frost2_spring.index.year.unique()) if len(frost2_spring.index.year.unique()) > 0 else 0
    frost2_autumn_mean_d = len(frost2_autumn) / len(frost2_autumn.index.year.unique()) if len(frost2_autumn.index.year.unique()) > 0 else 0
    frost3_spring_mean_d = len(frost3_spring) / len(frost3_spring.index.year.unique()) if len(frost3_spring.index.year.unique()) > 0 else 0
    frost3_autumn_mean_d = len(frost3_autumn) / len(frost3_autumn.index.year.unique()) if len(frost3_autumn.index.year.unique()) > 0 else 0

    
    # 计算
    Cf = 0.5*(frost1_spring_mean_d*1+frost2_spring_mean_d*2+frost3_spring_mean_d*3)+0.5*(frost1_autumn_mean_d*1+frost2_autumn_mean_d*2+frost3_autumn_mean_d*3)
    
    return Cf

def normalize_grid_values(grid_data: np.ndarray) -> np.ndarray:
    """
    对栅格数据进行归一化到0-1范围
    """
    if grid_data.size == 0:
        return grid_data
    
    # 创建掩码，排除无效值
    valid_mask = ~np.isnan(grid_data)
    
    if not np.any(valid_mask):
        return np.zeros_like(grid_data)
    
    # 获取有效值
    valid_values = grid_data[valid_mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    
    # 初始化结果数组
    normalized_grid = np.zeros_like(grid_data)
    
    # 处理所有值相同的情况
    if max_val == min_val:
        normalized_grid[valid_mask] = 0.5
    else:
        # 归一化计算：最小值→0，最大值→1，中间值按比例计算
        normalized_grid[valid_mask] = (valid_values - min_val) / (max_val - min_val)
    
    # 无效值保持为0
    normalized_grid[~valid_mask] = 0.0
    
    return normalized_grid
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
class SPMA_ZH:
    '''
    辽宁-春玉米-灾害区划
    霜冻
    '''

    def _calculate_frost(self, params):
        """霜冻灾害风险指数模型 - 先插值后归一化"""
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
        #计算霜冻气候区划评估指标---------------------------------------------------------
        # 第一步：收集所有站点的Cf值（原始值）
        station_R_values = []
        station_data_map = {}  # 存储站点数据以便后续使用
        
        print("收集所有站点的R值...")
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            station_data_map[sid] = daily
            R = calculate_Cf(daily)
            station_R_values.append(R)
            #print(f"站点 {sid}: R={R:.4f}")
        
        # 第二步：使用原始R值进行插值
        station_values: Dict[str, float] = {}
        for i, sid in enumerate(station_ids):
            R_value = station_R_values[i]
            station_values[sid] = float(R_value) if np.isfinite(R_value) else np.nan
        
        # 插值配置
        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }
        max_value = max(station_values.values())
        min_value = min(station_values.values())
        print("最大值:"+str(max_value)+",最小值:"+str(min_value))
        iparams['max_value']=20
        iparams['min_value']=0
        #iparams['radius_dist']=2.0
        #iparams['min_num']=10
        #iparams['first_size']=min_value

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)
        
        # 第三步：对插值后的格点数据进行归一化
        raw_grid_data = result['data']
        print(f"插值后栅格数据统计 - 最小值: {np.nanmin(raw_grid_data):.4f}, 最大值: {np.nanmax(raw_grid_data):.4f}")
        
        # 归一化处理
        normalized_grid = normalize_grid_values(raw_grid_data)
        print(f"归一化后栅格数据统计 - 最小值: {np.nanmin(normalized_grid):.4f}, 最大值: {np.nanmax(normalized_grid):.4f}")
        #normalized_grid=raw_grid_data
        # 第四步：保存归一化后的结果
        result_path = cfg.get("resultPath")
        intermediate_dir = os.path.join(result_path, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        R_tif_path = os.path.join(intermediate_dir, "霜冻气候区划评估指标归一化结果.tif")   
        self._save_geotiff(normalized_grid, result['meta'], R_tif_path, 0)
        
        print(f"霜冻气候区划评估指标归一化结果已保存至: {R_tif_path}")

        # 分级--------------------------------------------------------------------------
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]    
        # 执行
        classdata = classificator.execute(normalized_grid, class_conf) 
     
        return {
            'data': classdata,
            'meta': {
                'width': result['meta']['width'],
                'height': result['meta']['height'],
                'transform': result['meta']['transform'],
                'crs': result['meta']['crs']
            },
            'type': '辽宁春玉米霜冻'
        }        
        
        
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
        
    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'drought':
            return self.calculate_drought(params)
        elif disaster_type == 'SD':
            return self._calculate_frost(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")