# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:12:07 2025

@author: HTHT
"""
from typing import Dict, Tuple
from algorithms.data_manager import DataManager
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from algorithms.interpolation.kriging import KrigingInterpolation
import os
from osgeo import gdal
import pandas as pd
import numpy as np
import time

def calculate_seasonal_std(daily_data):
    """
    计算站点的季节变化标准差
    """
    # 从DataFrame中提取tavg列
    tavg_series = daily_data["tavg"]
    
    # 直接使用Series的索引获取月份和年份
    months = tavg_series.index.month
    years = tavg_series.index.year
    
    # 定义季节
    seasons = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8], 
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    
    seasonal_std = {}
    seasonal_data = {}
    
    for season_name, season_months in seasons.items():
        # 创建季节掩码
        season_mask = np.isin(months, season_months)
        season_temps = tavg_series[season_mask]
        
        # 计算标准差
        std_value = season_temps.std()
        seasonal_std[season_name] = std_value
        seasonal_data[season_name] = season_temps
    
    # 计算季节变化标准差
    seasonal_variation_std = np.mean(list(seasonal_std.values()))
    
    detailed_results = {
        'seasonal_std': seasonal_std,
        'seasonal_data': seasonal_data,
        'overall_std': tavg_series.std(),
        'data_years': f"{years.min()}-{years.max()}",
        'total_days': len(tavg_series)
    }
    
    return seasonal_variation_std, detailed_results
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

class CITR_BC:
    '''
    江西-柑橘-虫害区划
    木虱
    '''

    def calculate(self, params):
        config = params['config']
        pest_type = config['element']
        if pest_type == 'MS':
            return self.calculate_MS(params)
        else:
            raise ValueError(f"不支持的虫害类型: {pest_type}")


    def calculate_MS(self, params):
        '''
        木虱虫害区划
        '''
        # 获取配置参数
        station_indicators = params.get('station_indicators', {})
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        
        # 1. 计算站点级别的各个指标
        # 1.1 季节变化标准差
        season_std_values = self._calculate_station_season_std(cfg, station_coords)
        
        # 1.2 提取其他参数的站点值
        indicator_values = self._extract_station_indicators(station_indicators)
        
        # 2. 对站点值进行插值及归一化
        interp_results = self._interpolate_and_normalize_indicators(
            cfg, algorithm_config, station_coords, 
            season_std_values, indicator_values
        )
        
        # 3. 保存中间量
        self._save_intermediate_results(cfg, interp_results)
        
        # 4. 计算最终指标
        result = self._calculate_final_result(interp_results)
        
        # 5. 分级
        class_result = self._classify_result(params, algorithm_config, result,interp_results["D1"])
        
        return class_result
    
    def _calculate_station_season_std(self, cfg, station_coords):
        """计算站点的季节变化标准差"""
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), 
                         multiprocess=False, num_processes=1)
        
        station_ids = [sid for sid in station_coords.keys() 
                      if sid in dm.get_all_stations()]
        
        station_values = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, cfg.get('startDate'), cfg.get('endDate'))
            season_std, _ = calculate_seasonal_std(daily)
            station_values[sid] = float(season_std) if np.isfinite(season_std) else np.nan
        
        return station_values
    
    def _extract_station_indicators(self, station_indicators):
        """从站点指标中提取需要的参数"""
        indicators = {
            'P1': {}, 'T1': {}, 'D1': {}, 'T2': {}, 'D2': {}
        }
        
        for station_id, inds in station_indicators.items():
            indicators['P1'][station_id] = inds.get('P1', np.nan)
            indicators['T1'][station_id] = inds.get('T1', np.nan)
            indicators['D1'][station_id] = inds.get('D1', np.nan)
            indicators['T2'][station_id] = inds.get('T2', np.nan)
            indicators['D2'][station_id] = inds.get('D2', np.nan)
        
        return indicators
    
    def _interpolate_and_normalize_indicators(self, cfg, algorithm_config, station_coords, 
                                            season_std_values, indicator_values):
        """对所有指标进行插值和归一化"""
        interp_conf = algorithm_config.get('interpolation', {})
        iparams = interp_conf.get('params', {}).copy()
        iparams.setdefault('var_name', 'value')
        
        # 定义插值配置
        interpolation_configs = {
            'P1': {
                'values': indicator_values['P1'],
                'method': 'kriging',
                'iparams': {
                    "block_size": 256,
                    "radius_dist": 5.0,
                    "min_num": 10,
                    "first_size": 100,
                    "nodata": 0
                }
            },
            'season_std': {
                'values': season_std_values,
                'method': 'lsmidw',
                'iparams': iparams
            },
            'T1': {'values': indicator_values['T1'], 'method': 'lsmidw', 'iparams': iparams},
            'D1': {'values': indicator_values['D1'], 'method': 'lsmidw', 'iparams': iparams},
            'T2': {'values': indicator_values['T2'], 'method': 'lsmidw', 'iparams': iparams},
            'D2': {'values': indicator_values['D2'], 'method': 'lsmidw', 'iparams': iparams}
        }
        
        results = {}
        for key, config in interpolation_configs.items():
            interp_data = {
                'station_values': config['values'],
                'station_coords': station_coords,
                'grid_path': cfg.get('gridFilePath'),
                'dem_path': cfg.get('demFilePath'),
                'area_code': cfg.get('areaCode'),
                'shp_path': cfg.get('shpFilePath')
            }
            
            if config['method'] == 'kriging':
                result = KrigingInterpolation().execute(interp_data, config['iparams'])
            else:
                result = LSMIDWInterpolation().execute(interp_data, config['iparams'])
            print(key+"插值完成")
            results[key] = {
                'data': result['data'],
                'norm': normalize_array(result['data']),
                'meta': result['meta']
            }
            time.sleep(3)
         
        return results
    
    def _save_intermediate_results(self, cfg, interp_results):
        """保存中间结果文件"""
        result_path = cfg.get("resultPath")
        intermediate_dir = os.path.join(result_path, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        file_mappings = {
            'P1': "最冷季节总降水量P1.tif",
            'T1': "最冷季节平均气温T1.tif", 
            'D1': "冬季平均气温小于0天数D1.tif",
            'T2': "最暖月最高气温T2.tif",
            'D2': "8月份平均气温大于30度天数D2.tif",
            'season_std': "季节变化标准差.tif"
        }
        
        for key, filename in file_mappings.items():
            if key in interp_results:
                tif_path = os.path.join(intermediate_dir, filename)
                self._save_geotiff(interp_results[key]['norm'], 
                                 interp_results[key]['meta'], tif_path, 0)
                print(f"{filename.split('.')[0]}: {tif_path}")
    
    def _calculate_final_result(self, interp_results):
        """计算最终的综合指标"""
        weights = {
            'P1': 0.2, 'T1': 0.03, 'D1': 0.71, 
            'season_std': 0.03, 'T2': 0.01, 'D2': 0.02
        }
        
        result = np.zeros_like(interp_results['P1']['norm'])
        for key, weight in weights.items():
            if key in interp_results:
                result += weight * interp_results[key]['norm']
        
        return result
    
    def _classify_result(self, params, algorithm_config, result,d1data):
        """对结果进行分级"""
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]
        
        classdata = classificator.execute(result, class_conf)
        
        # 使用任意一个插值结果的元数据
        meta_source = d1data["meta"]
        
        return {
            'data': classdata,
            'meta': {
                'width': classdata.shape[1],
                'height': classdata.shape[0],
                'transform': meta_source['transform'],
                'crs': meta_source['crs']
            },
            'type': '广西柑橘木虱病虫害'
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
