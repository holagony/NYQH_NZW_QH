# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:12:07 2025

@author: HTHT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal
from scipy.ndimage import sobel
from pathlib import Path


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


def _sat_vapor_pressure(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))  # 饱和水汽压(kPa),T为气温(°C)


def _slope_delta(T):
    es = _sat_vapor_pressure(T)
    return 4098.0 * es / ((T + 237.3)**2)  # 饱和水汽压曲线斜率(kPa/°C)


def _pressure_from_elevation(z):
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0)**5.26  # 海拔高度z(m)处的大气压(kPa)


def _psychrometric_constant(P):
    return 0.000665 * P  # 湿度常数γ(kPa/°C)




class SPSO_ZH:
    '''
    黑龙江-大豆-灾害区划
    大豆干旱
    大豆冷害
    大豆霜冻 TODO
    大豆渍涝 TODO
    '''

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _interpolate_risk(self, data, station_coords, config, crop_config, type):
        interpolation = crop_config.get("interpolation")
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")

        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        print(f"使用 {interpolation_method} 方法对综合风险指数进行插值")

        # 准备插值数据
        interpolation_data = {'station_values': data, 'station_coords': station_coords, 'dem_path': config.get("demFilePath", ""), 'shp_path': config.get("shpFilePath", ""), 'grid_path': config.get("gridFilePath", ""), 'area_code': config.get("areaCode", "")}

        # 执行插值
        try:
            interpolated_result = interpolator.execute(interpolation_data, interpolation_params)
            print(f"{type}指数插值完成")
            # 保存中间结果
            self._save_intermediate_result(interpolated_result, config, type)

            return interpolated_result

        except Exception as e:
            print(f"{type}指数插值失败: {str(e)}")
            raise

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], indicator_name: str) -> None:
        """保存中间结果 - 各个指标的插值结果"""
        try:
            print(f"保存中间结果: {indicator_name}")

            # 生成中间结果文件名
            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            # 使用与最终结果相同的保存逻辑
            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                print(f"警告: 中间结果 {indicator_name} 格式不支持,跳过保存")
                return
            meta["nodata"] = -32768
            # 保存为GeoTIFF
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")
            # 不抛出异常,继续处理其他指标

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
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

    def _numpy_to_gdal_dtype(self, numpy_dtype: np.dtype) -> int:
        """将numpy数据类型转换为GDAL数据类型"""
        from osgeo import gdal

        dtype_map = {
            np.bool_: gdal.GDT_Byte,
            np.uint8: gdal.GDT_Byte,
            np.uint16: gdal.GDT_UInt16,
            np.int16: gdal.GDT_Int16,
            np.uint32: gdal.GDT_UInt32,
            np.int32: gdal.GDT_Int32,
            np.float32: gdal.GDT_Float32,
            np.float64: gdal.GDT_Float64,
            np.complex64: gdal.GDT_CFloat32,
            np.complex128: gdal.GDT_CFloat64
        }

        for np_type, gdal_type in dtype_map.items():
            if np.issubdtype(numpy_dtype, np_type):
                return gdal_type

        # 默认使用Float32
        print(f"警告: 无法映射numpy数据类型 {numpy_dtype}，默认使用GDT_Float32")
        return gdal.GDT_Float32
    
    def _perform_classification(self, data_interpolated, params):
        """分级计算"""
        print("执行区域分级计算...")
        
        algorithmConfig = params['algorithmConfig']
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(data_interpolated['data'], classification)
        
        data_interpolated['data'] = data
        return data_interpolated


    def calculate_ZL_HazardRisk(self, station_indicators, params):
        ZL = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            D50 = indicators.get('D50', np.nan)  # 总降水量
            D100 = indicators.get('D100', np.nan)  # 总日照时数
            D250 = indicators.get('D250', np.nan)  # 降水日数

            # 计算单站点致灾因子危险性指数
            result = 0.25 * D50 + 0.3 * D100 + 0.5 * D250
            ZL[station_id] = result

        filename = f'黑龙江大豆致灾因子危险性指数.csv'
        result_df = pd.DataFrame(list(ZL.items()), columns=['站点ID', '致灾因子危险性指数'])
        intermediate_dir = Path(params["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'黑龙江大豆致灾因子危险性指数站点级数据已保存至：{output_path}')
        return ZL


    def _calculate_ZL(self, params):
        '''
        计算大豆渍涝指数
        '''
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']
        self._algorithms = params['algorithms']

        print("开始计算大豆渍涝气候风险区划 ")
        print("第一步：站点级别计算致灾因子危险性指数")
        ZL_HazardRisk = self.calculate_ZL_HazardRisk(station_indicators, config)

        print("第二步：对致灾因子危险性指数进行插值")
        interpolated_ZL_HazardRisk = self._interpolate_risk(ZL_HazardRisk, station_coords, config, algorithmConfig, 'ZL_HazardRisk')
        print('第三步，基于插值栅格化指数进行区划分级')

        final_result = self._perform_classification(interpolated_ZL_HazardRisk, params)
        
        print(f'计算{params["config"].get("cropCode","")}-{params["config"].get("zoningType","")}-{params["config"].get("element","")}-区划完成')
        return final_result
    def calculate(self, params):
        config = params['config']
 
        return self._calculate_ZL(params)
