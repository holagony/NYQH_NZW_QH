import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
from algorithms.data_manager import DataManager
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from algorithms.indicators import IndicatorCalculator
from pathlib import Path
import os
import json
import datetime


class WIWH_ZH:
    """
    河南冬小麦干热风区划主类
    基于JSON配置文件实现干热风风险区划功能
    """
    
    def __init__(self):
        """
        初始化WIWH_ZH类
        """

        
        # 干热风等级权重和标记值
        self.WEIGHTS = {
            'GRF_light': 0.35,    # 轻度权重
            'GRF_severe': 0.65     # 重度权重
        }
        

    def calculate_GRF_index(self, station_indicators: pd.DataFrame, config: Dict[str, Any]) -> Dict[int, float]:
        """
        计算干热风强度指数R
        
        计算公式：R = Σ(Wi * Ni)
        其中：Wi为权重，Di为标记值，Ni为干热风日数
        """
        GRF_indicators = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            GRF_light = indicators.get('GRF_light', np.nan)  # 总降水量
            GRF_severe = indicators.get('GRF_severe', np.nan)  # 降水日数

            # str转字典
            GRF_light = pd.DataFrame.from_dict(GRF_light, orient='index')
            GRF_severe = pd.DataFrame.from_dict(GRF_severe, orient='index')

            merged_df = pd.concat([GRF_light, GRF_severe], axis=1)
            merged_df.columns = ['GRF_light', 'GRF_severe']

            '''
            merged_df['GRF_index'] = merged_df['GRF_light'] * self.WEIGHTS['GRF_light'] * self.MARKS['GRF_light'] + \
                                     merged_df['GRF_moderate'] * self.WEIGHTS['GRF_moderate'] * self.MARKS['GRF_moderate'] + \
                                     merged_df['GRF_severe'] * self.WEIGHTS['GRF_severe'] * self.MARKS['GRF_severe']

            GRF_indicators[station_id] = merged_df['GRF_index'].mean()
            '''
            GRF_indicators[station_id] = merged_df['GRF_light'].mean() * self.WEIGHTS['GRF_light']+ \
                                     merged_df['GRF_severe'].mean() * self.WEIGHTS['GRF_severe']
        max_value = max(GRF_indicators.values())
        max_keys = [key for key, value in GRF_indicators.items() if value == max_value]
        min_value = min(GRF_indicators.values())
        min_keys = [key for key, value in GRF_indicators.items() if value == min_value]
        print(f'河南冬小麦干热风区划：单站最高干热风强度指数：{max_keys}：{max_value}')
        print(f'河南冬小麦干热风区划:单站最低干热风强度指数：{min_keys}：{min_value}')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'干热风强度指数_{timestamp}.csv'
        result_df = pd.DataFrame(list(GRF_indicators.items()), columns=['站点ID', '干热风强度指数'])   
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"干热风强度指数文件已保存为 '{output_path}'")

        return GRF_indicators

    def _get_algorithm(self, algorithm_name):
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name,min_value=np.nan,max_value=np.nan):
        """对指定指标进行插值计算"""
        print(f"执行{indicator_name}插值计算...")
        
        config = params['config']        
        # 获取插值算法
        algorithmConfig = params['algorithmConfig']
        interpolation_config = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        
        # 准备插值参数
        interpolation_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value':min_value,
            'max_value':max_value,
        }
        
        file_name = f"intermediate_{indicator_name}.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name
        
        if not os.path.exists(output_path):
            interpolated_result = interpolator.execute(interpolation_data, interpolation_config.get("params", {}))
            print(f"{indicator_name}插值完成")
            
            # 保存中间结果
            self._save_intermediate_raster(interpolated_result, output_path)
        else:
            interpolated_result = self._load_intermediate_raster(output_path)
        
        return interpolated_result

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

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        主计算方法
        
        参数:
            params: 参数字典，包含数据、配置等信息
            
        返回:
            最终计算结果
        """
        try:
            station_indicators = params['station_indicators']
            station_coords = params['station_coords']
            algorithmConfig = params['algorithmConfig']
            config = params['config']
            self._algorithms = params['algorithms']
            print("开始计算河南冬小麦干热风区划")
            print('第一步，计算各站的干热风强度指数，并求历年平均')
            GRF_index = self.calculate_GRF_index(station_indicators,config)

            print('第二步，根据干热风强度指数R，插值栅格化指数')
            GRF_index_raster = self._perform_interpolation_for_indicator(GRF_index, station_coords, params, "GRF_risk")

            print('第三步，基于插值栅格化指数进行区划分级')
            final_result = self._perform_classification(GRF_index_raster, params)
            
            print(f'计算{params["config"].get("cropCode","")}-{params["config"].get("zoningType","")}-{params["config"].get("element","")}-区划完成')
            return final_result

        except Exception as e:
            print(f"计算过程中出错: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }    



    def _save_intermediate_raster(self, result, output_path):
        """保存中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        data = result['data']
        meta = result['meta']
        
        # 根据数据类型确定GDAL数据类型
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32
        
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
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
        band.SetNoDataValue(0)
        
        band.FlushCache()
        dataset = None
        
        print(f"中间栅格结果已保存: {output_path}")

    def _load_intermediate_raster(self, input_path):
        """加载中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        dataset = gdal.Open(str(input_path))
        if dataset is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        
        transform = dataset.GetGeoTransform()
        crs = dataset.GetProjection()
        
        meta = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        
        dataset = None
        
        return {
            'data': data,
            'meta': meta
        }















































































