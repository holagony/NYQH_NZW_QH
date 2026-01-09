import numpy as np
import pandas as pd
from typing import Dict, Any
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from pathlib import Path
import importlib
import ast
import datetime
import os

def pre_category(x):
    """降雨量赋值标准"""
    if x < 80:
        return 0
    elif 80 <= x <= 110:
        return 1
    elif 110 < x <= 130:
        return 2
    elif x > 130:
        return 3
    else:
        return np.nan


def pre_days_category(x):
    """降雨量赋值标准"""
    if x < 8:
        return 0
    elif 8 <= x <= 10:
        return 1
    elif 10 < x <= 13:
        return 2
    elif x > 13:
        return 3
    else:
        return np.nan


def ssh_category(x):
    """日照时赋值标准"""
    if x > 120:
        return 0
    elif 110 <= x <= 120:
        return 1
    elif 95 <= x < 110:
        return 2
    elif x < 95:
        return 3
    else:
        return np.nan


def index(x):
    "年度指数划分"
    if x <= 2:
        return 0
    elif 3 <= x <= 4:
        return 1
    elif 5 <= x <= 6:
        return 2
    elif x > 6:
        return 3
    else:
        return np.nan

class WIWH_ZH:


    def _calculate_continuous_rain_indicators_station(self, station_indicators, params):
        """在站点级别计算连阴雨指标"""
        continuous_rain_indicators = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            Pre = indicators.get('Pre', np.nan)  # 总降水量
            SSH = indicators.get('SSH', np.nan)  # 总日照时数
            Pre_days = indicators.get('Pre_days', np.nan)  # 降水日数

            # str转字典
            Pre_df = pd.DataFrame.from_dict(Pre, orient='index')
            SSH_df = pd.DataFrame.from_dict(SSH, orient='index')
            Pre_days_df = pd.DataFrame.from_dict(Pre_days, orient='index')

            merged_df = pd.concat([Pre_df, SSH_df, Pre_days_df], axis=1)
            merged_df.columns = ['Pre', 'SSH', 'Pre_days']

            # 按标准赋值
            merged_df['Pre'] = merged_df['Pre'].apply(pre_category)
            merged_df['SSH'] = merged_df['SSH'].apply(ssh_category)
            merged_df['Pre_days'] = merged_df['Pre_days'].apply(pre_days_category)
            cleaned_df = merged_df.dropna()

            # 年度指数
            cleaned_df['年度指数'] = cleaned_df['Pre'] + cleaned_df['SSH'] + cleaned_df['Pre_days']

            # 连阴雨程度与
            cleaned_df['连阴雨程度'] = cleaned_df['年度指数'].apply(index)

            # 综合指数
            frequency = cleaned_df['连阴雨程度'].value_counts().sort_index()
            for level in [0, 1, 2, 3]:
                if level not in frequency:
                    frequency[level] = 0
            weighted_frequency = (0.5 * frequency.get(3, 0) + 0.3 * frequency.get(2, 0) + 0.2 * frequency.get(1, 0))
            continuous_rain_indicators[station_id] = weighted_frequency / len(cleaned_df)

        max_value = max(continuous_rain_indicators.values())
        max_keys = [key for key, value in continuous_rain_indicators.items() if value == max_value]
        min_value = min(continuous_rain_indicators.values())
        min_keys = [key for key, value in continuous_rain_indicators.items() if value == min_value]
        print(f'麦收区连阴雨气候区划:有效站点数据：{len(cleaned_df)}')
        print(f'麦收区连阴雨气候区划：单站最高综合指数：{max_keys}：{max_value}')
        print(f'麦收区连阴雨气候区划:单站最低综合指数：{min_keys}：{min_value}')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'连阴雨指标_{timestamp}.csv'
        result_df = pd.DataFrame(list(continuous_rain_indicators.items()), columns=['站点ID', '连阴雨综合指数'])
        intermediate_dir = Path(params["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"连阴雨指标综合指数文件已保存为 '{output_path}'")

        return continuous_rain_indicators

    def _interpolate_continuous_rain_risk(self, continuous_rain_risk_station, station_coords, config, crop_config):
        """对连阴雨综合风险指数进行插值"""
        interpolation = crop_config.get("interpolation")
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")

        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        print(f"使用 {interpolation_method} 方法对综合风险指数进行插值")

        # 准备插值数据
        interpolation_data = {'station_values': continuous_rain_risk_station, 'station_coords': station_coords, 'dem_path': config.get("demFilePath", ""), 'shp_path': config.get("shpFilePath", ""), 'grid_path': config.get("gridFilePath", ""), 'area_code': config.get("areaCode", "")}

        # 执行插值
        try:
            interpolated_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合风险指数插值完成")
            # 保存中间结果
            self._save_intermediate_result(interpolated_result, config, "continuous_rain_risk_interpolated")

            return interpolated_result

        except Exception as e:
            print(f"综合风险指数插值失败: {str(e)}")
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
                print(f"警告: 中间结果 {indicator_name} 格式不支持，跳过保存")
                return
            meta["nodata"] = -32768
            # 保存为GeoTIFF
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")
            # 不抛出异常，继续处理其他指标

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

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]


    def calculate_wet(self, params):
        """计算小麦连阴雨风险 - 先计算站点综合风险指数再插值"""
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']
        print(station_indicators)
        print(station_coords)
       
        print("开始计算小麦连阴雨风险 - 新流程：先计算站点综合风险指数")

        try:
            # 第一步：在站点级别计算连阴雨指标
            print("第一步：在站点级别计算连阴雨指标")
            continuous_rain_indicators = self._calculate_continuous_rain_indicators_station(station_indicators, config)

            # 第二步：对综合风险指数F进行插值
            print("第二步：对综合风险指数F进行插值")
            interpolated_risk = self._interpolate_continuous_rain_risk(continuous_rain_indicators, station_coords, config, algorithmConfig)

            # 第三步：对插值结果进行分类
            print("第四步：对插值结果进行分类")
            classification = algorithmConfig['classification']
            classification_method = classification.get('method', 'custom_thresholds')
            classifier = self._get_algorithm(f"classification.{classification_method}")

            classified_data = classifier.execute(interpolated_risk['data'], classification)
            # 准备最终结果
            result = {'data': classified_data, 'meta': interpolated_risk['meta'], 'type': 'continuous_rain_risk', 'process': 'station_level_calculation'}
            print("小麦连阴雨风险计算完成")

        except Exception as e:
            print(f"小麦连阴雨风险计算失败: {str(e)}")
            result = np.nan
        return result

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']

        return self.calculate_wet(params)
