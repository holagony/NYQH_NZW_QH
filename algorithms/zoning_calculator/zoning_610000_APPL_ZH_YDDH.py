# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from osgeo import gdal


class APPL_ZH:
    def __init__(self):
        self.WEIGHTS = {
            'YDDH_light': 0.3,
            'YDDH_medium': 0.5,
            'YDDH_heavy': 0.7
        }

    def _get_algorithm(self, algorithm_name):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff(self, data, meta, output_path, nodata=0):
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
            datatype = gdal.GDT_Float32
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        ds.SetGeoTransform(meta['transform'])
        ds.SetProjection(meta['crs'])
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        ds = None
    
    def calculate_YDDH_index(self, station_indicators: pd.DataFrame, config):
        YDDH_indicators = {}
        for station_id, indicators in station_indicators.items():
            light = indicators.get('YDDH_light', np.nan)
            medium = indicators.get('YDDH_medium', np.nan)
            heavy = indicators.get('YDDH_heavy', np.nan)
            light_df = pd.DataFrame.from_dict(light, orient='index')
            medium_df = pd.DataFrame.from_dict(medium, orient='index')
            heavy_df = pd.DataFrame.from_dict(heavy, orient='index')
            merged_df = pd.concat([light_df, medium_df, heavy_df], axis=1)
            merged_df.columns = ['YDDH_light', 'YDDH_medium', 'YDDH_heavy']
            YDDH_indicators[station_id] = merged_df['YDDH_light'].mean() * self.WEIGHTS['YDDH_light'] + \
                                          merged_df['YDDH_medium'].mean() * self.WEIGHTS['YDDH_medium'] + \
                                          merged_df['YDDH_heavy'].mean() * self.WEIGHTS['YDDH_heavy']
        max_value = max(YDDH_indicators.values())
        max_keys = [key for key, value in YDDH_indicators.items() if value == max_value]
        min_value = min(YDDH_indicators.values())
        min_keys = [key for key, value in YDDH_indicators.items() if value == min_value]
        print(f'苹果越冬冻害区划：单站最高指数：{max_keys}：{max_value}')
        print(f'苹果越冬冻害区划：单站最低指数：{min_keys}：{min_value}')
        import datetime
        from pathlib import Path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'越冬冻害强度指数_{timestamp}.csv'
        result_df = pd.DataFrame(list(YDDH_indicators.items()), columns=['站点ID', '越冬冻害强度指数'])
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"越冬冻害强度指数文件已保存为 '{output_path}'")
        return YDDH_indicators

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name, min_value=np.nan, max_value=np.nan):
        print(f"执行{indicator_name}插值计算...")
        config = params['config']
        algorithmConfig = params['algorithmConfig']
        interpolation_config = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        interpolation_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value': min_value,
            'max_value': max_value
        }
        from pathlib import Path
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / "intermediate_YDDH_risk.tif"
        if not os.path.exists(output_path):
            interpolated_result = interpolator.execute(interpolation_data, interpolation_config.get("params", {}))
            print(f"{indicator_name}插值完成")
            self._save_intermediate_raster(interpolated_result, output_path)
        else:
            interpolated_result = self._load_intermediate_raster(output_path)
        return interpolated_result

    def _perform_classification(self, data_interpolated, params):
        print("执行区域分级计算...")
        algorithmConfig = params['algorithmConfig']
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(data_interpolated['data'], classification)
        data_interpolated['data'] = data
        return data_interpolated

    def calculate_YDDH(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        hazard_config = algorithm_config.get('hazard', algorithm_config)
        cfg = params.get('config', {})
        print("开始计算陕西苹果越冬冻害区划")
        station_values = {}
        provided_inds = params.get('station_indicators', {})
        print("第一步，根据站点指标计算越冬冻害强度指数并求历年平均")
        YDDH_index = self.calculate_YDDH_index(provided_inds, cfg)
        station_values = {sid: float(val) if np.isfinite(val) else np.nan for sid, val in YDDH_index.items()}

        print("第二步，根据越冬冻害指数插值栅格化指数")
        YDDH_index_raster = self._perform_interpolation_for_indicator(station_values, station_coords, params, "YDDH_risk")
        print("第三步，基于插值栅格化指数进行区划分级")
        final_result = self._perform_classification(YDDH_index_raster, params)
        return {
            'data': final_result['data'],
            'meta': {
                'width': final_result['meta']['width'],
                'height': final_result['meta']['height'],
                'transform': final_result['meta']['transform'],
                'crs': final_result['meta']['crs']
            }
        }

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'YDDH':
            return self.calculate_YDDH(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")

    def _save_intermediate_raster(self, result, output_path):
        data = result['data']
        meta = result['meta']
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

