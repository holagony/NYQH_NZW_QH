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
 
    def calculate_HQDH_index(self, station_coords: dict, config, algorithm_config):
        dm = DataManager(config.get('inputFilePath'), config.get('stationFilePath'), multiprocess=False)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        tw_cfg = algorithm_config.get('station_time_windows', [])
        tw_map = {}
        for grp in tw_cfg:
            s = grp.get('start_date')
            e = grp.get('end_date')
            sts = grp.get('stations', [])
            for sid in sts:
                tw_map[str(sid)] = (s, e)
        default_tw = algorithm_config.get('default_time_window', ["04-16", "05-05"])
        results = {}
        for sid in station_ids:
            sid_str = str(sid)
            start_str, end_str = tw_map.get(sid_str, (default_tw[0], default_tw[1]))
            light_cfg = {
                "type": "conditional_count",
                "frequency": "yearly",
                "start_date": start_str,
                "end_date": end_str,
                "conditions": [
                    {"variable": "tmin", "operator": ">", "value": -2},
                    {"variable": "tmin", "operator": "<=", "value": 0}
                ]
            }
            medium_cfg = {
                "type": "conditional_count",
                "frequency": "yearly",
                "start_date": start_str,
                "end_date": end_str,
                "conditions": [
                    {"variable": "tmin", "operator": ">", "value": -4},
                    {"variable": "tmin", "operator": "<=", "value": -2}
                ]
            }
            heavy_cfg = {
                "type": "conditional_count",
                "frequency": "yearly",
                "start_date": start_str,
                "end_date": end_str,
                "conditions": [
                    {"variable": "tmin", "operator": "<=", "value": -4}
                ]
            }
            try:
                light = dm.calculate_indicator(sid_str, light_cfg, start_date=config.get('startDate'), end_date=config.get('endDate'))
                medium = dm.calculate_indicator(sid_str, medium_cfg, start_date=config.get('startDate'), end_date=config.get('endDate'))
                heavy = dm.calculate_indicator(sid_str, heavy_cfg, start_date=config.get('startDate'), end_date=config.get('endDate'))
                light_df = pd.DataFrame.from_dict(light, orient='index')
                medium_df = pd.DataFrame.from_dict(medium, orient='index')
                heavy_df = pd.DataFrame.from_dict(heavy, orient='index')
                merged_df = pd.concat([light_df, medium_df, heavy_df], axis=1)
                merged_df.columns = ['YDDH_light', 'YDDH_medium', 'YDDH_heavy']
                idx_val = merged_df['YDDH_light'].mean() * self.WEIGHTS['YDDH_light'] + merged_df['YDDH_medium'].mean() * self.WEIGHTS['YDDH_medium'] + merged_df['YDDH_heavy'].mean() * self.WEIGHTS['YDDH_heavy']
                results[sid_str] = float(idx_val) if np.isfinite(idx_val) else np.nan
            except Exception:
                results[sid_str] = np.nan
        if len(results) > 0 and all([np.isnan(v) for v in results.values()]):
            pass
        else:
            max_value = max([v for v in results.values() if not np.isnan(v)]) if any([not np.isnan(v) for v in results.values()]) else np.nan
            min_value = min([v for v in results.values() if not np.isnan(v)]) if any([not np.isnan(v) for v in results.values()]) else np.nan
            max_keys = [k for k, v in results.items() if v == max_value]
            min_keys = [k for k, v in results.items() if v == min_value]
            print(f'苹果花期冻害区划：单站最高指数：{max_keys}：{max_value}')
            print(f'苹果花期冻害区划：单站最低指数：{min_keys}：{min_value}')
            import datetime
            from pathlib import Path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'花期冻害强度指数_{timestamp}.csv'
            result_df = pd.DataFrame(list(results.items()), columns=['站点ID', '花期冻害强度指数'])
            intermediate_dir = Path(config["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / filename
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"花期冻害强度指数文件已保存为 '{output_path}'")
        return results

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
        output_path = intermediate_dir / "intermediate_HQDH_risk.tif"
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

    def calculate_HQDH(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        hazard_config = algorithm_config.get('hazard', algorithm_config)
        cfg = params.get('config', {})
        self.WEIGHTS = algorithm_config.get('weights', self.WEIGHTS)
        print("开始计算陕西苹果花期冻害区划")
        station_values = {}
        print("第一步，根据站点指标计算花期冻害强度指数并求历年平均")
        HQDH_index = self.calculate_HQDH_index(station_coords, cfg, algorithm_config)
        station_values = {sid: float(val) if np.isfinite(val) else np.nan for sid, val in HQDH_index.items()}

        print("第二步，根据花期冻害指数插值栅格化指数")
        YDDH_index_raster = self._perform_interpolation_for_indicator(station_values, station_coords, params, "HQDH_risk")
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
        if disaster_type == 'HQDH':
            return self.calculate_HQDH(params)
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

