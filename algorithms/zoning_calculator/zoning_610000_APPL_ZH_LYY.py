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
            'RAIN_DAYS': 0.7,
            'PRECIP_SUM': 0.3
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

    def calculate_LYY_index(self, station_indicators: pd.DataFrame, config, algorithm_config):
        weights = algorithm_config.get('weights', {})
        w_days = float(weights.get('RAIN_DAYS', self.WEIGHTS['RAIN_DAYS']))
        w_precip = float(weights.get('PRECIP_SUM', self.WEIGHTS['PRECIP_SUM']))
        results = {}
        for station_id, indicators in station_indicators.items():
            days = indicators.get('RAIN_DAYS', np.nan)
            psum = indicators.get('PRECIP_SUM', np.nan)
            try:
                if isinstance(days, dict):
                    days_df = pd.DataFrame.from_dict(days, orient='index')
                else:
                    days_df = pd.DataFrame({'RAIN_DAYS': [days]})
                if isinstance(psum, dict):
                    psum_df = pd.DataFrame.from_dict(psum, orient='index')
                else:
                    psum_df = pd.DataFrame({'PRECIP_SUM': [psum]})
                days_df.columns = ['RAIN_DAYS']
                psum_df.columns = ['PRECIP_SUM']
                merged_df = pd.concat([days_df, psum_df], axis=1)
                vals_days = merged_df['RAIN_DAYS'].values.astype(float)
                vals_psum = merged_df['PRECIP_SUM'].values.astype(float)
                valid_days = vals_days[~np.isnan(vals_days)]
                valid_psum = vals_psum[~np.isnan(vals_psum)]
                if len(valid_days) > 0:
                    mn_d = np.min(valid_days)
                    mx_d = np.max(valid_days)
                    if mx_d == mn_d:
                        std_days = np.full_like(vals_days, 0.5, dtype=float)
                    else:
                        std_days = (vals_days - mn_d) / (mx_d - mn_d)
                else:
                    std_days = np.array([np.nan])
                if len(valid_psum) > 0:
                    mn_p = np.min(valid_psum)
                    mx_p = np.max(valid_psum)
                    if mx_p == mn_p:
                        std_psum = np.full_like(vals_psum, 0.5, dtype=float)
                    else:
                        std_psum = (vals_psum - mn_p) / (mx_p - mn_p)
                else:
                    std_psum = np.array([np.nan])
                val = np.nanmean(std_days) * w_days + np.nanmean(std_psum) * w_precip
                results[str(station_id)] = float(val) if np.isfinite(val) else np.nan
            except Exception:
                results[str(station_id)] = np.nan
        if len(results) > 0 and all([np.isnan(v) for v in results.values()]):
            pass
        else:
            max_value = max([v for v in results.values() if not np.isnan(v)]) if any([not np.isnan(v) for v in results.values()]) else np.nan
            min_value = min([v for v in results.values() if not np.isnan(v)]) if any([not np.isnan(v) for v in results.values()]) else np.nan
            max_keys = [k for k, v in results.items() if v == max_value]
            min_keys = [k for k, v in results.items() if v == min_value]
            print(f'苹果连阴雨风险：单站最高指数：{max_keys}：{max_value}')
            print(f'苹果连阴雨风险：单站最低指数：{min_keys}：{min_value}')
            import datetime
            from pathlib import Path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'连阴雨风险指数_{timestamp}.csv'
            result_df = pd.DataFrame(list(results.items()), columns=['站点ID', '连阴雨风险指数'])
            intermediate_dir = Path(config["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / filename
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"连阴雨风险指数文件已保存为 '{output_path}'")
        return results

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name, min_value=np.nan, max_value=np.nan):
        print(f"执行{indicator_name}插值计算...")
        config = params['config']
        algorithmConfig = params['algorithmConfig']
        interpolation_config = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation_config.get("method", "idw")
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
        output_path = intermediate_dir / "intermediate_LYY_risk.tif"
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
        classification = algorithmConfig.get('classification', {'method': 'natural_breaks', 'num_classes': 3})
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(data_interpolated['data'], classification)
        data_interpolated['data'] = data
        return data_interpolated

    def calculate_LYY(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        self.WEIGHTS = algorithm_config.get('weights', self.WEIGHTS)
        print("开始计算陕西苹果着色—成熟期连阴雨气候风险")
        station_values = {}
        print("第一步，计算连阴雨风险指数并求历年平均")
        provided_inds = params.get('station_indicators', {})
        LYY_index = self.calculate_LYY_index(provided_inds, cfg, algorithm_config)
        station_values = {sid: float(val) if np.isfinite(val) else np.nan for sid, val in LYY_index.items()}
        print("第二步，根据风险指数插值栅格化")
        LYY_raster = self._perform_interpolation_for_indicator(station_values, station_coords, params, "LYY_risk")
        print("第三步，基于栅格指数进行自然断点分级")
        final_result = self._perform_classification(LYY_raster, params)
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
        if disaster_type == 'LYY':
            return self.calculate_LYY(params)
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
