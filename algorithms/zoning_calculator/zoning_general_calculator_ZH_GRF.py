import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import os
import datetime


class ZH_GRF:
    def _get_algorithm(self, algorithm_name: str):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name, min_value=np.nan, max_value=np.nan):
        config = params['config']
        algorithm_config = params['algorithmConfig']
        interpolation_config = algorithm_config.get("interpolation", {})
        method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{method}")

        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value': min_value,
            'max_value': max_value,
        }

        file_name = f"intermediate_{indicator_name}.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name

        if not os.path.exists(output_path):
            result = interpolator.execute(data, interpolation_config.get("params", {}))
            self._save_intermediate_raster(result, output_path)
        else:
            result = self._load_intermediate_raster(output_path)

        return result

    def _perform_classification(self, data_interpolated, params):
        algorithm_config = params['algorithmConfig']
        classification = algorithm_config.get('classification', {})
        method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{method}")
        data = classifier.execute(data_interpolated['data'], classification)
        data_interpolated['data'] = data
        return data_interpolated

    def _save_intermediate_raster(self, result, output_path: Path):
        from osgeo import gdal
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
        ds = driver.Create(str(output_path), meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        ds.SetGeoTransform(meta['transform'])
        ds.SetProjection(meta['crs'])
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0)
        band.FlushCache()
        ds = None

    def _load_intermediate_raster(self, input_path: Path):
        from osgeo import gdal
        ds = gdal.Open(str(input_path))
        if ds is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        transform = ds.GetGeoTransform()
        crs = ds.GetProjection()
        meta = {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        ds = None
        return {'data': data, 'meta': meta}

    def _calculate_grf_index(self, station_indicators: pd.DataFrame, algorithm_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[int, float]:
        weights_default = {
            'GRF_light': 0.2,
            'GRF_moderate': 0.3,
            'GRF_severe': 0.5
        }
        marks_default = {
            'GRF_light': 1,
            'GRF_moderate': 2,
            'GRF_severe': 3
        }

        weights = algorithm_config.get('weights', weights_default)
        marks = algorithm_config.get('marks', marks_default)

        indicators_keys = list(algorithm_config.get('indicators', {}).keys())
        result = {}
        for station_id, indicators in station_indicators.items():
            dfs = []
            cols = []
            for key in indicators_keys:
                val = indicators.get(key, np.nan)
                df = pd.DataFrame.from_dict(val, orient='index')
                dfs.append(df)
                cols.append(key)
            merged = pd.concat(dfs, axis=1)
            merged.columns = cols
            total = 0.0
            for key in cols:
                w = float(weights.get(key, 0.0))
                m = float(marks.get(key, 0.0))
                ni = float(merged[key].mean()) if key in merged.columns else 0.0
                total += w * m * ni
            result[station_id] = total

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"干热风强度指数_{timestamp}.csv"
        df_out = pd.DataFrame(list(result.items()), columns=['站点ID', '干热风强度指数'])
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        out_path = intermediate_dir / filename
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        return result

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithm_config = params['algorithmConfig']
        config = params['config']
        self._algorithms = params['algorithms']

        indices = self._calculate_grf_index(station_indicators, algorithm_config, config)
        raster = self._perform_interpolation_for_indicator(indices, station_coords, params, "GRF_risk")
        final_result = self._perform_classification(raster, params)
        return final_result


class WIWH_ZH(ZH_GRF):
    pass
