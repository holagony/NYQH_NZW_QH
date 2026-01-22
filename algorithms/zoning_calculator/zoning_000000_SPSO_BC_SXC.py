import os
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def _normalize_array(array):
    if array.size == 0:
        return array
    mask = ~np.isnan(array)
    if not np.any(mask):
        return array
    valid = array[mask].astype(float)
    mn = float(np.min(valid))
    mx = float(np.max(valid))
    if mx == mn:
        out = np.full_like(array, 0.5, dtype=float)
    else:
        out = (array.astype(float) - mn) / (mx - mn)
    out[~mask] = np.nan
    return out


class SPSO_BC:
    """全国大豆食心虫病害区划计算器"""

    def _get_algorithm(self, algorithm_name):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
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
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        dataset = None

    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        interpolation = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")
        }
        result = interpolator.execute(data, interpolation_params)
        return result

    def _compute_bean_moth_F(self, daily, current_year):
        if daily is None or len(daily) == 0:
            return np.nan
        tavg = daily["tavg"] if "tavg" in daily.columns else pd.Series(index=daily.index, dtype=float)
        rhum = daily["rhum"] if "rhum" in daily.columns else pd.Series(index=daily.index, dtype=float)
        prev_year = int(current_year) - 1
        t9 = tavg[(tavg.index.month == 9) & (tavg.index.year == prev_year) & (tavg.index.day >= 21)].mean()
        t12 = tavg[(tavg.index.month == 12) & (tavg.index.year == prev_year)].mean()
        t4 = tavg[(tavg.index.month == 4) & (tavg.index.year == current_year) & (tavg.index.day <= 10)].mean()
        t56 = tavg[((tavg.index.month == 5) | (tavg.index.month == 6)) & (tavg.index.year == current_year)].mean()
        h8 = rhum[(rhum.index.month == 8) & (rhum.index.year == current_year)].mean()
        vals = [t9, t12, t4, t56, h8]
        if any(pd.isna(v) for v in vals):
            return np.nan
        f = 0.157 * float(t9) + 0.113 * float(t12) + 0.066 * float(t4) + 0.357 * float(t56) + 0.311 * float(h8)
        return float(f)

    def calculate_SXC(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')
        current_year = int(str(end_date)[:4]) if end_date else int(str(start_date)[:4])

        station_values = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            f = self._compute_bean_moth_F(daily, current_year)
            station_values[sid] = float(f) if np.isfinite(f) else np.nan

        interp = self._interpolate(station_values, station_coords, cfg, algorithm_config)
        interp['data'] = _normalize_array(interp['data'])

        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "全国大豆食心虫综合风险指数.tif")
        self._save_geotiff_gdal(interp['data'].astype(np.float32), interp['meta'], tif_path, 0)

        class_conf = algorithm_config.get('classification', {})
        data_out = interp['data']
        if not class_conf:
            class_conf = {
                'method': 'custom_thresholds',
                'thresholds': [
                    {'min': 0.0, 'max': 0.47, 'level': 4, 'label': '低'},
                    {'min': 0.47, 'max': 0.59, 'level': 3, 'label': '中'},
                    {'min': 0.59, 'max': 0.69, 'level': 2, 'label': '较高'},
                    {'min': 0.69, 'max': 1.01, 'level': 1, 'label': '极高'},
                ]
            }
        method = class_conf.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{method}")
        data_out = classifier.execute(interp['data'].astype(float), class_conf)

        final_tif = str(out_dir / "全国大豆食心虫_分级.tif")
        self._save_geotiff_gdal(np.array(data_out).astype(np.float32), interp['meta'], final_tif, 0)
        return {
            'data': np.array(data_out),
            'meta': {
                'width': interp['meta']['width'],
                'height': interp['meta']['height'],
                'transform': interp['meta']['transform'],
                'crs': interp['meta']['crs']
            }
        }

    def calculate(self, params):
        config = params['config']
        self._algorithms = params['algorithms']
        d = config.get('element')
        if d == 'SXC':
            return self.calculate_SXC(params)
        raise ValueError(f"不支持的灾害类型: {d}")
