import os
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager
from concurrent.futures import ProcessPoolExecutor, as_completed


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
            'area_code': config.get("areaCode", "")}
        result = interpolator.execute(data, interpolation_params)
        return result

    def _doc_predictors(self, daily, current_year):
        if daily is None or len(daily) == 0:
            return np.nan, np.nan, np.nan
        tmin = daily["tmin"] if "tmin" in daily.columns else pd.Series(index=daily.index, dtype=float)
        tavg = daily["tavg"] if "tavg" in daily.columns else pd.Series(index=daily.index, dtype=float)
        precip = daily["precip"] if "precip" in daily.columns else pd.Series(index=daily.index, dtype=float)
        x1 = tmin[(tmin.index.year == current_year) & (tmin.index.month == 1)].mean()
        def _dekad_ranges(year, month):
            import calendar
            _, ndays = calendar.monthrange(year, month)
            return [(1, 10), (11, 20), (21, ndays)]
        dekads = _dekad_ranges(current_year, 5) + _dekad_ranges(current_year, 6)
        x2 = 0.0
        valid = True
        for d0, d1 in dekads:
            mask = (tavg.index.year == current_year) & ((tavg.index.month == 5) | (tavg.index.month == 6)) & (tavg.index.day >= d0) & (tavg.index.day <= d1)
            m = tavg[mask].mean()
            if pd.isna(m):
                valid = False
                break
            x2 += float(m)
        if not valid:
            x2 = np.nan
        prev_year = int(current_year) - 1
        x3 = precip[(precip.index.year == prev_year) & (precip.index.month == 9) & (precip.index.day >= 11) & (precip.index.day <= 20)].sum()
        return x1, x2, x3

    def _classify_rate_level(self, rate):
        if pd.isna(rate):
            return None
        r = float(rate)
        if r <= 25:
            return 1
        if r <= 35:
            return 2
        if r <= 45:
            return 3
        return 4

    def _compute_hazard_index(self, daily, years):
        '''
        1.对每个年份 y：
            计算 x1_y、x2_y、x3_y（逐年定义：x1为1月tmin月平均；x2为5–6月6旬的tavg旬平均累积；x3为上年9月中旬降水量之和）
            用式(6.1)得到当年虫食率；按表6.1映射等级
        2.统计各等级频率 F_ij
        3.用表6.3权重合成 H_i
        '''
        weights = {1: 0.055, 2: 0.118, 3: 0.262, 4: 0.565}
        levels = []
        b1 = -1.1507
        b2 = 0.7855
        b3 = 0.2336
        b0 = -24.4108
        for y in years:
            x1, x2, x3 = self._doc_predictors(daily, y)
            if not any(pd.isna(v) for v in [x1, x2, x3]):
                rate = float(b0) + b1 * float(x1) + b2 * float(x2) + b3 * float(x3)
                lvl = self._classify_rate_level(rate)
                if lvl is not None:
                    levels.append(lvl)
        n = len(levels)
        if n == 0:
            return np.nan
        freq = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for lvl in levels:
            freq[lvl] += 1.0
        for j in freq:
            freq[j] = freq[j] / n
        return float(sum(freq[j] * weights[j] for j in (1, 2, 3, 4)))

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
        sy = int(str(start_date)[:4]) if start_date else None
        ey = int(str(end_date)[:4]) if end_date else None
        years = list(range(sy, ey + 1))
        load_start = f"{years[0]-1}0901"
        load_end = f"{years[-1]}0630"

        hazard_values = {}
        max_workers = 16
        chunk_size = max(1, len(station_ids) // (max_workers * 2) + 1)
        args_common = (cfg.get('inputFilePath'), cfg.get('stationFilePath'), load_start, load_end, years)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in range(0, len(station_ids), chunk_size):
                chunk = station_ids[i:i + chunk_size]
                futures.append(ex.submit(_sxc_hazard_batch_worker, (chunk, *args_common)))
            for fut in as_completed(futures):
                hazard_values.update(fut.result())

        hazard_interpolated = self._interpolate(hazard_values, station_coords, cfg, algorithm_config)

        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_hazard = str(inter_dir / "全国大豆食心虫_危险性指数.tif")
        self._save_geotiff_gdal(hazard_interpolated['data'].astype(np.float32), hazard_interpolated['meta'], tif_hazard, 0)

        class_conf = algorithm_config.get('classification', {})
        if not class_conf:
            class_conf = {'method': 'natural_breaks', 'num_classes': 4}
        method = class_conf.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{method}")
        data_out = classifier.execute(hazard_interpolated['data'].astype(float), class_conf)

        final_tif = str(out_dir / "全国大豆食心虫_危险性_分级.tif")
        self._save_geotiff_gdal(np.array(data_out).astype(np.float32), hazard_interpolated['meta'], final_tif, 0)
        return {
            'data': np.array(data_out),
            'meta': {
                'width': hazard_interpolated['meta']['width'],
                'height': hazard_interpolated['meta']['height'],
                'transform': hazard_interpolated['meta']['transform'],
                'crs': hazard_interpolated['meta']['crs']
            }
        }

    def calculate(self, params):
        config = params['config']
        self._algorithms = params['algorithms']
        d = config.get('element')
        if d == 'SXC':
            return self.calculate_SXC(params)
        raise ValueError(f"不支持的灾害类型: {d}")


def _sxc_hazard_batch_worker(args):
    chunk, input_path, station_path, load_start, load_end, years = args
    dm = DataManager(input_path, station_path, multiprocess=False, num_processes=1)
    calc = SPSO_BC()
    vals = {}
    for sid in chunk:
        daily = dm.load_station_data(sid, load_start, load_end)
        h = calc._compute_hazard_index(daily, years)
        vals[str(sid)] = float(h) if np.isfinite(h) else np.nan
    return vals
