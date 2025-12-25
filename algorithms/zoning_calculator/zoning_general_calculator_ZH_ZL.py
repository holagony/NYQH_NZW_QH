import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
from algorithms.data_manager import DataManager

class ZH_ZL:
    def __init__(self):
        pass

    def _get_algorithm(self, name: str):
        if name not in self._algorithms:
            raise ValueError(f"不支持的算法: {name}")
        return self._algorithms[name]

    def _save_geotiff_gdal(self, data, meta, output_path: str, nodata=0):
        from osgeo import gdal
        driver = gdal.GetDriverByName('GTiff')
        y, x = data.shape
        ds = driver.Create(output_path, x, y, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(meta['transform'])
        ds.SetProjection(meta['crs'])
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        ds = None

    def _interpolate(self, station_values: Dict[str, float], station_coords: Dict[str, Any], cfg: Dict[str, Any], algo_cfg: Dict[str, Any], var_name: str):
        interp = algo_cfg.get('interpolation', {})
        method = str(interp.get('method', 'idw')).lower()
        params = interp.get('params', {})
        if 'var_name' not in params:
            params['var_name'] = 'value'
        interpolator = self._get_algorithm(f"interpolation.{method}")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }
        result = interpolator.execute(data, params)
        intermediate_dir = Path(cfg["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        g_tif_path = str(intermediate_dir / f"{var_name}.tif")
        meta = {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}
        self._save_geotiff_gdal(result['data'], meta, g_tif_path, 0)
        return result

    def calculate(self, params):
        cfg = params['config']
        algo_cfg = params.get('algorithmConfig', {})
        self._algorithms = params['algorithms']
        station_indicators = params.get('station_indicators', {})
        station_coords = params.get('station_coords', {})
        ths = algo_cfg.get('threshold')
        weights_map = {}
        bounds_map = {}
        for t in ths:
            lab = t.get('label')
            if lab:
                weights_map[lab] = float(t.get('weight', 0))
                m = t.get('min')
                M = t.get('max')
                bounds_map[lab] = (float(m) if m != "" and m is not None else None,
                                   float(M) if M != "" and M is not None else None)
        station_values: Dict[str, float] = {}
        if station_indicators:
            labs = ['暴雨日数', '大暴雨日数', '特大暴雨日数']
            default_w = {'暴雨日数': 0.2, '大暴雨日数': 0.30, '特大暴雨日数': 0.5}
            w_obj = algo_cfg.get('weights', {})
            key_map = {'暴雨日数': 'D50', '大暴雨日数': 'D100', '特大暴雨日数': 'D250'}
            w_conf = {}
            for lab in labs:
                if weights_map:
                    w_conf[lab] = weights_map.get(lab, default_w[lab])
                else:
                    w_conf[lab] = w_obj.get(key_map[lab], default_w[lab])
            for sid, inds in station_indicators.items():
                val = 0.0
                for lab in labs:
                    k = lab if lab in inds else key_map[lab]
                    v = inds.get(k, np.nan)
                    wv = w_conf[lab]
                    if np.isfinite(v):
                        val += wv * v
                station_values[sid] = float(val) if np.isfinite(val) else np.nan
        else:
            dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
            station_ids = list(station_coords.keys())
            if not station_ids:
                station_ids = dm.get_all_stations()
            available = set(dm.get_all_stations())
            station_ids = [sid for sid in station_ids if sid in available]
            sdate = algo_cfg.get('start_date') if 'start_date' in algo_cfg else cfg.get('startDate')
            edate = algo_cfg.get('end_date') if 'end_date' in algo_cfg else cfg.get('endDate')
            win_default = {
                '暴雨日数': ('05-01', '08-31'),
                '大暴雨日数': ('05-01', '08-31'),
                '特大暴雨日数': ('05-01', '08-31')
            }
            win_cfg = algo_cfg.get('windows', {})
            for sid in station_ids:
                daily = dm.load_station_data(sid, sdate, edate)
                if daily is None or len(daily) == 0:
                    station_values[sid] = np.nan
                    continue
                if 'date' in daily.columns:
                    dt = pd.to_datetime(daily['date'])
                else:
                    dt = pd.to_datetime(daily.iloc[:, 0], errors='coerce')
                p = daily['precip'] if 'precip' in daily.columns else pd.Series(np.nan, index=daily.index)
                years = sorted(list(set([d.year for d in dt if pd.notnull(d)])))
                c_map = {}
                for lab, b in bounds_map.items():
                    mn, mx = b
                    wmd = win_cfg.get(lab, win_default.get(lab, ('05-01', '08-31')))
                    vlist = []
                    for y in years:
                        st = pd.to_datetime(f"{y}-{wmd[0]}")
                        en = pd.to_datetime(f"{y}-{wmd[1]}")
                        mask = (dt >= st) & (dt <= en)
                        if mn is not None and mx is not None:
                            cond = (p >= mn) & (p <= mx)
                        elif mn is not None:
                            cond = (p >= mn)
                        elif mx is not None:
                            cond = (p <= mx)
                        else:
                            cond = pd.Series(False, index=daily.index)
                        cnt = int(((mask & cond)).sum())
                        vlist.append(cnt)
                    if vlist:
                        c_map[lab] = float(np.mean(vlist))
                    else:
                        c_map[lab] = np.nan
                val = 0.0
                for lab, wv in weights_map.items():
                    vv = c_map.get(lab, np.nan)
                    if np.isfinite(vv):
                        val += wv * vv
                station_values[sid] = float(val) if np.isfinite(val) else np.nan
        result = self._interpolate(station_values, station_coords, cfg, algo_cfg, '渍涝危险性指数')
        class_conf = algo_cfg.get('classification', {})
        if class_conf:
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in self._algorithms:
                result['data'] = self._algorithms[key].execute(result['data'], class_conf)
        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}}
