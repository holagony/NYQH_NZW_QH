# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 13:32:18 2025

@author: hx
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import os
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation


class SPMA_ZH:
    def _get_algorithm(self, algorithm_name: str) -> Any:
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]
    def _save_params_snapshot(self, params: Dict[str, Any]) -> str:
        try:
            import json
            cfg = params.get('config', {})
            rp = cfg.get('resultPath') or os.getcwd()
            outdir = Path(rp)
            outdir.mkdir(parents=True, exist_ok=True)
            fp = outdir / "params_SPMA_DWLH.json"
            snap = {}
            snap['station_coords'] = params.get('station_coords', {})
            snap['algorithmConfig'] = params.get('algorithmConfig', {})
            snap['config'] = cfg
            snap['startDate'] = params.get('startDate') or cfg.get('startDate')
            snap['endDate'] = params.get('endDate') or cfg.get('endDate')
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            return str(fp)
        except Exception:
            return ""
    def _load_params_snapshot(self, snapshot_path: str) -> Dict[str, Any]:
        import json
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snap = json.load(f)
        station_coords = snap.get('station_coords', {})
        algorithmConfig = snap.get('algorithmConfig', {})
        config = snap.get('config', {})
        sd = snap.get('startDate') or config.get('startDate')
        ed = snap.get('endDate') or config.get('endDate')
        if not config.get('element'):
            af = config.get('algoFrom', '')
            el = None
            if isinstance(af, str) and af:
                try:
                    el = af.split('_')[-1]
                except Exception:
                    el = None
            config['element'] = el or 'DWLH'
        return {'station_coords': station_coords, 'algorithmConfig': algorithmConfig, 'config': config}

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], indicator_name: str) -> None:
        try:
            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name
            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                return
            meta["nodata"] = -32768
            self._save_geotiff_gdal(data, meta, output_path)
        except Exception:
            pass

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        from osgeo import gdal
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

    def _calc_active_heat(self, df: pd.DataFrame, years: list) -> Dict[int, float]:
        df=pd.DataFrame(df)
        df.dropna(subset=['tavg'], inplace=True)
        res = {}
        base = 10.0
        for y in years:
            sub = df[(df.index.year == y)]# & (df.index.month.isin([5, 6, 7, 8, 9]))]
            if sub.empty:
                continue
            tavg = sub['tavg'].values.astype(float)
            #val = np.sum(np.maximum(tavg - base, 0.0))
            val = np.sum(tavg[tavg >= base])
            res[y] = float(val)
        return res

    def _ensure_tavg(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'tavg' in df.columns:
            return df


    def _risk_station(self, dm: DataManager, sid: str, start_date: int, end_date: int) -> Dict[str, Any]:
        daily = dm.load_station_data(sid, start_date, end_date)
        if daily is None or len(daily) == 0:
            return {}
        daily = self._ensure_tavg(daily)
        years = sorted(daily.index.year.unique())
        heat_by_year = self._calc_active_heat(daily, years)
        if not heat_by_year:
            return {}
        years2 = sorted(heat_by_year.keys())
        vals = np.array([heat_by_year[y] for y in years2], dtype=float)
        hbar = float(np.nanmean(vals))
        ha = {y: float(heat_by_year[y] - hbar) for y in years2}
        general_count = int(np.sum([1 for y in years2 if (ha[y] < -120.0) and (ha[y] >= -200.0)]))
        severe_count = int(np.sum([1 for y in years2 if ha[y] < -200.0]))
        total_years = len(years2)
        f_general = float(general_count) / float(total_years) if total_years > 0 else 0.0
        f_severe = float(severe_count) / float(total_years) if total_years > 0 else 0.0
        hd = (f_general * 1.0 + f_severe * 2.0) * 100
        per_year = []
        for y in years2:
            per_year.append({
                'year': y,
                'Ht': float(heat_by_year[y]),
                'Hbar': hbar,
                'Ha': float(ha[y]),
                'general': int((ha[y] < -120.0) and (ha[y] >= -200.0)),
                'severe': int(ha[y] < -200.0)
            })
        return {'risk': hd, 'detail': per_year}

    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        interpolation = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        data = {'station_values': station_values, 'station_coords': station_coords, 'dem_path': config.get("demFilePath", ""), 'shp_path': config.get("shpFilePath", ""), 'grid_path': config.get("gridFilePath", ""), 'area_code': config.get("areaCode", "")}
        result = interpolator.execute(data, interpolation_params)
        return result

    def _export_station_csv(self, station_stats: Dict[str, Any], result_path: str):
        rows = []
        for sid, val in station_stats.items():
            for d in val.get('detail', []):
                rows.append({
                    'station_id': sid,
                    'year': d['year'],
                    'Ht': d['Ht'],
                    'Hbar': d['Hbar'],
                    'Ha': d['Ha'],
                    'general': d['general'],
                    'severe': d['severe']
                })
        if rows:
            df = pd.DataFrame(rows)
            outdir = Path(result_path) / "intermediate"
            outdir.mkdir(parents=True, exist_ok=True)
            fp = outdir / "春玉米低温冷害站点统计.csv"
            try:
                df.to_csv(fp, index=False, encoding='utf-8-sig')
            except Exception:
                df.to_csv(fp, index=False)

    def _calculate_DWLH(self, params):
        self._algorithms = params['algorithms']
        station_coords = params['station_coords']
        algorithmConfig = params.get('algorithmConfig', {})
        config = params['config']
        data_dir = config.get('inputFilePath')
        station_file = config.get('stationFilePath')
        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = params.get('startDate') or config.get('startDate')
        end_date = params.get('endDate') or config.get('endDate')
        station_stats = {}
        for sid in station_ids:
            st = self._risk_station(dm, sid, start_date, end_date)
            if st:
                station_stats[sid] = st
        self._export_station_csv(station_stats, config.get("resultPath"))
        station_values_hd = {sid: float(v['risk']) for sid, v in station_stats.items() if 'risk' in v}
        heat_mean_station = {}
        for sid, v in station_stats.items():
            det = v.get('detail', [])
            if det:
                hts = [d['Ht'] for d in det if 'Ht' in d]
                if len(hts) > 0:
                    heat_mean_station[sid] = float(np.nanmean(hts))
        grid_path = config.get('gridFilePath')
        dem_path = config.get('demFilePath')
        area_code = config.get('areaCode')
        shp_path = config.get('shpFilePath')
        heat_conf = algorithmConfig.get('heat_interpolation', {'method': 'lsm_idw', 'params': {'block_size': 256, 'radius_dist': 5.0, 'min_num': 10, 'first_size': 100, 'nodata': 0, 'var_name': 'value'}})
        heat_method = heat_conf.get('method', 'lsm_idw')
        heat_params = heat_conf.get('params', {'block_size': 256, 'radius_dist': 5.0, 'min_num': 10, 'first_size': 100, 'nodata': 0, 'var_name': 'value'})
        interp_heat = self._get_algorithm(f"interpolation.{heat_method}").execute(
            {'station_values': heat_mean_station, 'station_coords': station_coords, 'grid_path': grid_path, 'dem_path': dem_path, 'area_code': area_code, 'shp_path': shp_path},
            heat_params
        )
        heat_grid = interp_heat['data']
        self._save_geotiff_gdal(heat_grid, interp_heat['meta'], os.path.join(config.get("resultPath"), "intermediate", "积温_lsminterp.tif"), 0)
        risk_conf = algorithmConfig.get('risk_interpolation', {'method': 'lsm_idw', 'params': {'block_size': 256, 'radius_dist': 5.0, 'min_num': 10, 'first_size': 100, 'nodata': 0, 'var_name': 'value'}})
        risk_method = risk_conf.get('method', 'lsm_idw')
        risk_params = risk_conf.get('params', {'block_size': 256, 'radius_dist': 5.0, 'min_num': 10, 'first_size': 100, 'nodata': 0, 'var_name': 'value'})
        interp_hd = self._get_algorithm(f"interpolation.{risk_method}").execute(
            {'station_values': station_values_hd, 'station_coords': station_coords, 'grid_path': grid_path, 'dem_path': dem_path, 'area_code': area_code, 'shp_path': shp_path},
            risk_params
        )
        hd_grid = interp_hd['data']
        self._save_geotiff_gdal(hd_grid, interp_hd['meta'], os.path.join(config.get("resultPath"), "intermediate", "累积影响指数_idwinterp.tif"), 0)
        from osgeo import gdal
        grid_ds = gdal.Open(grid_path, gdal.GA_ReadOnly)
        rows = grid_ds.RasterYSize
        cols = grid_ds.RasterXSize
        gt = grid_ds.GetGeoTransform()
        grid_arr = grid_ds.GetRasterBand(1).ReadAsArray()
        grid_nodata = 0
        y_coords = np.array([gt[3] + (r + 0.5) * gt[5] for r in range(rows)], dtype=np.float64)
        x_coords = np.array([gt[0] + (c + 0.5) * gt[1] for c in range(cols)], dtype=np.float64)
        lat_grid = np.repeat(y_coords.reshape(rows, 1), cols, axis=1).astype(np.float32)
        aligner = LSMIDWInterpolation()
        temp_aligned = os.path.join(os.path.dirname(__file__), "temp_align_dem.tif")
        aligned_dem_path = aligner._align_datasets(grid_path, dem_path, temp_aligned)
        dem_ds = gdal.Open(aligned_dem_path, gdal.GA_ReadOnly)
        alti_grid = dem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        dem_nodata = -32768
        mask = (grid_arr != grid_nodata) & (alti_grid != dem_nodata) & ~np.isnan(alti_grid)
        lat_score = np.full((rows, cols), np.nan, dtype=np.float32)
        lat_score[lat_grid >= 43.0] = 3.0
        lat_score[(lat_grid >= 42.0) & (lat_grid < 43.0)] = 2.5
        lat_score[(lat_grid >= 41.0) & (lat_grid < 42.0)] = 2.0
        lat_score[(lat_grid >= 40.0) & (lat_grid < 41.0)] = 1.5
        lat_score[(lat_grid >= 39.0) & (lat_grid < 40.0)] = 1.0
        lat_score[lat_grid < 39.0] = 0.5
        lat_score[~mask] = np.nan
        self._save_geotiff_gdal(lat_score, interp_heat['meta'], os.path.join(config.get("resultPath"), "intermediate", "纬度评分.tif"), 0)
        alti_score = np.full((rows, cols), np.nan, dtype=np.float32)
        alti_score[alti_grid >= 500.0] = 3.0
        alti_score[(alti_grid >= 400.0) & (alti_grid < 500.0)] = 2.5
        alti_score[(alti_grid >= 300.0) & (alti_grid < 400.0)] = 2.0
        alti_score[(alti_grid >= 200.0) & (alti_grid < 300.0)] = 1.5
        alti_score[(alti_grid >= 100.0) & (alti_grid < 200.0)] = 1.0
        alti_score[alti_grid < 100.0] = 0.5
        alti_score[~mask] = np.nan
        self._save_geotiff_gdal(alti_score, interp_heat['meta'], os.path.join(config.get("resultPath"), "intermediate", "海拔评分.tif"), 0)
        ht_score = np.full_like(heat_grid, np.nan, dtype=np.float32)
        ht_score[heat_grid <= 2900.0] = 5.0
        ht_score[(heat_grid > 2900.0) & (heat_grid <= 3000.0)] = 4.0
        ht_score[(heat_grid > 3000.0) & (heat_grid <= 3100.0)] = 3.0
        ht_score[(heat_grid > 3100.0) & (heat_grid <= 3200.0)] = 2.0
        ht_score[heat_grid > 3200.0] = 1.0
        ht_score[~mask] = np.nan
        self._save_geotiff_gdal(ht_score, interp_heat['meta'], os.path.join(config.get("resultPath"), "intermediate", "积温评分.tif"), 0)
        hd_class = self._get_algorithm("classification.natural_breaks").execute(hd_grid, {'num_classes': 5})
        hd_score = (5 - hd_class + 1).astype(np.float32)
        hd_score[~mask] = np.nan
        self._save_geotiff_gdal(hd_score, interp_hd['meta'], os.path.join(config.get("resultPath"), "intermediate", "累积影响评分.tif"), 0)
        wcfg = algorithmConfig.get('evaluation', {}).get('weights', {})
        w_lat = float(wcfg.get('lat', 0.2))
        w_alt = float(wcfg.get('alt', 0.2))
        w_heat = float(wcfg.get('heat', 0.3))
        w_risk = float(wcfg.get('risk', 0.3))
        hazard = w_lat * lat_score + w_alt * alti_score + w_heat * ht_score + w_risk * hd_score
        self._save_geotiff_gdal(hazard, interp_heat['meta'], os.path.join(config.get("resultPath"), "intermediate", "低温冷害气候区划综合指数.tif"), 0)
        class_conf = algorithmConfig.get('classification', {})
        classification_method = class_conf.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}") if class_conf else None
        data = hazard
        try:
            if classifier is not None:
                out = classifier.execute(hazard, class_conf)
                if isinstance(out, np.ndarray):
                    data = out
                else:
                    data = hazard
        except Exception:
            data = hazard
        meta = {'width': interp_heat['meta']['width'], 'height': interp_heat['meta']['height'], 'transform': interp_heat['meta']['transform'], 'crs': interp_heat['meta']['crs']}
        if data is None or meta is None:
            from osgeo import gdal
            gp = config.get('gridFilePath')
            ds = gdal.Open(gp, gdal.GA_ReadOnly)
            rows, cols = ds.RasterYSize, ds.RasterXSize
            data = np.zeros((rows, cols), dtype=np.int16)
            meta = {'width': cols, 'height': rows, 'transform': ds.GetGeoTransform(), 'crs': ds.GetProjection()}
            ds = None
        return {'data': data, 'meta': meta, 'type': '辽宁低温冷害'}

    def calculate(self, params):
        self._save_params_snapshot(params)
        
        return self._calculate_DWLH(params)

