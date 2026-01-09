import os
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def _normalize_array(array):
    """栅格归一化到 [0,1]，保留 NaN；常数场归一化为 0.5"""
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


class APPL_ZH:
    """苹果-果实膨大期高温干旱区划计算器
    
    职责
    - 计算站点致灾因子：高温致灾指数与干旱指数
    - 按权重合成站点风险值后插值为栅格
    - 对结果进行归一化与可选分级，输出专题产品
    """

    def _get_algorithm(self, algorithm_name):
        """从算法注册器中获取计算组件"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存 GeoTIFF 文件（单波段，LZW 压缩）"""
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
        """统一插值入口：支持 IDW/LSM-IDW 等方法"""
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

    def _calc_tmax_risk(self, daily, months):
        """计算高温致灾指数（按阈值分级频率加权）
        
        规则
        - 轻：35 ≤ Tmax < 38，权重 0.1
        - 中：38 ≤ Tmax < 40，权重 0.3
        - 重：Tmax ≥ 40，权重 0.5
        指标为各级别频率的加权和的年度平均
        """
        if daily is None or len(daily) == 0 or 'tmax' not in daily.columns:
            return np.nan
        df = daily.copy()
        df = df[(df.index.month.isin(months))]
        if df.empty:
            return np.nan
        years = sorted(df.index.year.unique())
        light_w = 0.1
        medium_w = 0.3
        heavy_w = 0.5
        vals = []
        for y in years:
            sub = df[df.index.year == y]['tmax']
            if sub.size == 0:
                continue
            days = float(sub.size)
            # 各等级频率（占比）
            fl = float(((sub >= 35) & (sub < 38)).sum()) / days
            fm = float(((sub >= 38) & (sub < 40)).sum()) / days
            fh = float(((sub >= 40)).sum()) / days
            vals.append(light_w * fl + medium_w * fm + heavy_w * fh)
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def _calc_dr(self, daily, thr, k, months):
        """计算干旱指数 Dr = 无降水日数 × k / 累计降水量（年度平均）"""
        if daily is None or len(daily) == 0:
            return np.nan
        if 'precip' in daily.columns:
            p = daily['precip']
        elif 'P' in daily.columns:
            p = daily['P']
        else:
            return np.nan
        p = p[p.index.month.isin(months)]
        if p.empty:
            return np.nan
        years = sorted(p.index.year.unique())
        vals = []
        for y in years:
            sub = p[p.index.year == y]
            if sub.size == 0:
                continue
            no_rain_days = int((sub <= thr).sum())
            rain_sum = float(np.nansum(sub.values))
            denom = rain_sum if rain_sum > 0 else 1e-6
            dr = float(no_rain_days * k / denom)
            vals.append(dr)
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def calculate_GSPDQGWGH(self, params):
        """果实膨大期高温干旱风险计算主流程
        
        步骤
        1) 加载站点逐日数据（时段默认 6–8 月）
        2) 计算站点高温致灾指数与干旱指数
        3) 对 Dr 进行站间归一化后按权重与高温指数线性合成
        4) 插值成栅格并归一化
        5) 可选分类，输出中间与最终产品
        """
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
        months = algorithm_config.get('months', [6, 7, 8])
        thr = float(algorithm_config.get('dry_threshold_mm', 0.1))
        k = float(algorithm_config.get('k', 2.83))
        tmax_weight = float(algorithm_config.get('tmax_weight', 0.4))
        dr_weight = float(algorithm_config.get('dr_weight', 0.6))
        tmax_map = {}
        dr_map = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            t_risk = self._calc_tmax_risk(daily, months)
            d_risk = self._calc_dr(daily, thr, k, months)
            tmax_map[sid] = float(t_risk) if np.isfinite(t_risk) else np.nan
            dr_map[sid] = float(d_risk) if np.isfinite(d_risk) else np.nan
        # 站间归一化
        dr_values = np.array([v for v in dr_map.values() if np.isfinite(v)], dtype=float)
        tv_values = np.array([v for v in tmax_map.values() if np.isfinite(v)], dtype=float)
        if dr_values.size > 0:
            mn = float(np.min(dr_values))
            mx = float(np.max(dr_values))
            if tv_values.size > 0:
                mn_tv = float(np.min(tv_values))
                mx_tv = float(np.max(tv_values))
            else:
                mn_tv = np.nan
                mx_tv = np.nan

            station_values = {}
            for sid in station_ids:
                tv = tmax_map.get(sid, np.nan)
                dv = dr_map.get(sid, np.nan)
                if np.isfinite(tv) and np.isfinite(mn_tv) and np.isfinite(mx_tv):
                    tvn = 0.5 if mx_tv == mn_tv else (tv - mn_tv) / (mx_tv - mn_tv)
                else:
                    tvn = np.nan
                if np.isfinite(dv):
                    if mx == mn:
                        dvn = 0.5
                    else:
                        dvn = (dv - mn) / (mx - mn)
                else:
                    dvn = np.nan
                sv = tmax_weight * tvn + dr_weight * dvn if np.isfinite(tvn) and np.isfinite(dvn) else np.nan
                station_values[sid] = float(sv) if np.isfinite(sv) else np.nan
        else:
            if tv_values.size > 0:
                mn_tv = float(np.min(tv_values))
                mx_tv = float(np.max(tv_values))
            else:
                mn_tv = np.nan
                mx_tv = np.nan

            station_values = {}
            for sid in station_ids:
                tv = tmax_map.get(sid, np.nan)
                if np.isfinite(tv) and np.isfinite(mn_tv) and np.isfinite(mx_tv):
                    tvn = 0.5 if mx_tv == mn_tv else (tv - mn_tv) / (mx_tv - mn_tv)
                else:
                    tvn = np.nan
                sv = tmax_weight * tvn if np.isfinite(tvn) else np.nan
                station_values[sid] = float(sv) if np.isfinite(sv) else np.nan

        # 插值与栅格归一化
        interp = self._interpolate(station_values, station_coords, cfg, algorithm_config)
        interp['data'] = _normalize_array(interp['data'])
        # 输出路径与中间产品
        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "果实膨大期高温干旱风险指数.tif")
        self._save_geotiff_gdal(interp['data'].astype(np.float32), interp['meta'], tif_path, 0)
        # 分类（可选）
        class_conf = algorithm_config.get('classification', {})
        data_out = interp['data']
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            try:
                classifier = self._get_algorithm(f"classification.{method}")
                data_out = classifier.execute(interp['data'].astype(float), class_conf)
            except Exception:
                data_out = interp['data']
        # 最终产品
        final_tif = str(out_dir / "果实膨大期高温干旱_分级.tif")
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
        """统一入口：根据 element 选择目标计算"""
        config = params['config']
        self._algorithms = params['algorithms']
        d = config.get('element')
        if d == 'GSPDQGWGH':
            return self.calculate_GSPDQGWGH(params)
        raise ValueError(f"不支持的灾害类型: {d}")
