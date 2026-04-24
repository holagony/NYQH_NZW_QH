import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi
from osgeo import gdal
from algorithms.data_manager import DataManager
from concurrent.futures import ProcessPoolExecutor, as_completed


def _build_kc_series_from_months_map(months_map, start_year, end_year):
    start_dt = pd.Timestamp(int(start_year), 1, 1)
    end_dt = pd.Timestamp(int(end_year), 12, 31)
    idx = pd.date_range(start=start_dt, end=end_dt, freq='D')
    s = pd.Series(0.0, index=idx)
    if months_map:
        for m, val in months_map.items():
            s[idx.month == int(m)] = float(val)
    return s

def _build_fixed_kc_series(start_year, end_year):
    start_dt = pd.Timestamp(int(start_year), 1, 1)
    end_dt = pd.Timestamp(int(end_year), 12, 31)
    idx = pd.date_range(start=start_dt, end=end_dt, freq='D')
    s = pd.Series(0.0, index=idx)
    years = idx.year.unique()
    for y in years:
        s[(idx >= pd.Timestamp(y, 5, 12)) & (idx <= pd.Timestamp(y, 5, 25))] = 0.32
        s[(idx >= pd.Timestamp(y, 5, 26)) & (idx <= pd.Timestamp(y, 6, 23))] = 0.51
        s[(idx >= pd.Timestamp(y, 6, 24)) & (idx <= pd.Timestamp(y, 7, 4))] = 0.69
        s[(idx >= pd.Timestamp(y, 7, 5)) & (idx <= pd.Timestamp(y, 8, 25))] = 1.15
        s[(idx >= pd.Timestamp(y, 8, 26)) & (idx <= pd.Timestamp(y, 9, 18))] = 0.5
    return s

 
def _compute_g_batch_worker(args):
    sids, input_path, station_file_path, start_date, end_date, prov_kc_table, cwdi_conf = args
    dm = DataManager(input_path, station_file_path, multiprocess=False, num_processes=1)
    out_vals = {}
    out_coords = {}
    sy = int(str(start_date)[:4])
    ey = int(str(end_date)[:4])
    calc = SPSO_ZH()
    for sid in sids:
        daily = dm.load_station_data(sid, start_date, end_date)
        info = dm.get_station_info(sid)
        st_prov = str(info.get('province', '')).strip()
        if st_prov in prov_kc_table:
            months_map = prov_kc_table[st_prov]
            kc_series = _build_kc_series_from_months_map(months_map, sy, ey)
            daily['kc'] = kc_series.reindex(daily.index).fillna(0.0)
        else:
            kc_series = _build_fixed_kc_series(sy, ey)
            daily['kc'] = kc_series.reindex(daily.index).fillna(0.0)
        g = calc._calc_drought_station_g(daily, cwdi_conf)
        out_vals[sid] = float(g) if np.isfinite(g) else np.nan
        out_coords[sid] = {
            'lat': float(info.get('lat', np.nan)),
            'lon': float(info.get('lon', np.nan)),
            'altitude': float(info.get('altitude', np.nan))
        }
    return out_vals, out_coords


def _get_prov_kc_table():
    t = {
        "山西省": {1:0.14,2:0.24,3:0.58,4:1.04,5:1.24,6:0.84,7:0.0,8:0.0,9:0.0,10:0.54,11:0.76,12:0.4},
        "甘肃省": {1:0.14,2:0.24,3:0.58,4:1.04,5:1.24,6:0.84,7:0.0,8:0.0,9:0.0,10:0.54,11:0.76,12:0.4},
        "宁夏回族自治区": {1:0.14,2:0.24,3:0.58,4:1.04,5:1.24,6:0.84,7:0.0,8:0.0,9:0.0,10:0.54,11:0.76,12:0.4},
        "河北省": {1:0.33,2:0.24,3:0.42,4:1.14,5:1.42,6:0.73,7:0.0,8:0.0,9:0.0,10:0.85,11:0.92,12:0.54},
        "北京市": {1:0.33,2:0.24,3:0.42,4:1.14,5:1.42,6:0.73,7:0.0,8:0.0,9:0.0,10:0.85,11:0.92,12:0.54},
        "天津市": {1:0.33,2:0.24,3:0.42,4:1.14,5:1.42,6:0.73,7:0.0,8:0.0,9:0.0,10:0.85,11:0.92,12:0.54},
        "陕西省": {1:0.33,2:0.24,3:0.42,4:1.14,5:1.42,6:0.73,7:0.0,8:0.0,9:0.0,10:0.85,11:0.92,12:0.54},
        "河南省": {1:0.31,2:0.5,3:0.91,4:1.4,5:1.29,6:0.6,7:0.0,8:0.0,9:0.0,10:0.63,11:0.83,12:0.93},
        "山东省": {1:0.64,2:0.41,3:0.9,4:1.22,5:1.13,6:0.83,7:0.0,8:0.0,9:0.0,10:0.67,11:0.7,12:0.74},
        "安徽省": {1:1.13,2:1.14,3:1.07,4:1.16,5:0.87,6:0.4,7:0.0,8:0.0,9:0.0,10:1.18,11:1.15,12:1.25},
        "江苏省": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "湖北省": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "新疆维吾尔自治区": {1:0.14,2:0.24,3:0.58,4:1.04,5:1.24,6:0.84,7:0.0,8:0.0,9:0.0,10:0.54,11:0.76,12:0.4},
        "西藏自治区": {1:0.14,2:0.24,3:0.58,4:1.04,5:1.24,6:1.14,7:0.0,8:0.0,9:0.0,10:0.54,11:0.76,12:0.4},
        "四川省": {1:1.14,2:1.14,3:1.42,4:1.42,5:0.83,6:0.4,7:0.0,8:0.0,9:0.0,10:0.4,11:0.93,12:1.14},
        "云南省": {1:1.14,2:1.14,3:1.42,4:1.42,5:0.83,6:0.4,7:0.0,8:0.0,9:0.0,10:0.4,11:0.93,12:1.14},
        "贵州省": {1:1.14,2:1.14,3:1.42,4:1.42,5:0.83,6:0.4,7:0.0,8:0.0,9:0.0,10:0.4,11:0.93,12:1.14},
        "重庆市": {1:1.14,2:1.14,3:1.42,4:1.42,5:0.83,6:0.4,7:0.0,8:0.0,9:0.0,10:0.4,11:0.93,12:1.14},
        "湖南省": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "上海市": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "浙江省": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "江西省": {1:0.82,2:0.91,3:0.86,4:1.77,5:1.43,6:0.41,7:0.0,8:0.0,9:0.0,10:1.14,11:1.14,12:1.19},
        "广西壮族自治区": {1:1.14,2:1.14,3:1.42,4:1.42,5:0.83,6:0.4,7:0.0,8:0.0,9:0.0,10:0.4,11:0.93,12:1.14}
    }
    return t


class SPSO_ZH:
    """全国冬小麦干旱区划计算器"""

    def _get_algorithm(self, algorithm_name):
        """从算法注册器中获取计算组件"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存 GeoTIFF 文件（单波段，LZW 压缩）"""
        # 按 numpy dtype 映射到 GDAL 数据类型
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
        # 创建并写入单波段 GeoTIFF，使用 LZW 压缩
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
        interpolation_params['min_value'] = 0
        interpolation_params['radius_dist'] = 10
        interpolation_params['min_num'] = 5
        interpolation_params['first_size'] = 100

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        # 组织插值所需输入，包含站点值/坐标、DEM/行政区/规则格网等路径
        grid_path = config.get("gridFilePath", "")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': grid_path,
            'area_code': config.get("areaCode", "")
        }
        result = interpolator.execute(data, interpolation_params)
        return result

    def _sat_vapor_pressure(self, T):
        return 0.6108 * np.exp(17.27 * T / (T + 237.3))

    def _slope_delta(self, T):
        es = self._sat_vapor_pressure(T)
        return 4098.0 * es / ((T + 237.3) ** 2)

    def _pressure_from_elev(self, z):
        return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26

    def _psy_const(self, P):
        return 0.000665 * P

    def _solar_geom(self, lat_rad, J):
        # 太阳-地球几何关系：地-日距离比 dr，太阳赤纬 delta，日落时角 omega_s
        dr = 1.0 + 0.033 * np.cos(2.0 * pi / 365.0 * J)
        delta = 0.409 * np.sin(2.0 * pi / 365.0 * J - 1.39)
        omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        # 日外辐射 Ra 与日长 N
        Ra = (24.0 * 60.0 / pi) * 0.0820 * dr * (omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s))
        N = 24.0 / pi * omega_s
        return Ra, N, omega_s

    def penman_et0(self, daily, lat_deg, elev_m, albedo=0.23, as_coeff=0.25, bs_coeff=0.5, k_rs=0.16):
        df = daily.copy()
        tmax = df['tmax']
        tmin = df['tmin']
        tmean = df['tavg'] if 'tavg' in df.columns else (tmax + tmin) / 2.0
        phi = np.deg2rad(lat_deg)
        J = df.index.dayofyear
        Ra, N, _ = self._solar_geom(phi, J)
        # 短波辐射 Rs：优先日照时数法，否则用温差经验法
        if 'sunshine' in df.columns:
            n = df['sunshine']
            Rs = (as_coeff + bs_coeff * (n / N)) * Ra
        else:
            Rs = k_rs * np.sqrt(np.maximum(tmax - tmin, 0.0)) * Ra
        # 晴空辐射 Rso 与净短波辐射 Rns
        Rso = (0.75 + 2e-5 * elev_m) * Ra
        Rns = (1.0 - albedo) * Rs
        # 饱和/实际水汽压与净长波辐射 Rnl
        es_tmax = self._sat_vapor_pressure(tmax)
        es_tmin = self._sat_vapor_pressure(tmin)
        es = (es_tmax + es_tmin) / 2.0
        ea = es * (df['rhum'] / 100.0) if 'rhum' in df.columns else es * 0.7
        sigma = 4.903e-9
        tmaxK = tmax + 273.16
        tminK = tmin + 273.16
        Rnl = sigma * ((tmaxK ** 4 + tminK ** 4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
        # 净辐射与心理常数/斜率
        Rn = Rns - Rnl
        P = self._pressure_from_elev(elev_m)
        gamma = self._psy_const(P)
        delta = self._slope_delta(tmean)
        # 2m 风速；缺省取 2 m/s
        u2 = df['wind'] if 'wind' in df.columns else pd.Series(2.0, index=df.index)
        # FAO-56 Penman-Monteith 计算日尺度 ET0
        et0 = (0.408 * delta * Rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
        return et0.clip(lower=0)

    def calculate_cwdi(self, daily_data, weights, lat_deg=None, elev_m=None):
        df = daily_data.copy()
        if df.empty:
            df["CWDI"] = pd.Series(index=df.index, dtype=float)
            return df
        if 'P' not in df.columns and 'precip' in df.columns:
            df = df.rename(columns={'precip': 'P'})

        if 'ET0' not in df.columns:
            if lat_deg is None and 'lat' in df.columns:
                lat_deg = float(df['lat'].iloc[0])
            if elev_m is None and 'altitude' in df.columns:
                elev_m = float(df['altitude'].iloc[0])
            df['ET0'] = self.penman_et0(df, lat_deg, elev_m)

        if 'kc' in df.columns:
            kc_series = df['kc']
        else:
            sy = int(df.index.year.min())
            ey = int(df.index.year.max())
            kc_series = _build_fixed_kc_series(sy, ey).reindex(df.index).fillna(0.0)
        df['ETc'] = kc_series * df['ET0']
        etc_shift = df['ETc'].shift(1)
        p_shift = df['P'].shift(1)
        w = np.array([weights[4], weights[3], weights[2], weights[1], weights[0]], dtype=float)
        def _cwdi_window(etc_window):
            p_window = p_shift.loc[etc_window.index].values
            etc_vals = etc_window.values
            if len(etc_vals) < 50:
                return np.nan
            etc_blocks = etc_vals.reshape(5, 10)
            p_blocks = p_window.reshape(5, 10)
            etc_sum = etc_blocks.sum(axis=1)
            p_sum = p_blocks.sum(axis=1)
            cond = (etc_sum > 0) & (etc_sum >= p_sum)
            cwdi_blocks = np.zeros(5, dtype=float)
            cwdi_blocks[cond] = (1 - p_sum[cond] / etc_sum[cond]) * 100.0
            return float(np.dot(w, cwdi_blocks))

        df['CWDI'] = etc_shift.rolling(window=50).apply(_cwdi_window, raw=False)
        return df

    def _normalize_array(self, array):
        if array.size == 0:
            return array
        mask = ~np.isnan(array)
        if not np.any(mask):
            return np.zeros_like(array)
        valid = array[mask]
        mn = np.min(valid)
        mx = np.max(valid)
        if mx == mn:
            out = np.full_like(array, 0.5, dtype=float)
            out[~mask] = np.nan
        else:
            out = (array - mn) / (mx - mn)
            out[~mask] = np.nan
        return out
    

    def _calc_drought_station_g(self, daily, cwdi_conf):
        # 计算站点 CWDI，并按指定生育期/年度窗口聚合为年度超阈积分
        df = self.calculate_cwdi(daily, cwdi_conf.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), cwdi_conf.get("lat_deg"), cwdi_conf.get("elev_m"))
        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        start_date_str = cwdi_conf.get("start_date")
        end_date_str = cwdi_conf.get("end_date")
        year_offset = int(cwdi_conf.get("year_offset", 0))
        if start_date_str and end_date_str and not series.empty:
            years = series.index.year.unique()
            masks = []
            for year in years:
                start_dt = pd.to_datetime(f"{year}-{start_date_str}")
                end_dt = pd.to_datetime(f"{year + year_offset}-{end_date_str}")
                masks.append((series.index >= start_dt) & (series.index <= end_dt))
            if masks:
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask | m
                series = series[mask]
        if series.empty:
            return np.nan
        years = sorted(series.index.year.unique())
        if not years:
            return np.nan
        sums = []
        for y in years:
            seg = series[series.index.year == y]
            # 超过 40 的日值求和
            exceed = np.where(seg > 40.0, seg - 40.0, 0.0)
            sums.append(float(np.nansum(exceed)))
        # 多年平均作为站点干旱强度 g
        return float(np.nanmean(sums))

    def calculate_drought(self, params):
        # 1) 读取配置与算法注册器
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cwdi_conf = algorithm_config.get('cwdi', {})
        cfg = params.get('config', {})

        # 2) 加载数据管理器与站点清单
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        station_ids = dm.get_all_stations()
        prov_kc_table = _get_prov_kc_table()

        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')
        # 3) 准备 Kc 配置与容器
        station_values = {}
        # 4) 并行计算站点干旱强度 g（省份按表赋值）
        sel_coords = {}
        args_common = (cfg.get('inputFilePath'), cfg.get('stationFilePath'), start_date, end_date, prov_kc_table, cwdi_conf)
        max_workers = 16
        chunk_size = max(1, len(station_ids) // (max_workers * 2) + 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in range(0, len(station_ids), chunk_size):
                chunk = station_ids[i:i+chunk_size]
                futures.append(ex.submit(_compute_g_batch_worker, (chunk, *args_common)))
            for fut in as_completed(futures):
                vals, coords = fut.result()
                station_values.update(vals)
                if isinstance(station_coords, dict) and station_coords:
                    for sid, coord in coords.items():
                        sel_coords[sid] = station_coords.get(sid, coord)
                else:
                    sel_coords.update(coords)

        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        station_df = pd.DataFrame(list(station_values.items()), columns=["station_id", "GH"])
        station_df.to_csv(
            str(inter_dir / "干旱强度指数_站点计算结果.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        valid = pd.to_numeric(station_df["GH"], errors="coerce")
        mask = valid.notna()
        if mask.any():
            mn = float(valid[mask].min())
            mx = float(valid[mask].max())
            if mx == mn:
                station_df["GH_norm"] = np.where(mask, 0.5, np.nan)
            else:
                station_df["GH_norm"] = np.where(mask, (valid - mn) / (mx - mn), np.nan)
        else:
            station_df["GH_norm"] = np.nan
        station_df[["station_id", "GH_norm"]].to_csv(
            str(inter_dir / "干旱强度指数_站点计算结果_归一化.csv"),
            index=False,
            encoding="utf-8-sig"
        )
    
        # 5) 空间插值生成干旱强度栅格
        interp = self._interpolate(station_values, sel_coords, cfg, algorithm_config)
        interp_data = np.maximum(interp['data'], 0.0)
        
        # 6) 输出中间与最终结果到 GeoTIFF
        tif_path = str(inter_dir / "干旱强度指数.tif")
        self._save_geotiff_gdal(interp_data.astype(np.float32), interp['meta'], tif_path, 0)
        class_conf = algorithm_config.get('classification', {})
        data_out = interp_data
    
        method = class_conf.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{method}")
        if method == 'custom_thresholds':
            data_out = classifier.execute(interp_data.astype(float), class_conf)
            final_tif = str(out_dir / "干旱强度指数_分级.tif")
            self._save_geotiff_gdal(np.array(data_out).astype(np.float32), interp['meta'], final_tif, 0)
        else:
            norm = self._normalize_array(interp_data)
            norm_tif = str(inter_dir / "干旱强度指数_归一化.tif")
            self._save_geotiff_gdal(norm.astype(np.float32), interp['meta'], norm_tif, 0)
            data_out = classifier.execute(norm.astype(float), class_conf)
            final_tif = str(out_dir / "干旱强度指数_分级.tif")
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
        if d == 'GH':
            return self.calculate_drought(params)
        raise ValueError(f"不支持的灾害类型: {d}")
