import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi
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


class WIWH_ZH:
    """全国冬小麦干旱区划计算器"""

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

    def _excel_num_to_cn_date(self, v):
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float)):
            base = pd.Timestamp('1899-12-30')
            dt = base + pd.Timedelta(days=float(v))
            return f"{dt.month}月{dt.day}日"
        return str(v)

    def _parse_cn_date(self, s):
        t = str(s).strip()
        off = 0
        if "上年" in t:
            off = -1
            t = t.replace("上年", "")
        if "当年" in t:
            t = t.replace("当年", "")
        if ("月" in t) and ("日" in t):
            mm = int(t.split("月")[0].strip())
            dd = int(t.split("月")[1].split("日")[0].strip())
            return pd.Timestamp(2000, mm, dd), off
        return None, off

    def _load_kc_series_from_excel(self, excel_path, start_year, end_year):
        df = pd.read_excel(excel_path, sheet_name='干旱区划模板')
        req = ['生育期', '开始日期', '结束日期', 'kc']
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Excel缺少列:{c}")
        df['开始日期'] = df['开始日期'].apply(self._excel_num_to_cn_date)
        df['结束日期'] = df['结束日期'].apply(self._excel_num_to_cn_date)
        dates = []
        vals = []
        for _, r in df.iterrows():
            s_txt = r['开始日期']
            e_txt = r['结束日期']
            kc = float(r['kc'])
            s_md, s_off = self._parse_cn_date(s_txt)
            e_md, e_off = self._parse_cn_date(e_txt)
            if s_md is None or e_md is None:
                continue
            for y in range(int(start_year), int(end_year) + 1):
                s_dt = pd.Timestamp(y + s_off, s_md.month, s_md.day)
                e_dt = pd.Timestamp(y + e_off, e_md.month, e_md.day)
                if e_dt < s_dt:
                    e_dt = pd.Timestamp(e_dt.year + 1, e_dt.month, e_dt.day)
                rng = pd.date_range(start=s_dt, end=e_dt, freq='D')
                dates.extend(list(rng))
                vals.extend([kc] * len(rng))
        if not dates:
            raise ValueError("未生成KC数据")
        s = pd.Series(vals, index=pd.DatetimeIndex(dates))
        s = s[~s.index.duplicated(keep='last')].sort_index()
        return s

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
        dr = 1.0 + 0.033 * np.cos(2.0 * pi / 365.0 * J)
        delta = 0.409 * np.sin(2.0 * pi / 365.0 * J - 1.39)
        omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
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
        if 'sunshine' in df.columns:
            n = df['sunshine']
            Rs = (as_coeff + bs_coeff * (n / N)) * Ra
        else:
            Rs = k_rs * np.sqrt(np.maximum(tmax - tmin, 0.0)) * Ra
        Rso = (0.75 + 2e-5 * elev_m) * Ra
        Rns = (1.0 - albedo) * Rs
        es_tmax = self._sat_vapor_pressure(tmax)
        es_tmin = self._sat_vapor_pressure(tmin)
        es = (es_tmax + es_tmin) / 2.0
        ea = es * (df['rhum'] / 100.0) if 'rhum' in df.columns else es * 0.7
        sigma = 4.903e-9
        tmaxK = tmax + 273.16
        tminK = tmin + 273.16
        Rnl = sigma * ((tmaxK ** 4 + tminK ** 4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
        Rn = Rns - Rnl
        P = self._pressure_from_elev(elev_m)
        gamma = self._psy_const(P)
        delta = self._slope_delta(tmean)
        u2 = df['wind'] if 'wind' in df.columns else pd.Series(2.0, index=df.index)
        et0 = (0.408 * delta * Rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
        return et0.clip(lower=0)

    def calculate_cwdi(self, daily_data, weights):
        df = daily_data.copy()
        if 'P' not in df.columns and 'precip' in df.columns:
            df = df.rename(columns={'precip': 'P'})
        if 'ET0' not in df.columns:
            lat_deg = float(df['lat'].iloc[0]) if 'lat' in df.columns else 35.0
            elev_m = float(df['altitude'].iloc[0]) if 'altitude' in df.columns else 0.0
            df['ET0'] = self.penman_et0(df, lat_deg, elev_m)
        if 'kc' in df.columns:
            kc_series = df['kc']
        else:
            kc_series = pd.Series(0.0, index=df.index)
            m = df.index.month
            kc_map = {10: 0.67, 11: 0.70, 12: 0.74, 1: 0.64, 2: 0.64, 3: 0.90, 4: 1.22, 5: 1.13, 6: 0.83}
            for mo, v in kc_map.items():
                kc_series[m == mo] = v
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
    
    def _kc_series_from_table(self, area_code, kc_table, start_year, end_year):
        months_map = {}
        if isinstance(kc_table, dict):
            row = kc_table.get(str(area_code)) or kc_table.get(int(area_code)) if area_code is not None else None
            if isinstance(row, dict):
                for k, v in row.items():
                    try:
                        m = int(k)
                        months_map[m] = float(v)
                    except:
                        continue
        start_dt = pd.Timestamp(int(start_year), 1, 1)
        end_dt = pd.Timestamp(int(end_year), 12, 31)
        idx = pd.date_range(start=start_dt, end=end_dt, freq='D')
        s = pd.Series(0.0, index=idx)
        if months_map:
            for m, val in months_map.items():
                s[idx.month == int(m)] = float(val)
        return s

    def _calc_drought_station_g(self, daily, cwdi_conf):
        df = self.calculate_cwdi(daily, cwdi_conf.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]))
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
            exceed = np.maximum(seg - 40.0, 0.0)
            sums.append(float(np.nansum(exceed)))
        return float(np.nanmean(sums))

    def calculate_drought(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cwdi_conf = algorithm_config.get('cwdi', {})
        cfg = params.get('config', {})
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')
        start_year = int(str(start_date)[:4])
        end_year = int(str(end_date)[:4])
        area_code = cfg.get('areaCode')
        kc_table = cfg.get('kc_table', {})
        if not kc_table:
            kc_table = cwdi_conf.get('kc_table', {})
        kc_series_all = self._kc_series_from_table(area_code, kc_table, start_year, end_year)
        station_values = {}
        area_name_map = {
            "140000": "山西省",
            "130000": "河北省",
            "410000": "河南省",
            "370000": "山东省",
            "340000": "安徽省",
            "320000": "江苏省"
        }
        expected_prov = str(cfg.get('provinceName') or area_name_map.get(str(area_code), '')).strip()
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            info = dm.get_station_info(sid)
            st_code = str(info.get('PAC_prov', ''))
            st_prov = str(info.get('province', '')).strip()
            if (st_code and (st_code == str(area_code))) or (expected_prov and (st_prov == expected_prov)):
                daily['kc'] = kc_series_all.reindex(daily.index).fillna(0.0)
            else:
                daily['kc'] = pd.Series(0.0, index=daily.index)
            g = self._calc_drought_station_g(daily, cwdi_conf)
            station_values[sid] = float(g) if np.isfinite(g) else np.nan
        interp = self._interpolate(station_values, station_coords, cfg, algorithm_config)
        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "干旱强度指数.tif")
        self._save_geotiff_gdal(interp['data'].astype(np.float32), interp['meta'], tif_path, 0)
        class_conf = algorithm_config.get('classification', {})
        data_out = interp['data']
        if not class_conf:
            class_conf = {
                'method': 'custom_thresholds',
                'thresholds': [
                    {'min': 0, 'max': 400, 'level': 1, 'label': '轻旱'},
                    {'min': 400, 'max': 1000, 'level': 2, 'label': '中旱'},
                    {'min': 1000, 'max': 2000, 'level': 3, 'label': '重旱'},
                    {'min': 2000, 'max': '', 'level': 4, 'label': '特旱'}
                ]
            }
        method = class_conf.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{method}")
        if method == 'custom_thresholds':
            data_out = classifier.execute(interp['data'].astype(float), class_conf)
            final_tif = str(out_dir / "干旱强度指数_分级.tif")
            self._save_geotiff_gdal(np.array(data_out).astype(np.float32), interp['meta'], final_tif, 0)
        else:
            norm = self._normalize_array(interp['data'])
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
