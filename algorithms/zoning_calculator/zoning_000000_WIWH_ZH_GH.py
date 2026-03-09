import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi
from osgeo import gdal
from osgeo import osr
from algorithms.data_manager import DataManager
from tqdm import tqdm
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

 
def _compute_g_batch_worker(args):
    sids, input_path, station_file_path, start_date, end_date, prov_kc_table, cwdi_conf = args
    dm = DataManager(input_path, station_file_path, multiprocess=False, num_processes=1)
    out_vals = {}
    out_coords = {}
    sy = int(str(start_date)[:4])
    ey = int(str(end_date)[:4])
    calc = WIWH_ZH()
    for sid in sids:
        daily = dm.load_station_data(sid, start_date, end_date)
        info = dm.get_station_info(sid)
        st_prov = str(info.get('province', '')).strip()
        if st_prov in prov_kc_table:
            months_map = prov_kc_table[st_prov]
            kc_series = _build_kc_series_from_months_map(months_map, sy, ey)
            daily['kc'] = kc_series.reindex(daily.index).fillna(0.0)
        else:
            daily['kc'] = pd.Series(0.0, index=daily.index)
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


def _prepare_mask_sampler(mask_path):
    ds = gdal.Open(mask_path)
    gt = ds.GetGeoTransform()
    inv_val = gdal.InvGeoTransform(gt)
    if isinstance(inv_val, (list, tuple)):
        if len(inv_val) == 2 and isinstance(inv_val[1], (list, tuple)):
            inv_gt = inv_val[1]
        else:
            inv_gt = inv_val
    else:
        inv_gt = gt  # 退化为正向变换以避免类型错误（坐标转换仍保证定位）
    proj = ds.GetProjection()
    srs_ds = osr.SpatialReference()
    if proj:
        srs_ds.ImportFromWkt(proj)
    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)
    transformer = None
    if srs_ds and not srs_ds.IsSame(srs_wgs84):
        transformer = osr.CoordinateTransformation(srs_wgs84, srs_ds)
    band = ds.GetRasterBand(1)
    def sample(lon, lat):
        X, Y = lon, lat
        if transformer is not None:
            X, Y, _ = transformer.TransformPoint(lon, lat)
        px, py = gdal.ApplyGeoTransform(inv_gt, X, Y)
        ix, iy = int(np.floor(px)), int(np.floor(py))
        if ix < 0 or iy < 0 or ix >= ds.RasterXSize or iy >= ds.RasterYSize:
            return 0
        arr = band.ReadAsArray(ix, iy, 1, 1)
        return arr[0, 0]
    return sample


def _mask_to_target_grid(mask_path, meta):
    src = gdal.Open(mask_path)
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', meta['width'], meta['height'], 1, gdal.GDT_Byte)
    dst.SetGeoTransform(meta['transform'])
    dst.SetProjection(meta['crs'])
    gdal.Warp(dst, src, resampleAlg=gdal.GRA_NearestNeighbour)
    arr = dst.GetRasterBand(1).ReadAsArray()
    return arr

class WIWH_ZH:
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
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        # 组织插值所需输入，包含站点值/坐标、DEM/行政区/规则格网等路径
        grid_path = config.get("gridFilePath", "")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': grid_path,
            'mask_path': config.get("maskFilePath", ""),
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
        # 读取并校验必要列
        req = ['生育期', '开始日期', '结束日期', 'kc']
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Excel缺少列:{c}")
        # 将 Excel 数值日期转为 “M月D日” 字符串，便于解析
        df['开始日期'] = df['开始日期'].apply(self._excel_num_to_cn_date)
        df['结束日期'] = df['结束日期'].apply(self._excel_num_to_cn_date)
        dates = []
        vals = []
        for _, r in df.iterrows():
            # 逐行展开为跨年的日序列（若结束早于开始则顺延至次年）
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
        # 去重并按时间排序得到日尺度 kc 序列
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

    def calculate_cwdi(self, daily_data, weights):
        df = daily_data.copy()
        # 字段兼容：降水 P / precip；缺失 ET0 时按站点信息估算
        if 'P' not in df.columns and 'precip' in df.columns:
            df = df.rename(columns={'precip': 'P'})
        if 'ET0' not in df.columns:
            lat_deg = float(df['lat'].iloc[0]) if 'lat' in df.columns else 35.0
            elev_m = float(df['altitude'].iloc[0]) if 'altitude' in df.columns else 0.0
            df['ET0'] = self.penman_et0(df, lat_deg, elev_m)
        # Kc 来源优先 df['kc']，否则按月份赋缺省值
        if 'kc' in df.columns:
            kc_series = df['kc']
        else:
            kc_series = pd.Series(0.0, index=df.index)
            m = df.index.month
            kc_map = {10: 0.67, 11: 0.70, 12: 0.74, 1: 0.64, 2: 0.64, 3: 0.90, 4: 1.22, 5: 1.13, 6: 0.83}
            for mo, v in kc_map.items():
                kc_series[m == mo] = v
        # 作物需水 ETc = Kc * ET0
        df['ETc'] = kc_series * df['ET0']
        # 滞后 1 天聚合（与 CWDI 定义保持一致）
        etc_shift = df['ETc'].shift(1)
        p_shift = df['P'].shift(1)
        # 5×10 天滑动窗权重，从近到远为 w0..w4，这里做反序
        w = np.array([weights[4], weights[3], weights[2], weights[1], weights[0]], dtype=float)
        def _cwdi_window(etc_window):
            # 将 50 天序列按 5 个 10 天段分块并求和，计算段旱情
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
        # 50 天滑动窗口计算 CWDI
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
            # 超过 40 的日值求和（不减去 40）
            exceed = np.where(seg > 40.0, seg, 0.0)
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
        mask_path = cfg.get('maskFilePath')
        candidate_ids = list(station_coords.keys()) if isinstance(station_coords, dict) and station_coords else dm.get_all_stations()
        available = set(dm.get_all_stations())
        candidate_ids = [sid for sid in candidate_ids if sid in available]
        station_ids = []
        prov_kc_table = _get_prov_kc_table()
        allowed_names = set(prov_kc_table.keys())
        prov_counts = {}
        for sid in candidate_ids:
            info = dm.get_station_info(sid)
            st_prov = str(info.get('province', '')).strip()
            if st_prov in allowed_names:
                station_ids.append(sid)
                prov_counts[st_prov] = prov_counts.get(st_prov, 0) + 1
        for name, cnt in prov_counts.items():
            print(f"{name}：{cnt}个")
        print(f"总计：{sum(prov_counts.values())}个")

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
    
        # 5) 空间插值生成干旱强度栅格（仅掩膜区域）
        interp = self._interpolate(station_values, sel_coords, cfg, algorithm_config)
        mask_arr = _mask_to_target_grid(mask_path, interp['meta'])
        interp_data_masked = np.where(mask_arr == 1, np.maximum(interp['data'], 0.0), np.nan)
        
        # 6) 输出中间与最终结果到 GeoTIFF
        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "干旱强度指数.tif")
        self._save_geotiff_gdal(interp_data_masked.astype(np.float32), interp['meta'], tif_path, 0)
        class_conf = algorithm_config.get('classification', {})
        data_out = interp_data_masked
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
            # 自定义阈值分级：仅在掩膜区域内分段
            data_out = classifier.execute(interp_data_masked.astype(float), class_conf)
            data_out = np.where(mask_arr == 1, data_out, np.nan)
            final_tif = str(out_dir / "干旱强度指数_分级.tif")
            self._save_geotiff_gdal(np.array(data_out).astype(np.float32), interp['meta'], final_tif, 0)
        else:
            # 其它分级：先归一化到 [0,1]（仅掩膜区域）再分级
            norm = self._normalize_array(interp_data_masked)
            norm_tif = str(inter_dir / "干旱强度指数_归一化.tif")
            self._save_geotiff_gdal(norm.astype(np.float32), interp['meta'], norm_tif, 0)
            data_out = classifier.execute(norm.astype(float), class_conf)
            data_out = np.where(mask_arr == 1, data_out, np.nan)
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
