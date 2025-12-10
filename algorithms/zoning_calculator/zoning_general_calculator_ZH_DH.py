import os
import numpy as np
import pandas as pd
# 冻害（DH）通用计算器：统计 tmin 满足阈值的日数，
# 信息扩散法得到“日数为0”的概率 p0，取互补 occ=1-p0，
# 按配置权重加权得到站点危险性指数 H，随后插值与分级。
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from osgeo import gdal


def _diffusion_h(a, b, m):
    # 信息扩散带宽 h 的取值规则（随样本数 m 变化）
    if m == 5:
        return 0.8146 * (b - a)
    if m == 6:
        return 0.5690 * (b - a)
    if m == 7:
        return 0.4560 * (b - a)
    if m == 8:
        return 0.3860 * (b - a)
    if m == 9:
        return 0.3362 * (b - a)
    if m == 10:
        return 0.2986 * (b - a)
    if m >= 11:
        return 2.6851 * (b - a) / (m - 1)
    raise ValueError("m must be >= 5")


def information_diffusion_probabilities(U, Y):
    # 在论域 U 上进行信息扩散，得到概率分布 p(u)
    if len(Y) == 0:
        return np.zeros(len(U), dtype=float)
    m = len(Y)
    a = min(Y)
    b = max(Y)
    if m < 5:
        h = (b - a)
        if h <= 0:
            u_range = (max(U) - min(U)) if U else 1.0
            h = max(1e-6, u_range / max(1, len(U)))
        else:
            h = max(1e-6, 0.5 * h)
    else:
        h = _diffusion_h(a, b, m)
    if h <= 0:
        u_range = (max(U) - min(U)) if U else 1.0
        h = max(1e-6, u_range / max(1, len(U)))
    f = []
    for y in Y:
        row = [1.0 / (h * np.sqrt(2 * np.pi)) * np.exp(-((y - u) ** 2) / (2 * h * h)) for u in U]
        f.append(row)
    f = np.array(f, dtype=float)
    c = f.sum(axis=1, keepdims=True)
    mu = f / c
    q = mu.sum(axis=0)
    Q = float(q.sum())
    p = q / Q
    return p


def normalize_array(array):
    # 栅格归一化到 [0,1]，保持 NaN 原样
    if array.size == 0:
        return array
    mask = ~np.isnan(array)
    if not np.any(mask):
        return np.zeros_like(array)
    valid = array[mask]
    min_val = np.min(valid)
    max_val = np.max(valid)
    if max_val == min_val:
        out = np.full_like(array, 0.5, dtype=float)
        out[~mask] = np.nan
    else:
        out = (array - min_val) / (max_val - min_val)
        out[~mask] = np.nan
    return out


class ZH_DH:
    def _save_geotiff(self, data, meta, output_path, nodata=0):
        # 保存 GeoTIFF（压缩 LZW，保持空间参考与仿射参数）
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

    def _calc_station_H(self, data, conf):
        # 站点级危险性计算流程：
        # 1) 年度窗口筛选（日最低气温 tmin）
        # 2) 统计满足阈值的日数（轻/中/重）
        # 3) 信息扩散求 p0（日数为0的概率），取 occ=1-p0
        # 4) 按权重加权得到 H
        tmin = data["tmin"] if "tmin" in data.columns else pd.Series(dtype=float)
        if tmin.empty:
            return float('nan')

        start_date_str = conf.get("start_date")
        end_date_str = conf.get("end_date")
        
        # 检查跨年情况（保持原有逻辑）
        start_date = pd.to_datetime(f"2000-{start_date_str}")
        end_date = pd.to_datetime(f"2000-{end_date_str}")
        if end_date < start_date:
            year_offset_conf = 1
        
        # year_offset_conf = conf.get("year_offset")
        year_offset = int(year_offset_conf) if year_offset_conf is not None else 0

        years = sorted(data.index.year.unique())
        thresholds = conf.get('threshold', [])  # 轻/中/重三个等级的阈值与权重

        def count_bin(vals, b, left_open=False):
            # 兼容 JSON 中 min/max 为空字符串的情况：
            # - 仅 min 有值 → 统计 >=min（或 >min 当 left_open=True）
            # - 仅 max 有值 → 统计 <=max
            # - min/max 都有值 → 统计 [min, max]（或 (min, max] 当 left_open=True）
            mn_raw = b.get('min')
            mx_raw = b.get('max')
            mn = float(mn_raw) if mn_raw not in (None, "") else None
            mx = float(mx_raw) if mx_raw not in (None, "") else None
            if mn is None and mx is None:
                return 0
            if mx is None:
                if mn is None:
                    return 0
                if left_open:
                    return int(np.count_nonzero(vals > mn))
                return int(np.count_nonzero(vals >= mn))
            if mn is None:
                return int(np.count_nonzero(vals <= mx))
            if left_open:
                return int(np.count_nonzero((vals > mn) & (vals <= mx)))
            return int(np.count_nonzero((vals >= mn) & (vals <= mx)))

        Y_low, Y_mid, Y_high = [], [], []
        for y in years:
            if start_date_str and end_date_str:
                start_dt = pd.to_datetime(f"{y}-{start_date_str}")
                end_dt = pd.to_datetime(f"{y + year_offset}-{end_date_str}")
                mask = (data.index >= start_dt) & (data.index <= end_dt)
                dfy = data.loc[mask]
            else:
                dfy = data[data.index.year == y]
            if dfy.empty:
                Y_low.append(0); Y_mid.append(0); Y_high.append(0)
                continue

            tvals = dfy["tmin"].to_numpy()
            # 等级定义（与 CITR 逻辑一致），兼容 JSON 空边界：
            b0, b1, b2 = thresholds[0], thresholds[1], thresholds[2]
            light = count_bin(tvals, b0, left_open=True)
            medium = count_bin(tvals, b1, left_open=True)
            heavy = count_bin(tvals, b2, left_open=False)

            Y_low.append(light)
            Y_mid.append(medium)
            Y_high.append(heavy)

        U_low = list(range(0, (max(Y_low) if Y_low else 0) + 1)) or [0]
        U_mid = list(range(0, (max(Y_mid) if Y_mid else 0) + 1)) or [0]
        U_high = list(range(0, (max(Y_high) if Y_high else 0) + 1)) or [0]

        Pl = information_diffusion_probabilities(U_low, Y_low)
        Pm = information_diffusion_probabilities(U_mid, Y_mid)
        Ps = information_diffusion_probabilities(U_high, Y_high)

        idx0_l = U_low.index(0) if 0 in U_low else None
        idx0_m = U_mid.index(0) if 0 in U_mid else None
        idx0_s = U_high.index(0) if 0 in U_high else None
        p0_l = float(Pl[idx0_l]) if idx0_l is not None and len(Pl) > idx0_l else 0.0
        p0_m = float(Pm[idx0_m]) if idx0_m is not None and len(Pm) > idx0_m else 0.0
        p0_s = float(Ps[idx0_s]) if idx0_s is not None and len(Ps) > idx0_s else 0.0
        occ_l = 1.0 - p0_l
        occ_m = 1.0 - p0_m
        occ_s = 1.0 - p0_s

        # 权重与等级一一对应
        w1 = float(thresholds[0].get('weight', 0)) if len(thresholds) >= 1 else 0.0
        w2 = float(thresholds[1].get('weight', 0)) if len(thresholds) >= 2 else 0.0
        w3 = float(thresholds[2].get('weight', 0)) if len(thresholds) >= 3 else 0.0
        H = w1 * occ_l + w2 * occ_m + w3 * occ_s
        return float(H)

    def calculate(self, params):
        # 主流程：站点危险性 → 空间插值 → 归一化 → 分级
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)

        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')

        station_values = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            H = self._calc_station_H(daily, algorithm_config)
            station_values[sid] = float(H) if np.isfinite(H) else np.nan

        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})
        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }
        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        result['data'] = normalize_array(result['data'])
        out_path = os.path.join(cfg.get("resultPath"), "intermediate", "低温冷害危险性指数.tif")
        self._save_geotiff(result['data'], result['meta'], out_path, 0)

        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            algos = params.get('algorithms', {})
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)

        return {
            'data': result['data'],
            'meta': {
                'width': result['meta']['width'],
                'height': result['meta']['height'],
                'transform': result['meta']['transform'],
                'crs': result['meta']['crs']
            }
        }

