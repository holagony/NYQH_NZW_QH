import os
import numpy as np
import pandas as pd
# 果实膨大期高温热害通用计算器（参数化）
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from osgeo import gdal


def _diffusion_h(a, b, m):
    # 信息扩散带宽取值（样本数自适应）
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
    # 信息扩散法计算论域上的概率分布
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
    # 数组归一化到 0-1（保留 NaN）
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


class ZH_GSPDQGWRH:
    def _save_geotiff(self, data, meta, output_path, nodata=0):
        # 保存 GeoTIFF 栅格
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
        # 站点级危险性计算：
        # 1) 按年度窗口筛选；支持跨年（end_date 早于 start_date）
        # 2) 使用 JSON 提供的温度阈值判据（tavg_threshold / tmax_threshold）
        # 3) 对满足判据的连续段做“游程统计”，按 threshold 分箱计数
        # 4) 信息扩散得到次数为0的概率 p0，计算 occ=1-p0
        # 5) 使用各分箱的 weight 加权，得到 H
        start_date_str = conf.get("start_date")
        end_date_str = conf.get("end_date")

        years = sorted(data.index.year.unique())
        thresholds = conf.get('threshold', [])  # 轻/中/重等分箱定义，含 min/max/weight
        bins = thresholds if isinstance(thresholds, list) else []
        Ys = [[] for _ in bins]

        for y in years:
            if start_date_str and end_date_str:
                start_dt = pd.to_datetime(f"{y}-{start_date_str}")
                end_dt = pd.to_datetime(f"{y}-{end_date_str}")
                dfy = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]
            else:
                dfy = data[data.index.year == y]
            if dfy.empty:
                for k in range(len(bins)):
                    Ys[k].append(0)
                continue

            ta_thr = float(conf.get('tavg_threshold', 30))  # 日平均温度阈值（JSON可配置）
            tm_thr = float(conf.get('tmax_threshold', 36))  # 日最高温度阈值（JSON可配置）
            cond = (dfy["tavg"] >= ta_thr) & (dfy["tmax"] >= tm_thr)

            runs = []  # 连续满足天数的游程长度列表
            c = 0
            for v in cond.values:
                if v:
                    c += 1
                else:
                    if c > 0:
                        runs.append(c)
                        c = 0
            if c > 0:
                runs.append(c)
            counts = [0 for _ in bins]  # 各分箱（如轻/中/重）年度次数（首匹配归属）
            for L in runs:
                for i, b in enumerate(bins):
                    mn = b.get('min')
                    mx = b.get('max')
                    mn_i = int(mn) if mn not in (None, "") else 0
                    if mx in (None, ""):
                        if L >= mn_i:
                            counts[i] += 1
                            break
                    else:
                        try:
                            mx_i = int(mx)
                        except Exception:
                            mx_i = None
                        if mx_i is not None and (mn_i <= L <= mx_i):
                            counts[i] += 1
                            break

            for k in range(len(bins)):
                Ys[k].append(counts[k])

        occs = []  # 各分箱事件出现概率的互补（不出现概率的互补）
        for k in range(len(bins)):
            U = list(range(0, (max(Ys[k]) if Ys[k] else 0) + 1)) or [0]
            P = information_diffusion_probabilities(U, Ys[k])
            idx0 = U.index(0) if 0 in U else None
            p0 = float(P[idx0]) if idx0 is not None and len(P) > idx0 else 0.0
            occs.append(1.0 - p0)
        
        weights = []  # 与 threshold 中各分箱的 weight 一一对应
        for b in bins:
            w = b.get('weight', 0)
            try:
                weights.append(float(w))
            except Exception:
                weights.append(0.0)

        H = float(np.dot(np.array(weights, dtype=float), np.array(occs, dtype=float))) if weights and occs else float('nan')
        return H

    def calculate(self, params):
        # 主流程：站点危险性计算 → 空间插值 → 归一化 → 分级
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
        out_path = os.path.join(cfg.get("resultPath"), "intermediate", "高温热害危险性指数.tif")
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
