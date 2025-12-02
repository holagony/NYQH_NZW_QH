import os
import numpy as np
import pandas as pd
from osgeo import gdal
from math import pi
from typing import Dict, Any
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation


def normalize_array(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return array

    # 创建一个掩码来标识非NaN值
    mask = ~np.isnan(array)

    if not np.any(mask):
        return np.zeros_like(array)

    valid_values = array[mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    # 如果所有有效值都相同，归一化到0.5
    if max_val == min_val:
        normalized_array = np.full_like(array, 0.5, dtype=float)
        normalized_array[~mask] = np.nan
    else:
        normalized_array = (array - min_val) / (max_val - min_val)
        normalized_array[~mask] = np.nan

    return normalized_array


def penman_et0(daily_data, lat_deg, elev_m, albedo=0.23, as_coeff=0.25, bs_coeff=0.5, k_rs=0.16):
    df = daily_data.copy()
    tmax = df['tmax']
    tmin = df['tmin']
    tmean = df['tavg'] if 'tavg' in df.columns else (tmax + tmin) / 2.0

    phi = np.deg2rad(lat_deg)
    J = df.index.dayofyear
    dr = 1.0 + 0.033 * np.cos(2.0 * pi / 365.0 * J)
    delta = 0.409 * np.sin(2.0 * pi / 365.0 * J - 1.39)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    Ra = (24.0 * 60.0 / pi) * 0.0820 * dr * (omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s))
    N = 24.0 / pi * omega_s

    if 'sunshine' in df.columns:
        n = df['sunshine']
        Rs = (as_coeff + bs_coeff * (n / N)) * Ra  # 实测日照时数估算入射短波辐射
    else:
        Rs = k_rs * np.sqrt(np.maximum(tmax - tmin, 0.0)) * Ra  # 无日照时数时用温差估算方法

    Rso = (0.75 + 2e-5 * elev_m) * Ra  # 晴空辐射
    Rns = (1.0 - albedo) * Rs

    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0  # 平均饱和水汽压(kPa)
    ea = es * (df['rhum'] / 100.0) if 'rhum' in df.columns else es * 0.7  # 缺湿度时经验系数

    sigma = 4.903e-9
    tmaxK = tmax + 273.16
    tminK = tmin + 273.16

    # 净长波辐射，含湿度与云量校正
    Rnl = sigma * (
        (tmaxK**4 + tminK**4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
    Rn = Rns - Rnl

    P = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0)**5.26
    gamma = 0.000665 * P
    es_tmean = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098.0 * es_tmean / ((tmean + 237.3)**2)
    u2 = df['wind'] if 'wind' in df.columns else pd.Series(2.0, index=df.index)

    # Penman-Monteith 主公式
    et0 = (0.408 * delta * (Rn) + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
    return et0.clip(lower=0)


def calculate_cwdi(daily_data, weights, lat_deg=None, elev_m=None, kc_map=None):
    df = daily_data.copy()
    if 'P' not in df.columns and 'precip' in df.columns:
        df = df.rename(columns={'precip': 'P'})

    if 'ET0' not in df.columns:
        if lat_deg is None and 'lat' in df.columns:
            lat_deg = float(df['lat'].iloc[0])
        if elev_m is None and 'altitude' in df.columns:
            elev_m = float(df['altitude'].iloc[0])
        df['ET0'] = penman_et0(df, lat_deg, elev_m)

    kc_series = pd.Series(0.0, index=df.index)
    m = df.index.month

    # kc系数
    default_kc_map = {10: 0.67, 11: 0.70, 12: 0.74, 1: 0.64, 2: 0.64, 3: 0.90, 4: 1.22, 5: 1.13, 6: 0.83}
    use_kc_map = kc_map if kc_map is not None else default_kc_map

    norm_map = {}
    for k, v in use_kc_map.items():
        try:
            kk = int(k)
            vv = float(v)
            norm_map[kk] = vv
        except Exception:
            pass

    for mo, v in norm_map.items():
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


class ZH_GH:
    '''
    干旱通用算法
    '''

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        """保存GeoTIFF文件"""

        # 根据输入数据的 dtype 确定 GDAL 数据类型
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
            datatype = gdal.GDT_Float32  # 默认情况

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])

        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])

        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)

        band.FlushCache()
        dataset = None

    def drought_station_g(self, data, config):
        '''
        计算每个站点的干旱风险性
        
        自定义传参说明：
        start : 段起始日期， "MM-DD" 格式
        end : 段结束日期， "MM-DD" 格式
        start_offset : 起始日期相对基准年的偏移（整数，默认 0 ）
        end_offset : 结束日期相对基准年的偏移（整数，默认 0 ）
        基准年为当前遍历的 y ，支持跨年段（例如上一年用 -1 ，当年用 0 ）
        weight: 对应权重

        格式如下：
        {
        "growingPeriod": [
            {"start": "08-01", "end": "10-10", "start_offset": -1, "end_offset": -1, "weight": 0.09},
            {"start": "10-11", "end": "12-20", "start_offset": -1, "end_offset": -1, "weight": 0.13},
            {"start": "12-21", "end": "02-20", "start_offset": -1, "end_offset": 0,  "weight": 0.11}
            ]
        }

        kc系数：
          "kcMap": {
                        "10": 0.67, "11": 0.70, "12": 0.74,
                        "1": 0.64, "2": 0.64, "3": 0.90,
                        "4": 1.22, "5": 1.13, "6": 0.83
                    }
        '''
        df = calculate_cwdi(data, config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("lat_deg"), config.get("elev_m"),
                            config.get("kcMap"))

        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        years = sorted(series.index.year.unique())
        if not years:
            return np.nan

        # 自定义传参
        range_defs = config.get("growingPeriod", [])
        CWDI = []
        for y in years:
            if range_defs != []:
                ranges = []
                for r in range_defs:
                    s = str(r.get("start", "01-01"))
                    e = str(r.get("end", "12-31"))
                    s_m, s_d = map(int, s.split("-"))
                    e_m, e_d = map(int, e.split("-"))
                    s_off = int(r.get("start_offset", 0))
                    e_off = int(r.get("end_offset", 0))
                    ranges.append((pd.Timestamp(y + s_off, s_m, s_d), pd.Timestamp(y + e_off, e_m, e_d)))

                # 提取对应权重
                w_vals = [r.get("weight") for r in range_defs]
                w = np.array(w_vals, dtype=float)

            # 河南报告P49
            else:
                ranges = [(pd.Timestamp(y - 1, 8, 1), pd.Timestamp(y - 1, 10, 10)), (pd.Timestamp(y - 1, 10, 11), pd.Timestamp(y - 1, 12, 20)),
                          (pd.Timestamp(y - 1, 12, 21), pd.Timestamp(y, 2, 20)), (pd.Timestamp(y, 2, 21), pd.Timestamp(y, 3, 31)),
                          (pd.Timestamp(y, 4, 1), pd.Timestamp(y, 4, 30)), (pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 20)),
                          (pd.Timestamp(y, 5, 21), pd.Timestamp(y, 6, 10))]
                w = np.array([0.09, 0.13, 0.11, 0.12, 0.20, 0.22, 0.13], dtype=float)

            k_cwdi = []
            for s, e in ranges:
                seg = series[(series.index >= s) & (series.index <= e)]
                k_cwdi.append(float(seg.mean()) if len(seg) > 0 else np.nan)

            k_cwdi = np.array(k_cwdi, dtype=float)
            CWDI.append(float(np.nansum(w * k_cwdi)))

        risk = float(np.nanmean(CWDI))

        return risk

    def calculate_drought(self, params):
        '''
        干旱区划
        '''
        station_coords = params.get('station_coords')
        algorithm_config = params.get('algorithmConfig')
        cwdi_config = algorithm_config.get('cwdi')
        cfg = params.get('config')
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')

        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())

        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')
        station_values: Dict[str, float] = {}

        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            g = self.drought_station_g(daily, cwdi_config)
            station_values[sid] = float(g) if np.isfinite(g) else np.nan

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

        # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])
        result['data'] = normalize_array(result['data'])  # 归一化
        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        meta = result['meta']
        self._save_geotiff_gdal(result['data'], meta, g_tif_path, 0)

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

    def calculate(self, params):
        self._algorithms = params['algorithms']
        return self.calculate_drought(params)
