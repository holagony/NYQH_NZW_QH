import os
import numpy as np
import pandas as pd
from osgeo import gdal
from math import pi
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation


def normalize_array(array: np.ndarray) -> np.ndarray:
    """将数组归一化到[0,1]，保留NaN；全常数时置为0.5"""
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
    """计算日尺度ET0（Penman-Monteith），单位mm/day"""
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


def calculate_cwdi(daily_data, weights, kc_map=None):
    """CWDI计算：ET0->ETc(kc*ET0)，与P前移一天；50日窗口按5×10天聚合加权"""
    df = daily_data.copy()
    if 'P' not in df.columns and 'precip' in df.columns:
        df = df.rename(columns={'precip': 'P'})  # 降水列标准化为P

    if 'ET0' not in df.columns:
        lat_deg = float(df['lat'].iloc[0])
        elev_m = float(df['altitude'].iloc[0])
        df['ET0'] = penman_et0(df, lat_deg, elev_m)

    kc_series = pd.Series(0.0, index=df.index)
    m = df.index.month

    # kc系数（未提供则用默认月表）
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
        # 50日窗口 -> 5个10天块
        etc_blocks = etc_vals.reshape(5, 10)
        p_blocks = p_window.reshape(5, 10)
        etc_sum = etc_blocks.sum(axis=1)
        p_sum = p_blocks.sum(axis=1)
        # 仅在ETc>0且ETc>=P时计算干旱强度
        cond = (etc_sum > 0) & (etc_sum >= p_sum)
        cwdi_blocks = np.zeros(5, dtype=float)
        cwdi_blocks[cond] = (1 - p_sum[cond] / etc_sum[cond]) * 100.0
        return float(np.dot(w, cwdi_blocks))

    df['CWDI'] = etc_shift.rolling(window=50).apply(_cwdi_window, raw=False)  # 逐日滚动计算CWDI

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
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])  # LZW压缩单波段

        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])

        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)

        band.FlushCache()
        dataset = None

    def drought_station_g(self, data, excel_path):    
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

        def _read_growth_kc_from_excel(path):
            gp_df = pd.read_excel(path, sheet_name='发育阶段')
            kc_df = pd.read_excel(path, sheet_name='作物系数')

            s_col = '开始日期'
            e_col = '结束日期'
            w_col = '权重'

            def parse_cn_date(s):  # 识别“上年”为-1偏移，仅用split解析中文日期
                t = str(s)
                off = -1 if ('上年' in t) else 0
                t = t.replace('上年', '').replace('当年', '')
                parts = t.split('月')
                if len(parts) < 2:
                    return None, off
                mm_str = parts[0].strip()
                rest = parts[1]
                dd_parts = rest.split('日')
                if len(dd_parts) < 1:
                    return None, off
                dd_str = dd_parts[0].strip()
                try:
                    mm = int(mm_str)
                    dd = int(dd_str)
                except:
                    return None, off
                return f"{mm:02d}-{dd:02d}", off

            gp_defs = []  # 生育期定义列表（来自Excel）
            for _, r in gp_df.iterrows():
                s_txt = r[s_col] if s_col in gp_df.columns else None
                e_txt = r[e_col] if e_col in gp_df.columns else None
                if pd.isna(s_txt) or pd.isna(e_txt):
                    continue
                s_md, s_off = parse_cn_date(s_txt)
                e_md, e_off = parse_cn_date(e_txt)
                if not s_md or not e_md:
                    continue
                wv = float(r[w_col]) if w_col in gp_df.columns and not pd.isna(r[w_col]) else 0.0
                gp_defs.append({'start': s_md, 'end': e_md, 'start_offset': s_off, 'end_offset': e_off, 'weight': wv})
            
            # kc系数字典（来自Excel月表）
            m_col = '月份'
            k_col = 'kc'
            kc_map = {}
            
            if m_col and k_col:
                for _, r in kc_df.iterrows():
                    if not pd.isna(r[m_col]) and not pd.isna(r[k_col]):
                        kc_map[int(r[m_col])] = float(r[k_col])
            return gp_defs, kc_map

        gp_defs, kc_excel = _read_growth_kc_from_excel(excel_path)
        df = calculate_cwdi(  # 用月kc计算ETc并得到CWDI序列
            data,
            [0.3, 0.25, 0.2, 0.15, 0.1],
            kc_excel)

        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        years = sorted(series.index.year.unique())
        if not years:
            return np.nan

        # 自定义传参
        CWDI = []  # 分段加权后的年度CWDI
        for y in years:
            if gp_defs != []:
                ranges = []  # 按偏移拼接跨年时间段
                for r in gp_defs:
                    s = str(r.get("start", "01-01"))
                    e = str(r.get("end", "12-31"))
                    s_m, s_d = map(int, s.split("-"))
                    e_m, e_d = map(int, e.split("-"))
                    s_off = int(r.get("start_offset", 0))
                    e_off = int(r.get("end_offset", 0))
                    ranges.append((pd.Timestamp(y + s_off, s_m, s_d), pd.Timestamp(y + e_off, e_m, e_d)))

                # 提取对应权重
                w_vals = [r.get("weight") for r in gp_defs]
                w = np.array(w_vals, dtype=float)

            # 河南报告P49
            else:
                ranges = [(pd.Timestamp(y - 1, 8, 1), pd.Timestamp(y - 1, 10, 10)), (pd.Timestamp(y - 1, 10, 11), pd.Timestamp(y - 1, 12, 20)),
                          (pd.Timestamp(y - 1, 12, 21), pd.Timestamp(y, 2, 20)), (pd.Timestamp(y, 2, 21), pd.Timestamp(y, 3, 31)),
                          (pd.Timestamp(y, 4, 1), pd.Timestamp(y, 4, 30)), (pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 20)),
                          (pd.Timestamp(y, 5, 21), pd.Timestamp(y, 6, 10))]
                w = np.array([0.09, 0.13, 0.11, 0.12, 0.20, 0.22, 0.13], dtype=float)

            k_cwdi = []  # 各段CWDI均值
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
        station_coords = params.get('station_coords')  # 站点坐标字典
        algorithm_config = params.get('algorithmConfig')
        cfg = params.get('config')
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')
        excel_path = cfg.get("growthPeriodPath")

        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())

        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')
        station_values: Dict[str, float] = {}

        def _compute(sid):
            daily = dm.load_station_data(sid, start_date, end_date)
            g = self.drought_station_g(daily, excel_path)
            return sid, g

        with ThreadPoolExecutor(max_workers=4) as ex:
            for sid, g in ex.map(_compute, station_ids):
                station_values[sid] = float(g) if np.isfinite(g) else np.nan

        interp_conf = algorithm_config.get('interpolation')
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params')

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

        if method == 'lsm_idw':  # 插值到格网
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])
        result['data'] = normalize_array(result['data'])  # 归一化
        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        meta = result['meta']
        self._save_geotiff_gdal(result['data'], meta, g_tif_path, 0)  # 保存中间结果tif

        class_conf = algorithm_config.get('classification')
        if class_conf:  # 可选分级
            algos = params.get('algorithms')
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)

        return {  # 返回栅格数据与空间元信息
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
