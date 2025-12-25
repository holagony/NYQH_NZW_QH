import os
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal
from math import pi
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

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

def calculate_station_drought_risk(daily_data, config):
    """计算单个站点的干旱风险指数 - 使用DataManager加载的数据"""
    df = daily_data.copy()
    
    # 确保有必要的列
    if df.empty or len(df) < 100:
        return np.nan
    
    # 获取配置参数
    cwdi_weights = config.get("cwdi_weights", [0.3, 0.25, 0.2, 0.15, 0.1])
    start_date_str = config.get("start_date")
    end_date_str = config.get("end_date")
    threshold_config = config.get("threshold", [])
    
    if not threshold_config:
        raise ValueError("未配置干旱等级阈值")
    
    try:
        # 计算CWDI
        df_result = calculate_cwdi(df, cwdi_weights, df['kc'])
        series = df_result["CWDI"] if "CWDI" in df_result.columns else pd.Series(dtype=float)
        
        # 时间范围筛选
        if start_date_str and end_date_str and not series.empty:
            # 判断是否跨年
            start_date = pd.to_datetime(f"2000-{start_date_str}")
            end_date = pd.to_datetime(f"2000-{end_date_str}")
            
            if end_date < start_date:
                year_offset = 1
            else:
                year_offset = 0
            
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
        
        # 获取年份数
        years = sorted(series.index.year.unique())
        n = len(years) if years else 0
        
        if n == 0:
            return np.nan
        
        # 初始化各等级统计
        drought_days = {}
        
        # 处理每个干旱等级
        for level_config in threshold_config:
            level = level_config.get("level")
            label = level_config.get("label", f"等级{level}")
            min_val = level_config.get("min")
            max_val = level_config.get("max", "")
            weight = float(level_config.get("weight", 0))
            
            # 构建条件
            if max_val == "" or max_val is None:
                # 最大值留空，表示≥最小值
                condition = (series >= min_val)
            else:
                condition = ((series >= min_val) & (series < max_val))
            
            # 统计该等级的天数
            days_count = condition.sum()
            drought_days[level] = {
                'label': label,
                'days': days_count,
                'weight': weight,
                'annual_avg': days_count / n if n > 0 else 0
            }
        
        # 计算干旱风险指数
        drought_risk = 0.0
        for level, data in drought_days.items():
            drought_risk += data['weight'] * data['annual_avg']
        
        return float(drought_risk)
        
    except Exception as e:
        print(f"计算干旱风险失败: {e}")
        return np.nan
    
# def calculate_station_drought_risk(daily_data, config):
#     """计算单个站点的干旱风险指数 - 使用DataManager加载的数据"""
#     df = daily_data.copy()
    
#     # 确保有必要的列
#     if df.empty or len(df) < 100:
#         return np.nan
    
#     # 获取配置参数
#     cwdi_weights = config.get("cwdi_weights", [0.3, 0.25, 0.2, 0.15, 0.1])
#     start_date_str = config.get("start_date")
#     end_date_str = config.get("end_date")
#     threshold_config = config.get("threshold", [])
    
#     if not threshold_config:
#         raise ValueError("未配置干旱等级阈值")
    
#     try:
#         # 计算CWDI
#         df_result = calculate_cwdi(df, cwdi_weights, df['kc'])
#         series = df_result["CWDI"] if "CWDI" in df_result.columns else pd.Series(dtype=float)
        
#         # 时间范围筛选
#         if start_date_str and end_date_str and not series.empty:
#             # 判断是否跨年
#             start_date = pd.to_datetime(f"2000-{start_date_str}")
#             end_date = pd.to_datetime(f"2000-{end_date_str}")
            
#             if end_date < start_date:
#                 year_offset = 1
#             else:
#                 year_offset = 0
            
#             years = series.index.year.unique()
#             masks = []
#             for year in years:
#                 start_dt = pd.to_datetime(f"{year}-{start_date_str}")
#                 end_dt = pd.to_datetime(f"{year + year_offset}-{end_date_str}")
#                 masks.append((series.index >= start_dt) & (series.index <= end_dt))
            
#             if masks:
#                 mask = masks[0]
#                 for m in masks[1:]:
#                     mask = mask | m
#                 series = series[mask]
        
#         if series.empty:
#             return np.nan
        
#         # 获取年份数
#         years = sorted(series.index.year.unique())
#         n = len(years) if years else 0
        
#         if n == 0:
#             return np.nan
        
#         # 初始化各等级统计
#         drought_days = {}
        
#         # 处理每个干旱等级
#         for level_config in threshold_config:
#             level = level_config.get("level")
#             label = level_config.get("label", f"等级{level}")
#             min_val = level_config.get("min")
#             max_val = level_config.get("max", "")
#             weight = float(level_config.get("weight", 0))
            
#             # 构建条件
#             if max_val == "" or max_val is None:
#                 # 最大值留空，表示≥最小值
#                 condition = (series >= min_val)
#             else:
#                 condition = ((series >= min_val) & (series < max_val))
            
#             # 统计该等级的天数
#             days_count = condition.sum()
#             drought_days[level] = {
#                 'label': label,
#                 'days': days_count,
#                 'weight': weight,
#                 'annual_avg': days_count / n if n > 0 else 0
#             }
        
#         # 计算干旱风险指数
#         drought_risk = 0.0
#         for level, data in drought_days.items():
#             drought_risk += data['weight'] * data['annual_avg']
        
#         return float(drought_risk)
        
#     except Exception as e:
#         print(f"计算干旱风险失败: {e}")
#         return np.nan

# class ZH_GH:
#     '''
#     干旱通用算法
#     '''

#     def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
#         """保存GeoTIFF文件"""

#         # 根据输入数据的 dtype 确定 GDAL 数据类型
#         if data.dtype == np.uint8:
#             datatype = gdal.GDT_Byte
#         elif data.dtype == np.uint16:
#             datatype = gdal.GDT_UInt16
#         elif data.dtype == np.int16:
#             datatype = gdal.GDT_Int16
#         elif data.dtype == np.uint32:
#             datatype = gdal.GDT_UInt32
#         elif data.dtype == np.int32:
#             datatype = gdal.GDT_Int32
#         elif data.dtype == np.float32:
#             datatype = gdal.GDT_Float32
#         elif data.dtype == np.float64:
#             datatype = gdal.GDT_Float64
#         else:
#             datatype = gdal.GDT_Float32  # 默认情况

#         driver = gdal.GetDriverByName('GTiff')
#         dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])  # LZW压缩单波段

#         dataset.SetGeoTransform(meta['transform'])
#         dataset.SetProjection(meta['crs'])

#         band = dataset.GetRasterBand(1)
#         band.WriteArray(data)
#         band.SetNoDataValue(nodata)

#         band.FlushCache()
#         dataset = None

#     def drought_station_g(self, data, excel_path, config=None):    
#         '''
#         计算每个站点的干旱风险性
        
#         自定义传参说明：
#         start : 段起始日期， "MM-DD" 格式
#         end : 段结束日期， "MM-DD" 格式
#         start_offset : 起始日期相对基准年的偏移（整数，默认 0 ）
#         end_offset : 结束日期相对基准年的偏移（整数，默认 0 ）
#         基准年为当前遍历的 y ，支持跨年段（例如上一年用 -1 ，当年用 0 ）
#         weight: 对应权重

#         格式如下：
#         {
#         "growingPeriod": [
#             {"start": "08-01", "end": "10-10", "start_offset": -1, "end_offset": -1, "weight": 0.09},
#             {"start": "10-11", "end": "12-20", "start_offset": -1, "end_offset": -1, "weight": 0.13},
#             {"start": "12-21", "end": "02-20", "start_offset": -1, "end_offset": 0,  "weight": 0.11}
#             ]
#         }

#         kc系数：
#           "kcMap": {
#                         "10": 0.67, "11": 0.70, "12": 0.74,
#                         "1": 0.64, "2": 0.64, "3": 0.90,
#                         "4": 1.22, "5": 1.13, "6": 0.83
#                     }
#         '''

#         def _read_growth_kc_from_excel(path):
#             gp_df = pd.read_excel(path, sheet_name='发育阶段')
#             kc_df = pd.read_excel(path, sheet_name='作物系数')

#             s_col = '开始日期'
#             e_col = '结束日期'
#             w_col = '权重'

#             def parse_cn_date(s):  # 识别“上年”为-1偏移，仅用split解析中文日期
#                 t = str(s)
#                 off = -1 if ('上年' in t) else 0
#                 t = t.replace('上年', '').replace('当年', '')
#                 parts = t.split('月')
#                 if len(parts) < 2:
#                     return None, off
#                 mm_str = parts[0].strip()
#                 rest = parts[1]
#                 dd_parts = rest.split('日')
#                 if len(dd_parts) < 1:
#                     return None, off
#                 dd_str = dd_parts[0].strip()
#                 try:
#                     mm = int(mm_str)
#                     dd = int(dd_str)
#                 except:
#                     return None, off
#                 return f"{mm:02d}-{dd:02d}", off

#             gp_defs = []  # 生育期定义列表（来自Excel）
#             for _, r in gp_df.iterrows():
#                 s_txt = r[s_col] if s_col in gp_df.columns else None
#                 e_txt = r[e_col] if e_col in gp_df.columns else None
#                 if pd.isna(s_txt) or pd.isna(e_txt):
#                     continue
#                 s_md, s_off = parse_cn_date(s_txt)
#                 e_md, e_off = parse_cn_date(e_txt)
#                 if not s_md or not e_md:
#                     continue
#                 wv = float(r[w_col]) if w_col in gp_df.columns and not pd.isna(r[w_col]) else 0.0
#                 gp_defs.append({'start': s_md, 'end': e_md, 'start_offset': s_off, 'end_offset': e_off, 'weight': wv})
            
#             # kc系数字典（来自Excel月表）
#             m_col = '月份'
#             k_col = 'kc'
#             kc_map = {}
            
#             if m_col and k_col:
#                 for _, r in kc_df.iterrows():
#                     if not pd.isna(r[m_col]) and not pd.isna(r[k_col]):
#                         kc_map[int(r[m_col])] = float(r[k_col])
#             return gp_defs, kc_map

#         gp_defs, kc_excel = _read_growth_kc_from_excel(excel_path)
#         df = calculate_cwdi(  # 用月kc计算ETc并得到CWDI序列
#             data,
#             [0.3, 0.25, 0.2, 0.15, 0.1],
#             kc_excel)

#         series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        
#         # 增加对数据的时间范围筛选
#         start_date_str = config.get("start_date")
#         end_date_str = config.get("end_date")

#         # 检查跨年情况（保持原有逻辑）
#         start_date = pd.to_datetime(f"2000-{start_date_str}")
#         end_date = pd.to_datetime(f"2000-{end_date_str}")
#         if end_date < start_date:
#             year_offset = 1
#         else:
#             year_offset = 0

#         if start_date_str and end_date_str and not series.empty:
#             years = series.index.year.unique()
#             masks = []
#             for year in years:
#                 start_dt = pd.to_datetime(f"{year}-{start_date_str}")
#                 end_dt = pd.to_datetime(f"{year + year_offset}-{end_date_str}")
#                 masks.append((series.index >= start_dt) & (series.index <= end_dt))
#             if masks:
#                 mask = masks[0]
#                 for m in masks[1:]:
#                     mask = mask | m
#                 series = series[mask]

#         years = sorted(series.index.year.unique())
#         if not years:
#             return np.nan

#         # 自定义传参
#         CWDI = []  # 分段加权后的年度CWDI
#         for y in years:
#             if gp_defs != []:
#                 ranges = []  # 按偏移拼接跨年时间段
#                 for r in gp_defs:
#                     s = str(r.get("start", "01-01"))
#                     e = str(r.get("end", "12-31"))
#                     s_m, s_d = map(int, s.split("-"))
#                     e_m, e_d = map(int, e.split("-"))
#                     s_off = int(r.get("start_offset", 0))
#                     e_off = int(r.get("end_offset", 0))
#                     ranges.append((pd.Timestamp(y + s_off, s_m, s_d), pd.Timestamp(y + e_off, e_m, e_d)))

#                 # 提取对应权重
#                 w_vals = [r.get("weight") for r in gp_defs]
#                 w = np.array(w_vals, dtype=float)

#             # 河南报告P49
#             else:
#                 ranges = [(pd.Timestamp(y - 1, 8, 1), pd.Timestamp(y - 1, 10, 10)), (pd.Timestamp(y - 1, 10, 11), pd.Timestamp(y - 1, 12, 20)),
#                           (pd.Timestamp(y - 1, 12, 21), pd.Timestamp(y, 2, 20)), (pd.Timestamp(y, 2, 21), pd.Timestamp(y, 3, 31)),
#                           (pd.Timestamp(y, 4, 1), pd.Timestamp(y, 4, 30)), (pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 20)),
#                           (pd.Timestamp(y, 5, 21), pd.Timestamp(y, 6, 10))]
#                 w = np.array([0.09, 0.13, 0.11, 0.12, 0.20, 0.22, 0.13], dtype=float)

#             k_cwdi = []  # 各段CWDI均值
#             for s, e in ranges:
#                 seg = series[(series.index >= s) & (series.index <= e)]
#                 k_cwdi.append(float(seg.mean()) if len(seg) > 0 else np.nan)

#             k_cwdi = np.array(k_cwdi, dtype=float)
#             CWDI.append(float(np.nansum(w * k_cwdi)))

#         risk = float(np.nanmean(CWDI))

#         return risk

#     def calculate_drought(self, params):
#         '''
#         干旱区划
#         '''
#         station_coords = params.get('station_coords')  # 站点坐标字典
#         algorithm_config = params.get('algorithmConfig')
#         cfg = params.get('config')
#         data_dir = cfg.get('inputFilePath')
#         station_file = cfg.get('stationFilePath')
#         excel_path = cfg.get("growthPeriodPath")

#         dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
#         station_ids = list(station_coords.keys())
#         if not station_ids:
#             station_ids = dm.get_all_stations()
#         available = set(dm.get_all_stations())

#         station_ids = [sid for sid in station_ids if sid in available]
#         start_date = cfg.get('startDate')
#         end_date = cfg.get('endDate')
#         station_values: Dict[str, float] = {}

#         # 单线程循环计算
#         for sid in station_ids:
#             daily = dm.load_station_data(sid, start_date, end_date)
#             g = self.drought_station_g(daily, excel_path, algorithm_config)
#             station_values[sid] = float(g) if np.isfinite(g) else np.nan

#         algorithm_config = params.get('algorithmConfig', {})
#         interp_conf = algorithm_config.get('interpolation', {})
#         method = str(interp_conf.get('method', 'idw')).lower()
#         iparams = interp_conf.get('params') or {}

#         if 'var_name' not in iparams:
#             iparams['var_name'] = 'value'

#         interp_data = {
#             'station_values': station_values,
#             'station_coords': station_coords,
#             'grid_path': cfg.get('gridFilePath'),
#             'dem_path': cfg.get('demFilePath'),
#             'area_code': cfg.get('areaCode'),
#             'shp_path': cfg.get('shpFilePath')
#         }

#         if method == 'lsm_idw':  # 插值到格网
#             result = LSMIDWInterpolation().execute(interp_data, iparams)
#         else:
#             result = IDWInterpolation().execute(interp_data, iparams)

#         # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])
#         result['data'] = normalize_array(result['data'])  # 归一化
#         g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
#         meta = result['meta']
#         self._save_geotiff_gdal(result['data'], meta, g_tif_path, 0)  # 保存中间结果tif

#         class_conf = algorithm_config.get('classification')
#         if class_conf:  # 可选分级
#             algos = params.get('algorithms')
#             key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
#             if key in algos:
#                 result['data'] = algos[key].execute(result['data'], class_conf)

#         return {  # 返回栅格数据与空间元信息
#             'data': result['data'],
#             'meta': {
#                 'width': result['meta']['width'],
#                 'height': result['meta']['height'],
#                 'transform': result['meta']['transform'],
#                 'crs': result['meta']['crs']
#             }
#         }

#     def calculate(self, params):
#         self._algorithms = params['algorithms']
#         return self.calculate_drought(params)


# from concurrent.futures import ProcessPoolExecutor


class KCDataLoader:
    """KC数据加载器 - 从Excel读取生育期数据，生成逐日KC序列"""
    
    @staticmethod
    def load_kc_data(excel_path: str, start_year: int = 1980, end_year: int = 2023) -> pd.Series:
        """
        从Excel读取生育期数据，生成逐日KC序列
        
        参数:
            excel_path: Excel文件路径，包含'生育期' sheet
            start_year: 起始年份
            end_year: 结束年份
            
        返回:
            Series逐日KC序列
        """
        try:
            kc_df = pd.read_excel(excel_path, sheet_name='干旱区划模板')
            
            required_cols = ['生育期', '开始日期', '结束日期', 'kc']
            for col in required_cols:
                if col not in kc_df.columns:
                    raise ValueError(f"Excel中缺少必要列: {col}")
            
            kc_df = KCDataLoader._convert_excel_dates_to_string(kc_df)
            
            all_dates = []
            all_kc_values = []
            
            # 处理每个生育期
            for _, row in kc_df.iterrows():
                start_str = str(row['开始日期'])
                end_str = str(row['结束日期'])
                kc_value = float(row['kc'])
                
                # 解析开始日期
                start_date, start_year_offset = KCDataLoader._parse_date_string(start_str)
                # 解析结束日期
                end_date, end_year_offset = KCDataLoader._parse_date_string(end_str)
                
                # 为指定年份范围生成该生育期的逐日KC
                for year in range(start_year, end_year + 1):
                    start_actual = pd.Timestamp(
                        year=year + start_year_offset,
                        month=start_date.month,
                        day=start_date.day
                    )
                    
                    end_actual = pd.Timestamp(
                        year=year + end_year_offset,
                        month=end_date.month,
                        day=end_date.day
                    )
                    
                    # 处理跨年情况
                    if end_actual < start_actual:
                        end_actual = pd.Timestamp(
                            year=year + end_year_offset + 1,
                            month=end_date.month,
                            day=end_date.day
                        )
                    
                    # 生成日期范围
                    date_range = pd.date_range(start=start_actual, end=end_actual, freq='D')
                    
                    # 添加到结果
                    all_dates.extend(date_range)
                    all_kc_values.extend([kc_value] * len(date_range))
            
            # 创建Series
            if all_dates:
                kc_series = pd.Series(all_kc_values, index=all_dates)
                # 去除重复日期（保留最后一个）
                kc_series = kc_series[~kc_series.index.duplicated(keep='last')]
                # 按日期排序
                kc_series = kc_series.sort_index()
                
                return kc_series
            else:
                raise ValueError("未生成任何KC数据")
                
        except Exception as e:
            print(f"读取KC Excel文件失败: {e}")
            raise
 
    @staticmethod
    def _convert_excel_dates_to_string(df):
        """将Excel中的数字日期转换为中文日期字符串"""
        # 检查并转换开始日期列
        if '开始日期' in df.columns:
            df['开始日期'] = df['开始日期'].apply(KCDataLoader._excel_number_to_chinese_date)
        
        # 检查并转换结束日期列
        if '结束日期' in df.columns:
            df['结束日期'] = df['结束日期'].apply(KCDataLoader._excel_number_to_chinese_date)
        
        return df

    @staticmethod
    def _excel_number_to_chinese_date(excel_number):
        """将Excel日期数字转换为中文日期字符串"""
        try:
            # 如果已经是字符串，直接返回
            if isinstance(excel_number, str):
                return excel_number
            
            # 如果是数字，转换为日期
            if isinstance(excel_number, (int, float)):
                # Excel日期从1900-01-01开始
                base_date = pd.Timestamp('1899-12-30')
                date = base_date + pd.Timedelta(days=excel_number)
                
                # 转换为中文格式 "M月D日"
                month = date.month
                day = date.day
                return f"{month}月{day}日"
            
            # 其他情况返回原值
            return excel_number
        except Exception as e:
            print(f"转换Excel日期失败: {excel_number}, 错误: {e}")
            return excel_number
    
    @staticmethod
    def _parse_date_string(date_str: str) -> Tuple[pd.Timestamp, int]:
        """解析日期字符串，返回日期对象和年份偏移量"""
        date_str = str(date_str).strip()
        
        year_offset = 0
        if '上年' in date_str:
            year_offset = -1
            date_str = date_str.replace('上年', '')
        elif '当年' in date_str:
            year_offset = 0
            date_str = date_str.replace('当年', '')
        
        try:
            if '月' in date_str and '日' in date_str:
                month_part, day_part = date_str.split('月')
                month = int(month_part.strip())
                day = int(day_part.replace('日', '').strip())
                date_obj = pd.Timestamp(year=2000, month=month, day=day)
                return date_obj, year_offset
            else:
                raise ValueError(f"无法解析日期格式: {date_str}")
        except Exception as e:
            raise ValueError(f"解析日期失败: {date_str}, 错误: {e}")


class StationProcessor:
    """站点处理器 - 负责单个站点的计算"""
    
    def __init__(self, station_id: str, csv_path: str, kc_series: pd.Series, config: Dict = None):
        self.station_id = station_id
        self.csv_path = csv_path
        self.kc_series = kc_series
        self.config = config or {}
        
        self.field_mapping = {
            "Datetime": "date",
            "TEM_Avg": "tavg",
            "TEM_Max": "tmax",
            "TEM_Min": "tmin",
            "SSH": "sunshine",
            "RHU_Avg": "rhum",
            "PRE_Time_2020": "precip",
            "WIN_S_2mi_Avg": "wind",
            "Station_Id_C": "station_id",
            "Lat": "lat",
            "Lon": "lon",
            "Alti": "altitude"
        }
    
    def preprocess_data(self, data: pd.Series) -> pd.Series:
        """数据预处理"""
        data = data.replace([
            "999999.0", "999990.0", "999.0", "999999", 
            "-999999", -999999, 999999, 999990, 999998, 
            999, "999", np.nan, None
        ], np.nan)
        
        data = pd.to_numeric(data, errors='coerce')
        
        if data.size > 0:
            mask_9600 = (data > 999600) & (data < 999700)
            mask_9700 = (data > 999700) & (data < 999800)
            mask_9800 = (data > 999800) & (data < 999900)
            
            data = np.where(mask_9600, data - 999600, data)
            data = np.where(mask_9700, data - 999700, data)
            data = np.where(mask_9800, data - 999800, data)
        
        return pd.Series(data)
    
    def load_and_process_station_data(self) -> pd.DataFrame:
        """加载和处理站点数据"""
        try:
            for encoding in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    data = pd.read_csv(self.csv_path, dtype=str, encoding=encoding, low_memory=False)
                    break
                except:
                    continue
            else:
                return pd.DataFrame()
            
        except Exception as e:
            return pd.DataFrame()
        
        data = data.rename(columns=self.field_mapping)
        
        if len(data) < 2:
            return pd.DataFrame()
        
        stn_info = data.iloc[0]
        data = data.iloc[1:].reset_index(drop=True)
        
        try:
            data['lat'] = float(stn_info.get('lat', 0))
            data['lon'] = float(stn_info.get('lon', 0))
            data['altitude'] = float(stn_info.get('altitude', 0))
        except:
            data['lat'] = 0.0
            data['lon'] = 0.0
            data['altitude'] = 0.0
        
        try:
            data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
        except:
            try:
                data['date'] = pd.to_datetime(data['date'])
            except:
                return pd.DataFrame()
        
        data = data.set_index('date')
        
        numeric_columns = ['tavg', 'tmax', 'tmin', 'sunshine', 'rhum', 'precip', 'wind']
        
        for col in numeric_columns:
            if col in data.columns:
                data[col] = self.preprocess_data(data[col])
        
        for col in ['tmax', 'tmin', 'precip']:
            if col not in data.columns:
                data[col] = np.nan
        
        if 'precip' in data.columns:
            data['P'] = data['precip']
        
        return data
    
    def calculate_drought_risk(self) -> float:
        """计算单个站点的干旱风险指数 - 新参数格式"""
        # 加载数据
        df = self.load_and_process_station_data()
        
        if df.empty or len(df) < 100:
            return np.nan
        
        # 获取配置参数
        cwdi_weights = self.config.get("cwdi_weights", [0.3, 0.25, 0.2, 0.15, 0.1])
        start_date_str = self.config.get("start_date")
        end_date_str = self.config.get("end_date")
        year_offset = int(self.config.get("year_offset", 0))
        threshold_config = self.config.get("threshold", [])
        
        if not threshold_config:
            raise ValueError("未配置干旱等级阈值")
        
        # 计算CWDI
        df_result = calculate_cwdi(df, cwdi_weights, self.kc_series)
        series = df_result["CWDI"] if "CWDI" in df_result.columns else pd.Series(dtype=float)
        
        # 时间范围筛选
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
        
        # 获取年份数
        years = sorted(series.index.year.unique())
        n = len(years) if years else 0
        
        if n == 0:
            return np.nan
        
        # 初始化各等级统计
        drought_days = {}
        
        # 处理每个干旱等级
        for level_config in threshold_config:
            level = level_config.get("level")
            label = level_config.get("label", f"等级{level}")
            min_val = level_config.get("min")
            max_val = level_config.get("max", "")
            weight = float(level_config.get("weight", 0))
            
            # 构建条件
            if max_val == "" or max_val is None:
                # 最大值留空，表示≥最小值
                condition = (series >= min_val)
            else:
                condition = ((series >= min_val) & (series < max_val))
            
            # 统计该等级的天数
            days_count = condition.sum()
            drought_days[level] = {
                'label': label,
                'days': days_count,
                'weight': weight,
                'annual_avg': days_count / n if n > 0 else 0
            }
        
        # 计算干旱风险指数
        drought_risk = 0.0
        for level, data in drought_days.items():
            drought_risk += data['weight'] * data['annual_avg']
        
        return float(drought_risk)

def process_single_station(args: Tuple) -> Tuple[str, float]:
    """处理单个站点的包装函数，用于多进程"""
    station_id, csv_path, kc_series, config = args
    processor = StationProcessor(station_id, csv_path, kc_series, config)
    risk = processor.calculate_drought_risk()
    return station_id, risk



class ZH_GH:
    '''优化后的干旱通用算法'''

    def __init__(self):
        self.kc_series = None
        self.dm = None  # DataManager实例

    def calculate_station_drought_risk(self, station_id, start_date, end_date, config):
        """计算单个站点的干旱风险指数 - 在类中可以直接使用dm"""
        # 加载站点数据
        try:
            daily = self.dm.load_station_data(station_id, start_date, end_date)
            
            if daily.empty or len(daily) < 100:
                return np.nan
            
            # 添加KC数据
            daily['kc'] = self.kc_series.reindex(daily.index).fillna(0)
            
            # 确保有必要的列
            if 'P' not in daily.columns and 'precip' in daily.columns:
                daily = daily.rename(columns={'precip': 'P'})
            
            # 获取配置参数
            cwdi_weights = config.get("cwdi_weights", [0.3, 0.25, 0.2, 0.15, 0.1])
            start_date_str = config.get("start_date")
            end_date_str = config.get("end_date")
            threshold_config = config.get("threshold", [])
            
            if not threshold_config:
                raise ValueError("未配置干旱等级阈值")
            
            # 计算CWDI
            df_result = self._calculate_cwdi(daily, cwdi_weights)
            series = df_result["CWDI"] if "CWDI" in df_result.columns else pd.Series(dtype=float)
            
            # 时间范围筛选
            if start_date_str and end_date_str and not series.empty:
                # 判断是否跨年
                start_date_dt = pd.to_datetime(f"2000-{start_date_str}")
                end_date_dt = pd.to_datetime(f"2000-{end_date_str}")
                
                if end_date_dt < start_date_dt:
                    year_offset = 1
                else:
                    year_offset = 0
                
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
            
            # 获取年份数
            years = sorted(series.index.year.unique())
            n = len(years) if years else 0
            
            if n == 0:
                return np.nan
            
            # 初始化各等级统计
            drought_days = {}
            
            # 处理每个干旱等级
            for level_config in threshold_config:
                level = level_config.get("level")
                label = level_config.get("label", f"等级{level}")
                min_val = level_config.get("min")
                max_val = level_config.get("max", "")
                weight = float(level_config.get("weight", 0))
                
                # 构建条件
                if max_val == "" or max_val is None:
                    # 最大值留空，表示≥最小值
                    condition = (series >= min_val)
                else:
                    condition = ((series >= min_val) & (series < max_val))
                
                # 统计该等级的天数
                days_count = condition.sum()
                drought_days[level] = {
                    'label': label,
                    'days': days_count,
                    'weight': weight,
                    'annual_avg': days_count / n if n > 0 else 0
                }
            
            # 计算干旱风险指数
            drought_risk = 0.0
            for level, data in drought_days.items():
                drought_risk += data['weight'] * data['annual_avg']
            
            return float(drought_risk)
            
        except Exception as e:
            print(f"站点 {station_id} 计算干旱风险失败: {e}")
            return np.nan
    
    def _calculate_cwdi(self, daily_data, cwdi_weights):
        """计算CWDI - 向量化版本"""
        df = daily_data.copy()
        
        # 确保有经纬度和高程信息
        if 'lat' not in df.columns or 'altitude' not in df.columns:
            raise ValueError("缺少站点经纬度或高程信息")
        
        lat_deg = float(df['lat'].iloc[0])
        elev_m = float(df['altitude'].iloc[0])
        
        # 计算ET0
        df['ET0'] = penman_et0(df, lat_deg, elev_m)
        
        # 计算ETc
        df['ETc'] = df['kc'] * df['ET0']
        etc_shift = df['ETc'].shift(1).values
        p_shift = df['P'].shift(1).values
        
        # CWDI计算权重数组
        w = np.array([cwdi_weights[4], cwdi_weights[3], cwdi_weights[2], cwdi_weights[1], cwdi_weights[0]], dtype=float)
        
        # 向量化滚动窗口计算
        n = len(etc_shift)
        if n < 50:
            df['CWDI'] = np.nan
            return df
        
        cwdi_result = np.full(n, np.nan, dtype=float)
        
        for i in range(49, n):
            start_idx = i - 49
            end_idx = i
            
            etc_window = etc_shift[start_idx:end_idx+1]
            p_window = p_shift[start_idx:end_idx+1]
            
            # 重塑为5×10
            etc_blocks = etc_window.reshape(5, 10)
            p_blocks = p_window.reshape(5, 10)
            
            etc_sum = etc_blocks.sum(axis=1)
            p_sum = p_blocks.sum(axis=1)
            
            # 条件计算
            cond = (etc_sum > 0) & (etc_sum >= p_sum)
            cwdi_blocks = np.zeros(5, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(p_sum[cond], etc_sum[cond])
                cwdi_blocks[cond] = (1 - ratio) * 100.0
            
            cwdi_result[i] = np.dot(w, cwdi_blocks)
        
        df['CWDI'] = cwdi_result
        return df
    
    def calculate_drought(self, params):
        '''干旱区划计算 - 使用多进程'''
        cfg = params.get('config', {})
        algorithm_config = params.get('algorithmConfig', {})
        
        data_dir = cfg.get('inputFilePath')
        excel_path = cfg.get("growthPeriodPath")
        station_file = cfg.get('stationFilePath')
        
        # 从配置读取开始和结束时间
        start_date = cfg.get('startDate')  # YYYYMMDD格式
        end_date = cfg.get('endDate')      # YYYYMMDD格式
        
        # 计算年份范围
        start_year = 1980  # 默认起始年份
        end_year = 2023    # 默认结束年份
        
        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date, format='%Y%m%d')
                end_dt = pd.to_datetime(end_date, format='%Y%m%d')
                
                # 获取年份，考虑到生育期可能有"上年"的情况，适当扩展年份范围
                start_year = start_dt.year - 1  # 减1年，考虑可能的"上年"生育期
                end_year = end_dt.year     # 加1年，考虑跨年情况
                
                print(f"根据数据时间范围设置KC序列年份: {start_year}-{end_year}")
            except Exception as e:
                print(f"解析日期失败，使用默认年份范围: {e}")
        
        # 一次性加载KC数据，生成逐日KC序列
        try:
            self.kc_series = KCDataLoader.load_kc_data(excel_path, start_year, end_year)
            print(f"KC数据加载完成: {len(self.kc_series)}天的KC序列")
            print(f"KC序列时间范围: {self.kc_series.index.min()} 到 {self.kc_series.index.max()}")
        except Exception as e:
            print(f"加载KC数据失败: {e}")
            raise
        
        # 初始化DataManager
        print("初始化DataManager...")
        self.dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        
        # 获取所有站点
        all_stations = self.dm.get_all_stations()
        print(f"DataManager中找到 {len(all_stations)} 个站点")
        
        # 站点筛选：如果有传入的站点坐标，只处理这些站点
        station_coords = params.get('station_coords', {})
        if station_coords:
            # 获取需要处理的站点ID
            target_stations = set(station_coords.keys())
            # 筛选站点，只保留目标站点
            station_ids = [sid for sid in all_stations if sid in target_stations]
            
            # 检查是否有目标站点没有对应的数据
            missing_stations = target_stations - set(station_ids)
            if missing_stations:
                print(f"警告: {len(missing_stations)} 个目标站点没有找到对应的数据: {list(missing_stations)[:10]}{'...' if len(missing_stations) > 10 else ''}")
        else:
            station_ids = all_stations
        
        if not station_ids:
            print("筛选后没有需要处理的站点")
            return {'data': np.array([]), 'meta': {}}
        
        print(f"开始处理 {len(station_ids)} 个站点")
        
        # 准备多进程参数
        # 由于方法在类中，我们需要使用类实例作为参数
        process_args = []
        for station_id in station_ids:
            # 注意：这里不能直接传递self，因为多进程不能pickle类实例
            # 我们需要使用不同的策略
            process_args.append((
                station_id, start_date, end_date, self.kc_series.copy(), algorithm_config, data_dir, station_file
            ))
        
        # 使用多进程计算
        station_values = {}
        num_workers = min(cpu_count(), len(station_ids))
        
        print(f"开始多进程计算，使用 {num_workers} 个进程")
        
        # 由于类方法不能直接用于多进程，我们使用一个包装函数
        with Pool(processes=num_workers) as pool:
            results = list(pool.imap_unordered(self._process_station_wrapper, process_args))
        
        # 整理结果
        for station_id, risk in results:
            if np.isfinite(risk):
                station_values[station_id] = float(risk)
            else:
                station_values[station_id] = np.nan
        
        valid_count = sum(1 for v in station_values.values() if np.isfinite(v))
        print(f"站点计算完成，有效结果数: {valid_count}/{len(station_values)}")
        
        # 插值到格网
        print("开始插值计算...")
        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params') or {}
        
        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'
        
        # 准备插值数据 - 使用传入的站点坐标或从CSV中读取的坐标
        # 注意：这里我们需要从CSV文件中读取站点坐标，或者使用传入的station_coords
        # 在实际应用中，可能需要一个站点坐标文件
        interp_data = {
            'station_values': station_values,
            'station_coords': station_coords,  # 使用传入的站点坐标
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }
        
        # 如果传入的站点坐标不全，尝试从CSV文件中读取缺失的坐标
        if station_coords:
            missing_coords = set(station_values.keys()) - set(station_coords.keys())
            if missing_coords:
                print(f"警告: {len(missing_coords)} 个站点没有坐标信息，将尝试从CSV文件中读取")
                # 这里可以添加从CSV读取坐标的逻辑
        
        # 执行插值
        try:
            if method == 'lsm_idw':
                from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
                result = LSMIDWInterpolation().execute(interp_data, iparams)
            else:
                from algorithms.interpolation.idw import IDWInterpolation
                result = IDWInterpolation().execute(interp_data, iparams)
            
            # 归一化
            def normalize_array(array: np.ndarray) -> np.ndarray:
                if array.size == 0:
                    return array
                
                mask = ~np.isnan(array)
                if not np.any(mask):
                    return np.zeros_like(array)
                
                valid_values = array[mask]
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                
                if max_val == min_val:
                    normalized_array = np.full_like(array, 0.5, dtype=float)
                    normalized_array[~mask] = np.nan
                else:
                    normalized_array = (array - min_val) / (max_val - min_val)
                    normalized_array[~mask] = np.nan
                
                return normalized_array
            
            result['data'] = normalize_array(result['data'])
            
            # 保存中间结果
            result_path = cfg.get("resultPath", ".")
            g_tif_path = Path(result_path) / "intermediate" / "干旱综合风险指数.tif"
            g_tif_path.parent.mkdir(parents=True, exist_ok=True)
            
            if 'meta' in result:
                self._save_geotiff_gdal(result['data'], result['meta'], str(g_tif_path), 0)
                print(f"结果已保存到: {g_tif_path}")
            
            # 可选分级
            class_conf = algorithm_config.get('classification')
            if class_conf:
                algos = params.get('algorithms', {})
                key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
                if key in algos:
                    result['data'] = algos[key].execute(result['data'], class_conf)
            
            # 返回结果
            if 'meta' in result:
                return {
                    'data': result['data'],
                    'meta': {
                        'width': result['meta'].get('width', 0),
                        'height': result['meta'].get('height', 0),
                        'transform': result['meta'].get('transform', (0, 1, 0, 0, 0, 1)),
                        'crs': result['meta'].get('crs', '')
                    }
                }
            else:
                return {'data': result.get('data', np.array([])), 'meta': {}}
                
        except ImportError as e:
            print(f"导入插值模块失败: {e}")
            return {'data': np.array(list(station_values.values())), 'meta': {}}
        
        
        # return {'data': np.array(list(station_values.values())), 'meta': {}}
    
    def _process_station_wrapper(self, args):
        """处理单个站点的包装函数，用于多进程"""
        # 解包参数
        station_id, start_date, end_date, kc_series, algorithm_config, data_dir, station_file = args
        
        # 由于多进程中不能共享实例状态，我们需要在每个进程中创建新的对象
        # 创建DataManager实例
        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        
        try:
            # 加载站点数据
            daily = dm.load_station_data(station_id, start_date, end_date)
            
            if daily.empty or len(daily) < 100:
                return station_id, np.nan
            
            # 添加KC数据
            daily['kc'] = kc_series.reindex(daily.index).fillna(0)
            
            # 确保有必要的列
            if 'P' not in daily.columns and 'precip' in daily.columns:
                daily = daily.rename(columns={'precip': 'P'})
            
            # 获取配置参数
            cwdi_weights = algorithm_config.get("cwdi_weights", [0.3, 0.25, 0.2, 0.15, 0.1])
            start_date_str = algorithm_config.get("start_date")
            end_date_str = algorithm_config.get("end_date")
            threshold_config = algorithm_config.get("threshold", [])
            
            if not threshold_config:
                return station_id, np.nan
            
            # 计算CWDI
            df_result = self._calculate_cwdi_in_process(daily, cwdi_weights)
            series = df_result["CWDI"] if "CWDI" in df_result.columns else pd.Series(dtype=float)
            
            # 时间范围筛选
            if start_date_str and end_date_str and not series.empty:
                # 判断是否跨年
                start_date_dt = pd.to_datetime(f"2000-{start_date_str}")
                end_date_dt = pd.to_datetime(f"2000-{end_date_str}")
                
                if end_date_dt < start_date_dt:
                    year_offset = 1
                else:
                    year_offset = 0
                
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
                return station_id, np.nan
            
            # 获取年份数
            years = sorted(series.index.year.unique())
            n = len(years) if years else 0
            
            if n == 0:
                return station_id, np.nan
            
            # 初始化各等级统计
            drought_days = {}
            
            # 处理每个干旱等级
            for level_config in threshold_config:
                level = level_config.get("level")
                label = level_config.get("label", f"等级{level}")
                min_val = level_config.get("min")
                max_val = level_config.get("max", "")
                weight = float(level_config.get("weight", 0))
                
                # 构建条件
                if max_val == "" or max_val is None:
                    # 最大值留空，表示≥最小值
                    condition = (series >= min_val)
                else:
                    condition = ((series >= min_val) & (series < max_val))
                
                # 统计该等级的天数
                days_count = condition.sum()
                drought_days[level] = {
                    'label': label,
                    'days': days_count,
                    'weight': weight,
                    'annual_avg': days_count / n if n > 0 else 0
                }
            
            # 计算干旱风险指数
            drought_risk = 0.0
            for level, data in drought_days.items():
                drought_risk += data['weight'] * data['annual_avg']
            
            return station_id, float(drought_risk)
            
        except Exception as e:
            print(f"站点 {station_id} 处理失败: {e}")
            return station_id, np.nan
    
    def _calculate_cwdi_in_process(self, daily_data, cwdi_weights):
        """在多进程中计算CWDI的辅助函数"""
        return self._calculate_cwdi(daily_data, cwdi_weights)
    
    def calculate(self, params):
        """主计算接口"""
        return self.calculate_drought(params)


    
    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        """保存GeoTIFF文件"""
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
    

# 配置验证函数
def validate_drought_config(config: Dict) -> bool:
    """验证干旱配置格式是否正确"""
    required_fields = ['start_date', 'end_date', 'threshold']
    
    for field in required_fields:
        if field not in config:
            print(f"缺少必要字段: {field}")
            return False
    
    # 验证threshold配置
    threshold_config = config.get('threshold', [])
    if not isinstance(threshold_config, list):
        print("threshold必须是列表")
        return False
    
    total_weight = 0.0
    for i, level_config in enumerate(threshold_config):
        if not isinstance(level_config, dict):
            print(f"threshold[{i}]必须是字典")
            return False
        
        required_level_fields = ['min', 'level', 'label', 'weight']
        for field in required_level_fields:
            if field not in level_config:
                print(f"threshold[{i}]缺少字段: {field}")
                return False
        
        try:
            weight = float(level_config['weight'])
            total_weight += weight
        except:
            print(f"threshold[{i}]的weight必须是数值")
            return False
    
    # 验证权重和为1
    if abs(total_weight - 1.0) > 0.001:
        print(f"各等级权重之和应为1.0，当前为{total_weight}")
        return False
    
    return True

# 示例调用代码
if __name__ == "__main__":
    # 示例配置
    drought_config = {
        "start_date": "05-01",
        "end_date": "08-31",
        "year_offset": 0,
        "cwdi_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
        "threshold": [
            {"min": 35, "max": 45, "level": 1, "label": "轻旱", "weight": 0.15},
            {"min": 45, "max": 55, "level": 2, "label": "中旱", "weight": 0.35},
            {"min": 55, "max": "", "level": 3, "label": "重旱", "weight": 0.5}
        ]
    }
    
    # 验证配置
    if validate_drought_config(drought_config):
        print("配置验证通过")
        
        # 创建算法实例
        zh_gh = ZH_GH()
        
        # 准备参数
        params = {
            'config': {
                'inputFilePath': '/path/to/data',
                'growthPeriodPath': '/path/to/生育期.xlsx',
                'resultPath': '/path/to/results'
            },
            'algorithmConfig': drought_config
        }
        
        # 执行计算
        result = zh_gh.calculate(params)
        print("计算完成")