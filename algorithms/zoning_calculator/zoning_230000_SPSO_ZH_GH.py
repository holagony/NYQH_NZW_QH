import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal
from pathlib import Path


def normalize_array(array: np.ndarray) -> np.ndarray:
    """
    归一化数组到0-1范围
    """
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


def _sat_vapor_pressure(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))  # 饱和水汽压(kPa),T为气温(°C)


def _slope_delta(T):
    es = _sat_vapor_pressure(T)
    return 4098.0 * es / ((T + 237.3)**2)  # 饱和水汽压曲线斜率(kPa/°C)


def _pressure_from_elevation(z):
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0)**5.26  # 海拔高度z(m)处的大气压(kPa)


def _psychrometric_constant(P):
    return 0.000665 * P  # 湿度常数γ(kPa/°C)


def _solar_geometry(lat_rad, day_of_year):
    # 太阳几何与地外辐射：返回Ra(地外辐射)、N(日照时数极限)、ωs(日落时角)
    dr = 1.0 + 0.033 * np.cos(2.0 * pi / 365.0 * day_of_year)
    delta = 0.409 * np.sin(2.0 * pi / 365.0 * day_of_year - 1.39)
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    Ra = (24.0 * 60.0 / pi) * 0.0820 * dr * (omega_s * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s))
    N = 24.0 / pi * omega_s
    return Ra, N, omega_s


def penman_et0(daily_data, lat_deg, elev_m, albedo=0.23, as_coeff=0.25, bs_coeff=0.5, k_rs=0.16):
    df = daily_data.copy()
    tmax = df['tmax']
    tmin = df['tmin']
    tmean = df['tavg'] if 'tavg' in df.columns else (tmax + tmin) / 2.0

    phi = np.deg2rad(lat_deg)
    J = df.index.dayofyear
    Ra, N, omega_s = _solar_geometry(phi, J)

    if 'sunshine' in df.columns:
        n = df['sunshine']
        Rs = (as_coeff + bs_coeff * (n / N)) * Ra  # 实测日照时数估算入射短波辐射
    else:
        Rs = k_rs * np.sqrt(np.maximum(tmax - tmin, 0.0)) * Ra  # 无日照时数时用温差估算方法

    Rso = (0.75 + 2e-5 * elev_m) * Ra  # 晴空辐射
    Rns = (1.0 - albedo) * Rs

    es_tmax = _sat_vapor_pressure(tmax)
    es_tmin = _sat_vapor_pressure(tmin)
    es = (es_tmax + es_tmin) / 2.0  # 平均饱和水汽压(kPa)
    ea = es * (df['rhum'] / 100.0) if 'rhum' in df.columns else es * 0.7  # 缺湿度时经验系数

    sigma = 4.903e-9
    tmaxK = tmax + 273.16
    tminK = tmin + 273.16
    # 净长波辐射,含湿度与云量校正
    Rnl = sigma * (
        (tmaxK**4 + tminK**4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
    Rn = Rns - Rnl

    P = _pressure_from_elevation(elev_m)
    gamma = _psychrometric_constant(P)
    delta = _slope_delta(tmean)
    u2 = df['wind'] if 'wind' in df.columns else pd.Series(2.0, index=df.index)

    # Penman-Monteith 主公式
    et0 = (0.408 * delta * (Rn) + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
    return et0.clip(lower=0)


def calculate_cwdi(daily_data, weights, lat_deg=None, elev_m=None):
    df = daily_data.copy()
    if 'P' not in df.columns and 'precip' in df.columns:
        df = df.rename(columns={'precip': 'P'})

    if 'ET0' not in df.columns:
        if lat_deg is None and 'lat' in df.columns:
            lat_deg = float(df['lat'].iloc[0])
        if elev_m is None and 'altitude' in df.columns:
            elev_m = float(df['altitude'].iloc[0])
        df['ET0'] = penman_et0(df, lat_deg, elev_m)

    years = df.index.year.unique()
    kc_series = pd.Series(0.0, index=df.index)
    for y in years:
        kc_series[(df.index >= pd.Timestamp(y, 4, 21)) & (df.index <= pd.Timestamp(y, 5, 20))] = 0.45
        kc_series[(df.index >= pd.Timestamp(y, 5, 21)) & (df.index <= pd.Timestamp(y, 6, 10))] = 0.6
        kc_series[(df.index >= pd.Timestamp(y, 6, 11)) & (df.index <= pd.Timestamp(y, 7, 10))] = 0.9
        kc_series[(df.index >= pd.Timestamp(y, 7, 11)) & (df.index <= pd.Timestamp(y, 8, 10))] = 1.32
        kc_series[(df.index >= pd.Timestamp(y, 8, 11)) & (df.index <= pd.Timestamp(y, 8, 31))] = 1.2
        kc_series[(df.index >= pd.Timestamp(y, 9, 1)) & (df.index <= pd.Timestamp(y, 10, 10))] = 0.7

    df['ETc'] = kc_series * df['ET0']
    etc_shift = df['ETc'].shift(1)
    p_shift = df['P'].shift(1)
    w = np.array([weights[4], weights[3], weights[2], weights[1], weights[0]], dtype=float)

    def _cwdi_window(etc_window):
        '''
        滑窗计算CWDI
        '''
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


class SPSO_ZH:
    '''
    黑龙江-大豆-灾害区划-大豆干旱
    '''
    def _get_algorithm(self, algorithm_name):
        """从算法注册器获取实例（插值/分类等）"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

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
        计算每个站点的干旱风险性G
        '''
        # 计算 CWDI 日序列（滑窗 50 天，分 5 个 10 天段，根据权重合成），用于后续干旱判识
        df = calculate_cwdi(data, config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("lat_deg"), config.get("elev_m"))

        # 提取 CWDI 并按给定生育期（日界）进行逐年筛选
        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        start_date_str = config.get("start_date")  # 形如 "MM-DD"
        end_date_str = config.get("end_date")  # 形如 "MM-DD"
        year_offset = int(config.get("year_offset", 0))  # 跨年发育期支持：结束日所属年份 = 年份 + year_offset
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

        # 计算相对干旱指数 CWDIa：按年去均值并缩放到近似 0~1 的量纲
        cwdi_mean = series.groupby(series.index.year).transform('mean')
        cwdi_a = (series - cwdi_mean) / (100 - cwdi_mean)

        # 取参与统计的年份；若超过 30 年，仅保留最后 30 年以符合多年统计口径
        years = sorted(cwdi_a.index.year.unique())
        if len(years) > 30:
            years = years[-30:]

        # 阈值设定：依据 GB/T 32136-2015，CWDIa > 0.4 判为干旱；CWDI0=0.4
        cwdi0 = 0.4
        stotals = []  # 年度总强度 Stotal（发育期内所有事件的面积之和）
        durations = []  # 年度总历时 Dtotal（发育期内所有事件的持续天数之和）
        counts = []  # 年度干旱过程数（事件数量）

        for y in years:
            # 提取该年的生育期内逐日 CWDIa
            sub = cwdi_a[cwdi_a.index.year == y]
            if sub.empty:
                continue
            v = sub.values.astype(float)
            # 若单位为百分数（>1），缩放到 0~1
            if np.nanmax(v) > 1.0:
                v = v / 100.0

            # 事件识别：将 v>cwdi0 的连续区段视为一次干旱事件，记录其开始(ts)与结束(ta)
            a = (v > cwdi0).astype(np.int8)
            d = np.diff(np.r_[0, a, 0])
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]

            Dtotal = 0.0  # 干旱历时和（天）
            Stotal = 0.0  # 干旱强度和（面积：∑(v - cwdi0)）
            event_count = 0

            for s0, e0 in zip(starts, ends):
                Di = float(e0 - s0)
                Si = float(np.sum(v[s0:e0] - cwdi0))
                # 只累积“有效干旱事件”。即：干旱事件持续时间 > 0 且干旱强度 > 0
                if Di > 0 and Si > 0:
                    Dtotal += Di
                    Stotal += Si
                    event_count += 1

            stotals.append(Stotal)
            durations.append(Dtotal)
            counts.append(event_count)

        if not stotals:
            return np.nan

        # 先按年求和，再对多年求平均
        return float(np.sum(stotals) / len(stotals))

    def calculate_drought(self, params):
        '''
        干旱区划
        '''
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cwdi_config = algorithm_config.get('cwdi', {})
        cfg = params.get('config', {})
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

        # 输出插值前站点数值范围
        vals = [v for v in station_values.values() if not np.isnan(v)]
        if vals:
            data_min = float(np.min(vals))
            data_max = float(np.max(vals))
            print(f"插值前站点数值范围: {data_min:.4f} ~ {data_max:.4f}")

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

        # 数值设置 + tiff保存
        result_norm = normalize_array(result['data'])  # 归一化
        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱危险性指数_归一化.tif")
        self._save_geotiff_gdal(result_norm, result['meta'], g_tif_path, 0)

        # 致灾因子危险性
        ZZ_array = result_norm

        # 脆弱度
        CR_percent_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/黑龙江承灾体/干旱/Z_AGME_C_BEHB_1991010120201231_P_ACDIS_ACRS_ADT-FRAIL-CVUY19912020-G_100M_CHN_L88_PD_000_00.tif"
        CR_temp_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/CR_temp.tif"
        CR_temp_path = LSMIDWInterpolation()._align_datasets(interp_data["grid_path"], CR_percent_path, CR_temp_path)
        in_ds_CR = gdal.Open(CR_temp_path)
        CR_array = in_ds_CR.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_CR.GetRasterBand(1).GetNoDataValue()
        CR_array = np.where(CR_array == Nodata, np.nan, CR_array)

        # 暴露度
        BL_percent_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/黑龙江承灾体/干旱/Z_AGME_C_BEHB_1991010120201231_P_ACDIS_ACRS_ADT-SENS_100M_CHN_L88_PD_000_00.tif"
        BL_temp_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/BL_temp.tif"
        BL_temp_path = LSMIDWInterpolation()._align_datasets(interp_data["grid_path"], BL_percent_path, BL_temp_path)
        in_ds_BL = gdal.Open(BL_temp_path)
        BL_array = in_ds_BL.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_BL.GetRasterBand(1).GetNoDataValue()
        BL_array = np.where(BL_array == Nodata, np.nan, BL_array)

        # 防灾减灾
        FZJZ_percent_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/黑龙江承灾体/干旱/Z_AGME_C_BEHB_1991010120201231_P_ACDIS_ACRS_ADT-PREV_100M_CHN_L88_PD_000_00.tif"
        FZJZ_temp_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/FZJZ_temp.tif"
        FZJZ_temp_path = LSMIDWInterpolation()._align_datasets(interp_data["grid_path"], FZJZ_percent_path, FZJZ_temp_path)
        in_ds_FZJZ = gdal.Open(FZJZ_temp_path)
        FZJZ_array = in_ds_FZJZ.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_FZJZ.GetRasterBand(1).GetNoDataValue()
        FZJZ_array = np.where(FZJZ_array == Nodata, np.nan, FZJZ_array)


        w_hazard, w_vulnerability, w_exposure, w_prevention = 0.7, 0.1, 0.1, 0.1
        risk = (
            np.nan_to_num(ZZ_array, nan=0.0).astype(np.float32) * w_hazard +
            np.nan_to_num(CR_array, nan=0.0).astype(np.float32) * w_vulnerability +
            np.nan_to_num(BL_array, nan=0.0).astype(np.float32) * w_exposure +
            (1.0 - np.nan_to_num(FZJZ_array, nan=0.0).astype(np.float32)) * w_prevention
        )
        risk[np.isnan(ZZ_array)] = np.nan

        print(f"综合风险指数数值范围: {np.nanmin(risk):.4f} ~ {np.nanmax(risk):.4f}")

        risk_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        self._save_geotiff_gdal(risk.astype(np.float32), result['meta'], risk_tif_path, 0)

        data_out = risk

        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            classifier = self._get_algorithm(f"classification.{method}")
            data_out = classifier.execute(risk, class_conf)
            class_tif = os.path.join(cfg.get("resultPath"), "干旱综合风险指数_分级.tif")
            self._save_geotiff_gdal(data_out.astype(np.int16), result['meta'], class_tif, 0)
        
        return {
            'data': data_out,
            'meta': {
                'width': result['meta']['width'],
                'height': result['meta']['height'],
                'transform': result['meta']['transform'],
                'crs': result['meta']['crs']
            }
        }

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']

        if disaster_type == 'GH':
            return self.calculate_drought(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
