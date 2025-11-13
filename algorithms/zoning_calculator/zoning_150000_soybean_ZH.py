import numpy as np
import pandas as pd
from typing import Dict, Any
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation


def _sat_vapor_pressure(T):
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))  # 饱和水汽压(kPa)，T为气温(°C)


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
    # 净长波辐射，含湿度与云量校正
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


def calculate_cwdi(daily_data, kc, weights, lat_deg=None, elev_m=None):
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
        kc_series[(df.index >= pd.Timestamp(y, 5, 12)) & (df.index <= pd.Timestamp(y, 5, 25))] = 0.32
        kc_series[(df.index >= pd.Timestamp(y, 5, 26)) & (df.index <= pd.Timestamp(y, 6, 23))] = 0.51
        kc_series[(df.index >= pd.Timestamp(y, 6, 24)) & (df.index <= pd.Timestamp(y, 7, 4))] = 0.69
        kc_series[(df.index >= pd.Timestamp(y, 7, 5)) & (df.index <= pd.Timestamp(y, 8, 25))] = 1.15
        kc_series[(df.index >= pd.Timestamp(y, 8, 26)) & (df.index <= pd.Timestamp(y, 9, 18))] = 0.5
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


class SoybeanDisasterZoning:
    '''
    内蒙古-大豆-灾害区划
    干旱和霜冻
    干旱--目前是生成危险性G的tif，其他风险区划的样例数据暂未提供
    霜冻 TODO
    '''

    def drought_station_g(self, data, config):
        '''
        计算每个站点的干旱风险性G
        '''
        df = calculate_cwdi(data, float(config.get("kc", 0.8)), config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("lat_deg"),
                            config.get("elev_m"))
        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        year_offset = int(config.get("year_offset", 0))
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
        n = len(years) if years else 0
        if n == 0:
            return np.nan
        light = ((series >= 35) & (series < 45)).sum()
        medium = ((series >= 45) & (series < 55)).sum()
        heavy = (series >= 55).sum()
        w_light = light / n
        w_medium = medium / n
        w_heavy = heavy / n
        return float(0.15 * w_light + 0.35 * w_medium + 0.5 * w_heavy)

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

        # 读取数据和计算
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

        # 分级
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
            },
            'type': '内蒙古大豆干旱'
        }

    def calculate_freeze(self, params):
        '''
        霜冻区划
        '''
        pass

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'drought':
            return self.calculate_drought(params)
        elif disaster_type == 'freeze':
            return self.calculate_freeze(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
