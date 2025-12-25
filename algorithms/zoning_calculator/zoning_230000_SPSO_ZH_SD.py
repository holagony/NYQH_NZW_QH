# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 15:12:07 2025

@author: HTHT
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal
from scipy.ndimage import sobel
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
    Rnl = sigma * ((tmaxK**4 + tminK**4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
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
    黑龙江-大豆-灾害区划
    大豆干旱
    大豆冷害
    大豆霜冻 TODO
    大豆渍涝 TODO
    '''

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _interpolate_risk(self, data, station_coords, config, crop_config, type):
        interpolation = crop_config.get("interpolation")
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")

        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        print(f"使用 {interpolation_method} 方法对综合风险指数进行插值")

        # 准备插值数据
        interpolation_data = {'station_values': data, 'station_coords': station_coords, 'dem_path': config.get("demFilePath", ""), 'shp_path': config.get("shpFilePath", ""), 'grid_path': config.get("gridFilePath", ""), 'area_code': config.get("areaCode", "")}

        # 执行插值
        try:
            interpolated_result = interpolator.execute(interpolation_data, interpolation_params)
            print(f"{type}指数插值完成")
            # 保存中间结果
            self._save_intermediate_result(interpolated_result, config, type)

            return interpolated_result

        except Exception as e:
            print(f"{type}指数插值失败: {str(e)}")
            raise

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], indicator_name: str) -> None:
        """保存中间结果 - 各个指标的插值结果"""
        try:
            print(f"保存中间结果: {indicator_name}")

            # 生成中间结果文件名
            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            # 使用与最终结果相同的保存逻辑
            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                print(f"警告: 中间结果 {indicator_name} 格式不支持,跳过保存")
                return
            meta["nodata"] = -32768
            # 保存为GeoTIFF
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")
            # 不抛出异常,继续处理其他指标

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
        """保存GeoTIFF文件"""
        from osgeo import gdal

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

    def _numpy_to_gdal_dtype(self, numpy_dtype: np.dtype) -> int:
        """将numpy数据类型转换为GDAL数据类型"""
        from osgeo import gdal

        dtype_map = {
            np.bool_: gdal.GDT_Byte,
            np.uint8: gdal.GDT_Byte,
            np.uint16: gdal.GDT_UInt16,
            np.int16: gdal.GDT_Int16,
            np.uint32: gdal.GDT_UInt32,
            np.int32: gdal.GDT_Int32,
            np.float32: gdal.GDT_Float32,
            np.float64: gdal.GDT_Float64,
            np.complex64: gdal.GDT_CFloat32,
            np.complex128: gdal.GDT_CFloat64
        }

        for np_type, gdal_type in dtype_map.items():
            if np.issubdtype(numpy_dtype, np_type):
                return gdal_type

        # 默认使用Float32
        print(f"警告: 无法映射numpy数据类型 {numpy_dtype}，默认使用GDT_Float32")
        return gdal.GDT_Float32

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

    def calculate_ZL_HazardRisk(self, station_indicators, params):
        ZL = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            D50 = indicators.get('D50', np.nan)  # 总降水量
            D100 = indicators.get('D100', np.nan)  # 总日照时数
            D250 = indicators.get('D250', np.nan)  # 降水日数

            # 计算单站点致灾因子危险性指数
            result = 0.25 * D50 + 0.3 * D100 + 0.5 * D250
            ZL[station_id] = result

        filename = f'黑龙江大豆致灾因子危险性指数.csv'
        result_df = pd.DataFrame(list(ZL.items()), columns=['站点ID', '致灾因子危险性指数'])
        intermediate_dir = Path(params["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'黑龙江大豆致灾因子危险性指数站点级数据已保存至：{output_path}')
        return ZL

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

        interp_data = {'station_values': station_values, 'station_coords': station_coords, 'grid_path': cfg.get('gridFilePath'), 'dem_path': cfg.get('demFilePath'), 'area_code': cfg.get('areaCode'), 'shp_path': cfg.get('shpFilePath')}

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 数值设置 + tiff保存
        result['data'] = normalize_array(result['data'])  # 归一化
        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱危险性指数.tif")
        self._save_geotiff_gdal(result['data'], result['meta'], g_tif_path, 0)

        # 分级
        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            algos = params.get('algorithms', {})
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)

        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}}

    def _calculate_frost(self, params):
        '''
        计算霜冻指数
        '''
        pass

    def _calculate_DWLH(self, params):
        '''
        计算大豆冷害指数
        '''
        # 读取输入参数：坐标、算法与通用配置，并按干旱流程取数
        station_coords = params['station_coords']
        algorithmConfig = params.get('algorithmConfig', {})
        config = params['config']

        data_dir = config.get('inputFilePath')
        station_file = config.get('stationFilePath')
        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)

        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]

        start_date = params.get('startDate') or config.get('startDate')
        end_date = params.get('endDate') or config.get('endDate')
        start_year_cfg = int(str(start_date)[:4])
        end_year_cfg = int(str(end_date)[:4])

        min_target_year = start_year_cfg
        max_target_year = end_year_cfg

        # 数据加载时间范围与基准期选择规则：
        # - 若目标年份 ≤1991，加载1961–1990并采用固定基准期；
        # - 否则加载最近30年并采用滑动30年窗口。
        if min_target_year is not None and max_target_year is not None:
            if min_target_year <= 1991:
                load_start_date = int("19610101")
                end_y = max(1990, max_target_year)
                load_end_date = int(f"{end_y}1231")
            else:
                load_start_date = int(f"{min_target_year - 30}0101")
                load_end_date = int(f"{max_target_year}1231")
        else:
            load_start_date = None
            load_end_date = None

        def _ensure_tavg(df):
            # 补全平均气温tavg：若缺失则用(tmax+tmin)/2
            if 'tavg' in df.columns:
                return df
            if 'tmax' in df.columns and 'tmin' in df.columns:
                df = df.copy()
                df['tavg'] = (df['tmax'] + df['tmin']) / 2.0
            return df

        def _sum_monthly_means(df, year):
            # 计算某年5–9月各月平均温度之和ΣT5–9
            sub = df[(df.index.year == year) & (df.index.month.isin([5, 6, 7, 8, 9]))]
            if sub.empty:
                return np.nan
            m = sub['tavg'].groupby(sub.index.month).mean()
            return float(m.sum())

        dt_values = {}
        result_dict = {}
        for sid in station_ids:
            result_dict[sid] = dict()

            # 获取每个站的扩充数据
            daily = dm.load_station_data(sid, load_start_date, load_end_date)
            if daily.empty:
                dt_values[sid] = np.nan
                continue
            daily = _ensure_tavg(daily)

            # 计算给定年份 5–9 月各月平均气温之和
            years = sorted(daily.index.year.unique())
            vals_by_year = {}
            for y in years:
                v = _sum_monthly_means(daily, y)
                if not np.isnan(v):
                    vals_by_year[y] = v
            if not vals_by_year:
                dt_values[sid] = np.nan
                continue

            # 计算气象站5-9月各月月平均温度之和的距平值
            all_years = sorted(vals_by_year.keys())

            # 只保留实际年份
            if start_year_cfg is not None and end_year_cfg is not None:
                target_years = [y for y in all_years if start_year_cfg <= y <= end_year_cfg]
            else:
                target_years = all_years
            if not target_years:
                dt_values[sid] = np.nan
                continue

            # all_years 为该站点具有 ΣTi 的全部年份集合
            # first_year_all/last_year_all 分别表示该站可用数据的最早/最晚年份
            # total_years_all 用于判断是否具备至少 30 年的历史以支撑固定或滑动基准期
            first_year_all = all_years[0]
            total_years_all = len(all_years)

            # 对每个目标年份y，选择其基准期并计算距平ΔT5–9
            for y in target_years:
                result_dict[sid][str(y)] = dict()

                # 基准期选择：不足30年退化为[first_year_all, y-1]；
                # 固定基准期(y≤1991)用1961–1990；滑动基准期(y≥1992)用[y-30, y-1]
                if total_years_all < 30:
                    win_start, win_end = first_year_all, y - 1
                else:
                    # 固定基准期：当 y ≤ 1991 时，使用 1961–1990 的 30 年作为统一基准
                    if y <= 1991:
                        win_start, win_end = 1961, 1990
                    # 滑动基准期：当 y ≥ 1992 时，采用 y-30 至 y-1 的 30 年窗口
                    else:
                        win_start, win_end = y - 30, y - 1

                # 基准期年份集合；若为空则无法计算该年的距平，直接跳过
                win_years = [yy for yy in all_years if win_start <= yy <= win_end]
                if not win_years:
                    continue

                # 计算基准均值与距平：baseline_avg=mean(ΣT5–9@基准期)，delta=ΣT5–9@当年 - baseline_avg
                baseline_avg = float(np.mean([vals_by_year[yy] for yy in win_years]))
                delta = float(vals_by_year[y] - baseline_avg)
                result_dict[sid][str(y)]['delta'] = delta
                result_dict[sid][str(y)]['baseline_avg'] = baseline_avg

            # 阈值判定冷害年：按ΣT5–9归属等级I–V，并用ΔT5–9阈值判定轻/中/重
            cold_years = []
            intensities = []
            for y_str, vals in result_dict[sid].items():
                if 'delta' not in vals or 'baseline_avg' not in vals:
                    continue
                s = float(vals['baseline_avg'])
                d = float(vals['delta'])
                level = None
                if s <= 80:
                    level = 'I'
                elif 80 < s <= 85:
                    level = 'II'
                elif 85 < s <= 90:
                    level = 'III'
                elif 90 < s <= 95:
                    level = 'IV'
                else:
                    level = 'V'

                is_cold = False
                if level == 'I':
                    if -2.0 <= d <= -1.8:
                        is_cold = True
                    elif -2.2 <= d < -2.0:
                        is_cold = True
                    elif d < -2.2:
                        is_cold = True
                elif level == 'II':
                    if -2.2 <= d <= -1.9:
                        is_cold = True
                    elif -2.5 <= d < -2.2:
                        is_cold = True
                    elif d < -2.5:
                        is_cold = True
                elif level == 'III':
                    if -2.3 <= d <= -1.9:
                        is_cold = True
                        severity = 'light'
                    elif -2.7 <= d < -2.3:
                        is_cold = True
                    elif d < -2.7:
                        is_cold = True
                elif level == 'IV':
                    if -2.4 <= d <= -2.0:
                        is_cold = True
                    elif -2.9 <= d < -2.4:
                        is_cold = True
                    elif d < -2.9:
                        is_cold = True
                else:
                    if -2.6 <= d <= -2.0:
                        is_cold = True
                    elif -3.1 <= d < -2.6:
                        is_cold = True
                    elif d < -3.1:
                        is_cold = True

                # 增加结果key和value
                vals['level'] = level
                vals['is_cold_year'] = bool(is_cold)
                if is_cold:
                    cold_years.append(y_str)
                    intensities.append(abs(d))  # d本身为负表示冷害年强度，取绝对值

            # 每站冷害年统计：总年数、冷害年数、平均强度与频率
            total_years = len([y for y in result_dict[sid].keys() if str(y).isdigit()])
            cold_count = len(cold_years)
            avg_intensity = float(np.mean(intensities)) if len(intensities) > 0 else 0  # 这一行注意，未来插值看看考不考虑0的
            frequency = float(cold_count / total_years) if cold_count > 0 else 0
            result_dict[sid]['_stats'] = {'cold_year_count': cold_count, 'avg_intensity': avg_intensity, 'frequency': frequency}

        # 计算危险性
        sids = []
        intens_arr = []
        freq_arr = []
        for sid, stats in result_dict.items():
            sids.append(sid)
            intens_arr.append(stats['_stats']['avg_intensity'])
            freq_arr.append(stats['_stats']['frequency'])

        # 跨站点归一化与危险性综合：min-max归一化后dangerous=0.25*强度+0.75*频率
        intens_arr = np.array(intens_arr, dtype=float)
        freq_arr = np.array(freq_arr, dtype=float)
        ai_min, ai_max = float(np.min(intens_arr)), float(np.max(intens_arr))
        fr_min, fr_max = float(np.min(freq_arr)), float(np.max(freq_arr))
        ai_norm = np.zeros_like(intens_arr)
        fr_norm = np.zeros_like(freq_arr)
        if ai_max > ai_min:
            ai_norm = (intens_arr - ai_min) / (ai_max - ai_min)
        if fr_max > fr_min:
            fr_norm = (freq_arr - fr_min) / (fr_max - fr_min)
        dangerous_vals = 0.25 * ai_norm + 0.75 * fr_norm

        # 站点与坐标匹配
        sid_to_idx = {sid: i for i, sid in enumerate(sids)}
        common_sids = [sid for sid in sids if sid in station_coords]
        dangerous_station = {sid: float(dangerous_vals[sid_to_idx[sid]]) for sid in common_sids}
        coords_used = {sid: station_coords[sid] for sid in common_sids}

        # 插值参数与数据准备
        interp_conf = algorithmConfig.get('interpolation')
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {'station_values': dangerous_station, 'station_coords': coords_used, 'grid_path': config.get('gridFilePath'), 'dem_path': config.get('demFilePath'), 'area_code': config.get('areaCode'), 'shp_path': config.get('shpFilePath')}

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 归一化栅格并保存中间结果tif
        # result['data'] = normalize_array(result['data'])
        g_tif_path = os.path.join(config.get("resultPath"), "intermediate", "低温冷害危险性指数.tif")
        self._save_geotiff_gdal(result['data'], result['meta'], g_tif_path, 0)

        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}, 'type': '黑龙江低温冷害'}

    def _calculate_ZL(self, params):
        '''
        计算大豆渍涝指数
        '''
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print("开始计算大豆渍涝气候风险区划 ")
        print("第一步：站点级别计算致灾因子危险性指数")
        ZL_HazardRisk = self.calculate_ZL_HazardRisk(station_indicators, config)

        print("第二步：对致灾因子危险性指数进行插值")
        interpolated_ZL_HazardRisk = self._interpolate_risk(ZL_HazardRisk, station_coords, config, algorithmConfig, 'ZL_HazardRisk')

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']

        if disaster_type == 'GH':
            return self.calculate_drought(params)
        elif disaster_type == 'SD':
            return self._calculate_frost(params)
        elif disaster_type == 'DWLH':
            return self._calculate_DWLH(params)
        elif disaster_type == 'ZL':
            return self._calculate_ZL(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
