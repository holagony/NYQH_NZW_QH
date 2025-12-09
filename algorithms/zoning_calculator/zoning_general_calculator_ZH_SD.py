import numpy as np
import pandas as pd
from typing import Dict, Any
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from pathlib import Path
import importlib
import ast
import datetime
import os


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


def pre_category(x):
    """降雨量赋值标准"""
    if x < 80:
        return 0
    elif 80 <= x <= 110:
        return 1
    elif 110 < x <= 130:
        return 2
    elif x > 130:
        return 3
    else:
        return np.nan


def pre_days_category(x):
    """降雨量赋值标准"""
    if x < 8:
        return 0
    elif 8 <= x <= 10:
        return 1
    elif 10 < x <= 13:
        return 2
    elif x > 13:
        return 3
    else:
        return np.nan


def ssh_category(x):
    """日照时赋值标准"""
    if x > 120:
        return 0
    elif 110 <= x <= 120:
        return 1
    elif 95 <= x < 110:
        return 2
    elif x < 95:
        return 3
    else:
        return np.nan


def index(x):
    "年度指数划分"
    if x <= 2:
        return 0
    elif 3 <= x <= 4:
        return 1
    elif 5 <= x <= 6:
        return 2
    elif x > 6:
        return 3
    else:
        return np.nan


def calculate_tmin0(daily_data):
    """
    ≤0℃
    平均天数, 逐年天数
    """
    df = daily_data.copy()
    TMIN = df["tmin"]
    #统计4月1日到4月30日日最低气温小于0℃的天数
    april_cold_days_by_year = {}

    # 获取所有年份
    all_years = TMIN.index.year.unique()

    for year in all_years:
        # 筛选该年4月1日到4月30日的数据
        april_data = TMIN[(TMIN.index.year == year) & (TMIN.index.month == 4) & (TMIN.index.day >= 1) & (TMIN.index.day <= 30)]

        # 统计日最低气温小于0℃的天数
        cold_days_count = len(april_data[april_data < 0])
        april_cold_days_by_year[year] = cold_days_count

    # 计算统计值
    if april_cold_days_by_year:
        cold_days_values = list(april_cold_days_by_year.values())
        mean_cold_days = np.mean(cold_days_values)
        max_cold_days = np.max(cold_days_values)
        min_cold_days = np.min(cold_days_values)
    else:
        mean_cold_days = 0
        max_cold_days = 0
        min_cold_days = 0

    return mean_cold_days, cold_days_values


def calculate_tmin1(daily_data):
    """
    ≤-1.5℃
    平均天数, 逐年的天数
    """
    df = daily_data.copy()
    TMIN = df["tmin"]
    #统计4月1日到4月30日日最低气温小于0℃的天数
    april_cold_days_by_year = {}

    # 获取所有年份
    all_years = TMIN.index.year.unique()

    for year in all_years:
        # 筛选该年4月1日到4月30日的数据
        april_data = TMIN[(TMIN.index.year == year) & (TMIN.index.month == 4) & (TMIN.index.day >= 1) & (TMIN.index.day <= 30)]

        # 统计日最低气温小于0℃的天数
        cold_days_count = len(april_data[april_data < (-1.5)])
        april_cold_days_by_year[year] = cold_days_count

    # 计算统计值
    if april_cold_days_by_year:
        cold_days_values = list(april_cold_days_by_year.values())
        mean_cold_days = np.mean(cold_days_values)
        max_cold_days = np.max(cold_days_values)
        min_cold_days = np.min(cold_days_values)
    else:
        mean_cold_days = 0
        max_cold_days = 0
        min_cold_days = 0

    return mean_cold_days, cold_days_values


def calculate_frost_level_stats(daily_data, algorithm_config):
    """
    根据algorithm_config中的threshold配置和时间范围计算各霜冻等级的多年平均值和逐年天数

    时间范围从algorithm_config的start_date和end_date获取
    格式如: start_date="04-01", end_date="12-31"

    返回: (frost_means, frost_stats_by_year)
    frost_means: 字典，霜冻等级 -> 多年平均天数
    frost_stats_by_year: 字典，年份 -> 各级霜冻天数
    """
    df = daily_data.copy()
    TMIN = df["tmin"]

    # 从配置中获取阈值和时间范围
    thresholds = algorithm_config.get("threshold", [])
    if not thresholds:
        raise ValueError("algorithm_config中缺少threshold配置")

    # 获取时间范围
    start_date_str = algorithm_config.get("start_date", "01-01")
    end_date_str = algorithm_config.get("end_date", "12-31")

    # print(f"计算时间范围: {start_date_str} 到 {end_date_str}")

    # 解析开始和结束日期
    try:
        start_month, start_day = map(int, start_date_str.split('-'))
        end_month, end_day = map(int, end_date_str.split('-'))
    except ValueError as e:
        print(f"日期格式错误: start_date={start_date_str}, end_date={end_date_str}")
        raise ValueError(f"日期格式错误，应为MM-DD格式: {str(e)}")

    # 验证阈值配置
    valid_thresholds = []
    for thresh in thresholds:
        if "min" in thresh and "max" in thresh and "label" in thresh:
            if thresh["min"] < thresh["max"]:
                valid_thresholds.append(thresh)
            else:
                print(f"警告: 阈值配置错误，min({thresh['min']}) >= max({thresh['max']})")
        else:
            print(f"警告: 阈值配置缺少必要字段: {thresh}")

    if not valid_thresholds:
        raise ValueError("没有有效的阈值配置")

    # 获取所有年份
    all_years = sorted(TMIN.index.year.unique())

    # 初始化结果字典
    frost_stats_by_year = {}

    # 按年统计各级霜冻天数
    for year in all_years:
        try:
            # 构建该年份的时间范围
            start_date = pd.Timestamp(f"{year}-{start_month:02d}-{start_day:02d}")
            end_date = pd.Timestamp(f"{year}-{end_month:02d}-{end_day:02d}")

            # 处理跨年的情况（如start_date="12-01", end_date="02-28"）
            if start_date > end_date:
                # 假设结束日期是下一年
                end_date = pd.Timestamp(f"{year + 1}-{end_month:02d}-{end_day:02d}")

            # 筛选该时间段内的数据
            period_mask = (TMIN.index >= start_date) & (TMIN.index <= end_date)
            period_data = TMIN[period_mask]

            if period_data.empty:
                print(f"警告: {year}年时间段{start_date_str}到{end_date_str}内没有数据")
                # 继续处理，统计值为0
                period_data = pd.Series([], dtype=float)

            year_stats = {}
            total_days = 0

            # 统计各等级霜冻天数
            for thresh in valid_thresholds:
                label = thresh["label"]
                min_temp = thresh["min"]
                max_temp = thresh["max"]

                # 筛选该等级的温度
                frost_mask = (period_data >= min_temp) & (period_data < max_temp)
                frost_days = period_data[frost_mask]
                days_count = len(frost_days)

                year_stats[label] = days_count
                total_days += days_count

            # 存储该年的霜冻统计
            year_stats["total"] = total_days
            frost_stats_by_year[year] = year_stats

            # # 可选：打印详细信息
            # if year % 5 == 0:  # 每5年打印一次
            #     print(f"  {year}年: {start_date_str}到{end_date_str}, 总天数{len(period_data)}, 霜冻{total_days}天")

        except Exception as e:
            print(f"处理{year}年数据时出错: {str(e)}")
            # 创建空的统计记录
            frost_stats_by_year[year] = {thresh["label"]: 0 for thresh in valid_thresholds}
            frost_stats_by_year[year]["total"] = 0

    # 计算多年平均值
    frost_means = {}
    if frost_stats_by_year:
        # 为每个阈值计算平均值
        for thresh in valid_thresholds:
            label_name = thresh["label"]
            values = [stats.get(label_name, 0) for stats in frost_stats_by_year.values()]
            frost_means[label_name] = np.mean(values) if values else 0

        # 计算总霜冻天数平均值
        total_values = [stats.get("total", 0) for stats in frost_stats_by_year.values()]
        frost_means["total"] = np.mean(total_values) if total_values else 0

        # # 打印统计信息
        # print(f"\n时间范围 {start_date_str} 到 {end_date_str} 的霜冻统计:")
        # for thresh in valid_thresholds:
        #     label = thresh["label"]
        #     mean_days = frost_means[label]
        #     print(f"  {label}霜冻({thresh['min']}到{thresh['max']}°C): 平均{mean_days:.2f}天/年")
        # print(f"  总霜冻: 平均{frost_means['total']:.2f}天/年")

    return frost_means, frost_stats_by_year


def normalize_values(values, min_val, max_val):
    """
    归一化数值到0-1范围
    """
    if not values:
        return []

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_values:
        return [0.0] * len(values)

    # 如果所有值都相同，归一化到0.5
    if max_val == min_val:
        return [0.5 if v is not None and not np.isnan(v) else 0.0 for v in values]

    normalized = []
    for v in values:
        if v is None or np.isnan(v):
            normalized.append(0.0)
        else:
            norm_val = (v - min_val) / (max_val - min_val)
            normalized.append(norm_val)

    return normalized


class ZH_SD:
    '''
    河南-冬小麦-灾害区划
    干旱区划
    晚霜冻气候区划 TODO
    麦收区连阴雨气候区划 TODO
    干热风区划 TODO
    '''

    def _calculate_continuous_rain_indicators_station(self, station_indicators, params):
        """在站点级别计算连阴雨指标"""
        continuous_rain_indicators = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            Pre = indicators.get('Pre', np.nan)  # 总降水量
            SSH = indicators.get('SSH', np.nan)  # 总日照时数
            Pre_days = indicators.get('Pre_days', np.nan)  # 降水日数

            # str转字典
            Pre_df = pd.DataFrame.from_dict(Pre, orient='index')
            SSH_df = pd.DataFrame.from_dict(SSH, orient='index')
            Pre_days_df = pd.DataFrame.from_dict(Pre_days, orient='index')

            merged_df = pd.concat([Pre_df, SSH_df, Pre_days_df], axis=1)
            merged_df.columns = ['Pre', 'SSH', 'Pre_days']

            # 按标准赋值
            merged_df['Pre'] = merged_df['Pre'].apply(pre_category)
            merged_df['SSH'] = merged_df['SSH'].apply(ssh_category)
            merged_df['Pre_days'] = merged_df['Pre_days'].apply(pre_days_category)
            cleaned_df = merged_df.dropna()

            # 年度指数
            cleaned_df['年度指数'] = cleaned_df['Pre'] + cleaned_df['SSH'] + cleaned_df['Pre_days']

            # 连阴雨程度与
            cleaned_df['连阴雨程度'] = cleaned_df['年度指数'].apply(index)

            # 综合指数
            frequency = cleaned_df['连阴雨程度'].value_counts().sort_index()
            for level in [0, 1, 2, 3]:
                if level not in frequency:
                    frequency[level] = 0
            weighted_frequency = (0.5 * frequency.get(3, 0) + 0.3 * frequency.get(2, 0) + 0.2 * frequency.get(1, 0))
            continuous_rain_indicators[station_id] = weighted_frequency / len(cleaned_df)

        max_value = max(continuous_rain_indicators.values())
        max_keys = [key for key, value in continuous_rain_indicators.items() if value == max_value]
        min_value = min(continuous_rain_indicators.values())
        min_keys = [key for key, value in continuous_rain_indicators.items() if value == min_value]
        print(f'麦收区连阴雨气候区划:有效站点数据：{len(cleaned_df)}')
        print(f'麦收区连阴雨气候区划：单站最高综合指数：{max_keys}：{max_value}')
        print(f'麦收区连阴雨气候区划:单站最低综合指数：{min_keys}：{min_value}')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'连阴雨指标_{timestamp}.csv'
        result_df = pd.DataFrame(list(continuous_rain_indicators.items()), columns=['站点ID', '连阴雨综合指数'])
        intermediate_dir = Path(params["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"连阴雨指标综合指数文件已保存为 '{output_path}'")

        return continuous_rain_indicators

    def _interpolate_continuous_rain_risk(self, continuous_rain_risk_station, station_coords, config, crop_config):
        """对连阴雨综合风险指数进行插值"""
        interpolation = crop_config.get("interpolation")
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")

        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        print(f"使用 {interpolation_method} 方法对综合风险指数进行插值")

        # 准备插值数据
        interpolation_data = {'station_values': continuous_rain_risk_station, 'station_coords': station_coords, 'dem_path': config.get("demFilePath", ""), 'shp_path': config.get("shpFilePath", ""), 'grid_path': config.get("gridFilePath", ""), 'area_code': config.get("areaCode", "")}

        # 执行插值
        try:
            interpolated_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合风险指数插值完成")
            # 保存中间结果
            self._save_intermediate_result(interpolated_result, config, "continuous_rain_risk_interpolated")

            return interpolated_result

        except Exception as e:
            print(f"综合风险指数插值失败: {str(e)}")
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
                print(f"警告: 中间结果 {indicator_name} 格式不支持，跳过保存")
                return
            meta["nodata"] = -32768
            # 保存为GeoTIFF
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")
            # 不抛出异常，继续处理其他指标

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

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def drought_station_g(self, data, config):
        '''
        计算每个站点的干旱风险性G
        '''
        df = calculate_cwdi(data, config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("lat_deg"), config.get("elev_m"))

        # 根据输入参数mask数据
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
        if not years:
            return np.nan

        # 河南报告P49
        weights = np.array([0.09, 0.13, 0.11, 0.12, 0.20, 0.22, 0.13], dtype=float)
        vals = []
        for y in years:
            ranges = [(pd.Timestamp(y - 1, 8, 1), pd.Timestamp(y - 1, 10, 10)), (pd.Timestamp(y - 1, 10, 11), pd.Timestamp(y - 1, 12, 20)), (pd.Timestamp(y - 1, 12, 21), pd.Timestamp(y, 2, 20)), (pd.Timestamp(y, 2, 21), pd.Timestamp(y, 3, 31)), (pd.Timestamp(y, 4, 1), pd.Timestamp(y, 4, 30)),
                      (pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 20)), (pd.Timestamp(y, 5, 21), pd.Timestamp(y, 6, 10))]

            means = []
            for s, e in ranges:
                seg = series[(series.index >= s) & (series.index <= e)]
                means.append(float(seg.mean()) if len(seg) > 0 else np.nan)

            m = np.array(means, dtype=float)
            vals.append(float(np.nansum(weights * m)))

        risk = float(sum(vals)) / float(len(years))

        return risk

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

        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}, 'type': '河南冬小麦干旱'}

    def calculate_dry(self, params):
        '''
        干热风区划
        计算代码写这里
        '''
        pass

    def calculate_wet(self, params):
        """计算小麦连阴雨风险 - 先计算站点综合风险指数再插值"""
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print("开始计算小麦连阴雨风险 - 新流程：先计算站点综合风险指数")

        try:
            # 第一步：在站点级别计算连阴雨指标
            print("第一步：在站点级别计算连阴雨指标")
            continuous_rain_indicators = self._calculate_continuous_rain_indicators_station(station_indicators, config)

            # 第二步：对综合风险指数F进行插值
            print("第二步：对综合风险指数F进行插值")
            interpolated_risk = self._interpolate_continuous_rain_risk(continuous_rain_indicators, station_coords, config, algorithmConfig)

            # 第三步：对插值结果进行分类
            print("第四步：对插值结果进行分类")
            classification = algorithmConfig['classification']
            classification_method = classification.get('method', 'custom_thresholds')
            classifier = self._get_algorithm(f"classification.{classification_method}")

            classified_data = classifier.execute(interpolated_risk['data'], classification)
            # 准备最终结果
            result = {'data': classified_data, 'meta': interpolated_risk['meta'], 'type': 'continuous_rain_risk', 'process': 'station_level_calculation'}
            print("小麦连阴雨风险计算完成")

        except Exception as e:
            print(f"小麦连阴雨风险计算失败: {str(e)}")
            result = np.nan
        return result

    def _calculate_frost(self, params):
        """霜冻灾害风险指数模型"""
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
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

        # 第一步：收集所有站点的轻/中/重度值
        station_d0_values = []  # 轻霜冻多年平均值
        station_d1_values = []  # 中霜冻多年平均值
        station_d2_values = []  # 重霜冻多年平均值
        station_d0_list = []  # 轻霜冻逐年天数（所有站点合并）
        station_d1_list = []  # 中霜冻逐年天数（所有站点合并）
        station_d2_list = []  # 重霜冻逐年天数（所有站点合并）
        station_frost_stats = {}  # 存储每个站点的详细霜冻统计

        station_data_map = {}  # 存储站点数据以便后续使用

        for sid in station_ids:
            # 加载站点数据
            daily = dm.load_station_data(sid, start_date, end_date)
            station_data_map[sid] = daily

            try:
                # 计算该站点的霜冻等级统计
                frost_means, frost_stats_by_year = calculate_frost_level_stats(daily, algorithm_config)

                # 根据配置中的threshold顺序获取各等级值
                thresholds = algorithm_config.get("threshold", [])

                if len(thresholds) >= 1:  # 轻霜冻
                    d0 = frost_means.get(thresholds[0]["label"], frost_means.get("frost1", 0))
                    station_d0_values.append(d0)

                    # 收集逐年天数
                    for year_stats in frost_stats_by_year.values():
                        station_d0_list.append(year_stats.get(thresholds[0]["label"], year_stats.get("frost1", 0)))

                if len(thresholds) >= 2:  # 中霜冻
                    d1 = frost_means.get(thresholds[1]["label"], frost_means.get("frost2", 0))
                    station_d1_values.append(d1)

                    # 收集逐年天数
                    for year_stats in frost_stats_by_year.values():
                        station_d1_list.append(year_stats.get(thresholds[1]["label"], year_stats.get("frost2", 0)))

                if len(thresholds) >= 3:  # 重霜冻
                    d2 = frost_means.get(thresholds[2]["label"], frost_means.get("frost3", 0))
                    station_d2_values.append(d2)

                    # 收集逐年天数
                    for year_stats in frost_stats_by_year.values():
                        station_d2_list.append(year_stats.get(thresholds[2]["label"], year_stats.get("frost3", 0)))

                # 存储详细统计信息
                station_frost_stats[sid] = {
                    "means": frost_means,
                    "yearly_stats": frost_stats_by_year
                }

                # print(f"站点 {sid}: 轻={d0:.1f}天, 中={d1:.1f}天, 重={d2:.1f}天")

            except Exception as e:
                print(f"站点 {sid} 霜冻统计计算失败: {str(e)}")
                # 添加默认值以确保数组长度一致
                station_d0_values.append(0)
                station_d1_values.append(0)
                station_d2_values.append(0)

        # 在所有站点处理完成后打印统计信息
        print("\n" + "=" * 60)
        print("所有站点霜冻统计汇总")
        print("=" * 60)

        if station_d0_values and station_d1_values and station_d2_values:
            # 计算有效站点数
            valid_sites = len([v for v in station_d0_values if v > 0])
            total_sites = len(station_ids)

            print(f"站点总数: {total_sites}, 有效站点: {valid_sites}, 有效率: {valid_sites / total_sites * 100:.1f}%")

            # 各等级霜冻天数统计
            if len(thresholds) >= 1:
                d0_nonzero = [v for v in station_d0_values if v > 0]
                if d0_nonzero:
                    print(f"轻霜冻({thresholds[0]['label']}): "
                          f"平均{np.mean(d0_nonzero):.2f}天, "
                          f"范围[{min(d0_nonzero):.2f}, {max(d0_nonzero):.2f}], "
                          f"非零站点{len(d0_nonzero)}个")

            if len(thresholds) >= 2:
                d1_nonzero = [v for v in station_d1_values if v > 0]
                if d1_nonzero:
                    print(f"中霜冻({thresholds[1]['label']}): "
                          f"平均{np.mean(d1_nonzero):.2f}天, "
                          f"范围[{min(d1_nonzero):.2f}, {max(d1_nonzero):.2f}], "
                          f"非零站点{len(d1_nonzero)}个")

            if len(thresholds) >= 3:
                d2_nonzero = [v for v in station_d2_values if v > 0]
                if d2_nonzero:
                    print(f"重霜冻({thresholds[2]['label']}): "
                          f"平均{np.mean(d2_nonzero):.2f}天, "
                          f"范围[{min(d2_nonzero):.2f}, {max(d2_nonzero):.2f}], "
                          f"非零站点{len(d2_nonzero)}个")

            # 总霜冻统计
            total_frost_days = []
            for i in range(len(station_d0_values)):
                total = station_d0_values[i] + station_d1_values[i] + station_d2_values[i]
                total_frost_days.append(total)

            total_nonzero = [v for v in total_frost_days if v > 0]
            if total_nonzero:
                print(f"总霜冻天数: "
                      f"平均{np.mean(total_nonzero):.2f}天, "
                      f"范围[{min(total_nonzero):.2f}, {max(total_nonzero):.2f}]")

            # 逐年数据统计
            print(f"\n逐年数据样本数:")
            print(f"  轻霜冻逐年样本: {len(station_d0_list)} 个")
            print(f"  中霜冻逐年样本: {len(station_d1_list)} 个")
            print(f"  重霜冻逐年样本: {len(station_d2_list)} 个")

        # 第二步：对所有站点的d0\d1\d2进行归一化
        if station_d0_list and station_d0_values:  # 轻霜冻
            min_val_d0 = min(station_d0_list)
            max_val_d0 = max(station_d0_list)
            normalized_d0 = normalize_values(station_d0_values, min_val_d0, max_val_d0)
            print(f"轻霜冻归一化范围: [{min_val_d0:.2f}, {max_val_d0:.2f}], 站点数: {len(normalized_d0)}")
        else:
            normalized_d0 = []
            print("警告: 轻霜冻数据为空")

        if station_d1_list and station_d1_values:  # 中霜冻
            min_val_d1 = min(station_d1_list)
            max_val_d1 = max(station_d1_list)
            normalized_d1 = normalize_values(station_d1_values, min_val_d1, max_val_d1)
            print(f"中霜冻归一化范围: [{min_val_d1:.2f}, {max_val_d1:.2f}], 站点数: {len(normalized_d1)}")
        else:
            normalized_d1 = []
            print("警告: 中霜冻数据为空")

        if station_d2_list and station_d2_values:  # 重霜冻
            min_val_d2 = min(station_d2_list)
            max_val_d2 = max(station_d2_list)
            normalized_d2 = normalize_values(station_d2_values, min_val_d2, max_val_d2)
            print(f"重霜冻归一化范围: [{min_val_d2:.2f}, {max_val_d2:.2f}], 站点数: {len(normalized_d2)}")
        else:
            normalized_d2 = []
            print("警告: 重霜冻数据为空")

        # 数据一致性检查
        valid_stations_count = len(station_ids)
        if len(normalized_d0) != valid_stations_count:
            print(f"警告: 轻霜冻归一化数据长度({len(normalized_d0)})与站点数({valid_stations_count})不匹配")
        if len(normalized_d1) != valid_stations_count:
            print(f"警告: 中霜冻归一化数据长度({len(normalized_d1)})与站点数({valid_stations_count})不匹配")
        if len(normalized_d2) != valid_stations_count:
            print(f"警告: 重霜冻归一化数据长度({len(normalized_d2)})与站点数({valid_stations_count})不匹配")

        # 输出归一化前后的统计信息
        if station_d0_values and station_d1_values and station_d2_values:
            valid_d0 = [v for v in station_d0_values if v is not None and not np.isnan(v)]
            valid_d1 = [v for v in station_d1_values if v is not None and not np.isnan(v)]
            valid_d2 = [v for v in station_d2_values if v is not None and not np.isnan(v)]
            if valid_d0 and valid_d1 and valid_d2:
                print(f"d0原始范围: {min(valid_d0):.4f} ~ {max(valid_d0):.4f}")
                print(f"d1原始范围: {min(valid_d1):.4f} ~ {max(valid_d1):.4f}")
                print(f"d2原始范围: {min(valid_d2):.4f} ~ {max(valid_d2):.4f}")
                print(f"d0归一化范围: {min(normalized_d0):.4f} ~ {max(normalized_d0):.4f}")
                print(f"d1归一化范围: {min(normalized_d1):.4f} ~ {max(normalized_d1):.4f}")
                print(f"d2归一化范围: {min(normalized_d2):.4f} ~ {max(normalized_d2):.4f}")

        # 第三步：使用归一化后的轻/中/重计算每个站点的W值
        station_values: Dict[str, float] = {}
        formula_config = algorithm_config.get("formula", {})

        # 获取公式字符串
        formula_str = formula_config.get("formula", "w1*d0_norm+w2*d1_norm+w3*d2_norm")
        print(f"计算公式: {formula_str}")

        # 从threshold配置中提取权重
        thresholds = algorithm_config.get("threshold", [])
        weights_dict = {}

        # 提取各等级的权重，按照level排序
        for thresh in thresholds:
            level = thresh.get("level", 1)
            weight = thresh.get("weight", 0.0)
            label = thresh.get("label", "")

            # 映射到公式中的变量名
            if level == 1:
                weights_dict['w1'] = weight
                weights_dict['d0_norm_label'] = label  # 轻霜冻
            elif level == 2:
                weights_dict['w2'] = weight
                weights_dict['d1_norm_label'] = label  # 中霜冻
            elif level == 3:
                weights_dict['w3'] = weight
                weights_dict['d2_norm_label'] = label  # 重霜冻

        print(f"权重配置: w1={weights_dict.get('w1', 0.2)}, "
              f"w2={weights_dict.get('w2', 0.3)}, "
              f"w3={weights_dict.get('w3', 0.3)}")

        for i, sid in enumerate(station_ids):
            try:
                d0_norm = normalized_d0[i]
                d1_norm = normalized_d1[i]
                d2_norm = normalized_d2[i]

                # 准备变量字典 - 包含归一化值和权重
                variables = {
                    'd0_norm': float(d0_norm),
                    'd1_norm': float(d1_norm),
                    'd2_norm': float(d2_norm),
                    'w1': float(weights_dict.get('w1', 0.2)),
                    'w2': float(weights_dict.get('w2', 0.3)),
                    'w3': float(weights_dict.get('w3', 0.3)),
                    'np': np
                }

                # 安全地计算公式
                low_value = eval(formula_str, {"__builtins__": {}}, variables)

                # 转换并检查有效性
                if np.isfinite(low_value):
                    station_values[sid] = float(low_value)
                else:
                    station_values[sid] = np.nan
                    print(f"站点 {sid} 计算结果非有限值: {low_value}")
            except Exception as e:
                print(f"站点 {sid} 综合值计算失败: {str(e)}")
                station_values[sid] = np.nan
        # 第四步：插值计算
        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {'station_values': station_values, 'station_coords': station_coords,
                       'grid_path': cfg.get('gridFilePath'), 'dem_path': cfg.get('demFilePath'),
                       'area_code': cfg.get('areaCode'), 'shp_path': cfg.get('shpFilePath')}

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 分级
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]
        # 执行
        classdata = classificator.execute(result["data"], class_conf)

        return {'data': classdata, 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'],
                                            'transform': result['meta']['transform'], 'crs': result['meta']['crs']},
                'type': '霜冻'}

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']
        if disaster_type == 'GH':
            return self.calculate_drought(params)
        elif disaster_type == 'CJWSD':  # 晚霜冻
            return self._calculate_frost(params)
        elif disaster_type == 'SD':  # 霜冻
            return self._calculate_frost(params)
        elif disaster_type == 'dry':  # 干热风
            return self.calculate_dry(params)
        elif disaster_type == 'LCY':  # 麦收区连阴雨
            return self.calculate_wet(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
