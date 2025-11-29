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


class WIWH_ZH:
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
        all_stations_raw_data = pd.DataFrame()

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

            # 保存原始数据（分类前）到总数据框中
            raw_data_copy = merged_df.copy()
            raw_data_copy['站点ID'] = station_id
            all_stations_raw_data = pd.concat([all_stations_raw_data, raw_data_copy])

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

        station_avg = all_stations_raw_data.groupby('站点ID')[['Pre', 'SSH', 'Pre_days']].mean()
        station_avg.to_csv(intermediate_dir / f'连阴雨指标_站点多年平均_{timestamp}.csv', index=True, encoding='utf-8-sig')
        
        return continuous_rain_indicators

    def _calculate_GRF_index(self, station_indicators: pd.DataFrame, config: Dict[str, Any]) -> Dict[int, float]:
        """
        计算干热风强度指数R
        
        计算公式：R = Σ(Wi * Di * Ni)
        其中：Wi为权重，Di为标记值，Ni为干热风日数
        """
        GRF_indicators = {}

        for station_id, indicators in station_indicators.items():

            # 获取基础指标
            GRF_light = indicators.get('GRF_light', np.nan)  # 总降水量
            GRF_severe = indicators.get('GRF_severe', np.nan)  # 降水日数

            # str转字典
            GRF_light = pd.DataFrame.from_dict(GRF_light, orient='index')
            GRF_severe = pd.DataFrame.from_dict(GRF_severe, orient='index')

            merged_df = pd.concat([GRF_light, GRF_severe], axis=1)
            merged_df.columns = ['GRF_light', 'GRF_severe']


            merged_df['GRF_index'] = merged_df['GRF_light'] * 0.35+ \
                                     merged_df['GRF_severe'] * 0.65

            GRF_indicators[station_id] = merged_df['GRF_index'].mean()
            '''
            GRF_indicators[station_id] = merged_df['GRF_light'].mean() * self.WEIGHTS['GRF_light'] * self.MARKS['GRF_light'] + \
                                     merged_df['GRF_moderate'].mean() * self.WEIGHTS['GRF_moderate'] * self.MARKS['GRF_moderate'] + \
                                     merged_df['GRF_severe'].mean() * self.WEIGHTS['GRF_severe'] * self.MARKS['GRF_severe']
            '''
        max_value = max(GRF_indicators.values())
        max_keys = [key for key, value in GRF_indicators.items() if value == max_value]
        min_value = min(GRF_indicators.values())
        min_keys = [key for key, value in GRF_indicators.items() if value == min_value]
        print(f'河南冬小麦干热风区划:有效站点数据：{len(merged_df)}')
        print(f'河南冬小麦干热风区划：单站最高干热风强度指数：{max_keys}：{max_value}')
        print(f'河南冬小麦干热风区划:单站最低干热风强度指数：{min_keys}：{min_value}')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'干热风强度指数_{timestamp}.csv'
        result_df = pd.DataFrame(list(GRF_indicators.items()), columns=['站点ID', '干热风强度指数'])   
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / filename
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"干热风强度指数文件已保存为 '{output_path}'")

        return GRF_indicators

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name,min_value=np.nan,max_value=np.nan):
        """对指定指标进行插值计算"""
        print(f"执行{indicator_name}插值计算...")
        
        config = params['config']        
        # 获取插值算法
        algorithmConfig = params['algorithmConfig']
        interpolation_config = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        
        # 准备插值参数
        interpolation_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value':min_value,
            'max_value':max_value,
        }
        
        file_name = f"intermediate_{indicator_name}.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name
        
        if not os.path.exists(output_path):
            interpolated_result = interpolator.execute(interpolation_data, interpolation_config.get("params", {}))
            print(f"{indicator_name}插值完成")
            
            # 保存中间结果
            self._save_intermediate_raster(interpolated_result, output_path)
        else:
            interpolated_result = self._load_intermediate_raster(output_path)
        
        return interpolated_result

    def _perform_classification(self, data_interpolated, params):
        """分级计算"""
        print("执行区域分级计算...")
        
        algorithmConfig = params['algorithmConfig']
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(data_interpolated['data'], classification)
        
        data_interpolated['data'] = data
        return data_interpolated


    def _save_intermediate_raster(self, result, output_path):
        """保存中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        data = result['data']
        meta = result['meta']
        
        # 根据数据类型确定GDAL数据类型
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32
        
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
            meta['width'],
            meta['height'],
            1,
            datatype,
            ['COMPRESS=LZW']
        )
        
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0)
        
        band.FlushCache()
        dataset = None
        
        print(f"中间栅格结果已保存: {output_path}")

    def _load_intermediate_raster(self, input_path):
        """加载中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        dataset = gdal.Open(str(input_path))
        if dataset is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        
        transform = dataset.GetGeoTransform()
        crs = dataset.GetProjection()
        
        meta = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        
        dataset = None
        
        return {
            'data': data,
            'meta': meta
        }

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

       """计算干热风 - 先计算站点综合风险指数再插值"""
       station_indicators = params['station_indicators']
       station_coords = params['station_coords']
       algorithmConfig = params['algorithmConfig']
       config = params['config']

       print("开始计算干热风风险 - 新流程：先计算站点综合风险指数")

       try:
            # 第一步：在站点级别计算连阴雨指标
            print('第一步，计算各站的干热风强度指数，并求历年平均')
            GRF_index = self._calculate_GRF_index(station_indicators, config)

            # 第二步：对综合风险指数F进行插值
            print("第二步：对综合风险指数F进行插值")
            interpolated_risk = self._perform_interpolation_for_indicator(GRF_index, station_coords, params, "GRF_risk")

            # 第三步：对插值结果进行分类
            print('第三步，基于插值栅格化指数进行区划分级')
            result = self._perform_classification(interpolated_risk, params)
            
            print(f'计算{params["config"].get("cropCode","")}-{params["config"].get("zoningType","")}-{params["config"].get("element","")}-区划完成')

       except Exception as e:
            print(f"干热风风险计算失败: {str(e)}")
            result = np.nan

       return result

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
            interpolated_risk = self._perform_interpolation_for_indicator(continuous_rain_indicators, station_coords, params, "LCY_risk")

            # 第三步：对插值结果进行分类
            print('第三步，基于插值栅格化指数进行区划分级')
            result = self._perform_classification(interpolated_risk, params)
            
            print(f'计算{params["config"].get("cropCode","")}-{params["config"].get("zoningType","")}-{params["config"].get("element","")}-区划完成')

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

        # 第一步：收集所有站点的d0和d1值
        station_d0_values = []
        station_d1_values = []
        station_d0_list = []
        station_d1_list = []
        station_data_map = {}  # 存储站点数据以便后续使用

        print("收集所有站点的d0和d1值...")
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            station_data_map[sid] = daily
            d0, d0list = calculate_tmin0(daily)
            d1, d1list = calculate_tmin1(daily)
            station_d0_values.append(d0)
            station_d1_values.append(d1)
            station_d0_list.extend(d0list)
            station_d1_list.extend(d1list)
            #print(f"站点 {sid}: Ih={Ih:.4f}, Dv={Dv:.4f}")
        min_val_d0 = min(station_d0_list)
        max_val_d0 = max(station_d0_list)
        min_val_d1 = min(station_d1_list)
        max_val_d1 = max(station_d1_list)
        # 第二步：对所有站点的d0\d1进行归一化
        normalized_d0 = normalize_values(station_d0_values, min_val_d0, max_val_d0)
        normalized_d1 = normalize_values(station_d1_values, min_val_d1, max_val_d1)

        # 输出归一化前后的统计信息
        if station_d0_values and station_d1_values:
            valid_d0 = [v for v in station_d0_values if v is not None and not np.isnan(v)]
            valid_d1 = [v for v in station_d1_values if v is not None and not np.isnan(v)]
            if valid_d0 and valid_d1:
                print(f"d0原始范围: {min(valid_d0):.4f} ~ {max(valid_d0):.4f}")
                print(f"d1原始范围: {min(valid_d1):.4f} ~ {max(valid_d1):.4f}")
                print(f"d0归一化范围: {min(normalized_d0):.4f} ~ {max(normalized_d0):.4f}")
                print(f"d1归一化范围: {min(normalized_d1):.4f} ~ {max(normalized_d1):.4f}")

        # 第三步：使用归一化后的d0和d1计算每个站点的W值
        station_values: Dict[str, float] = {}
        for i, sid in enumerate(station_ids):
            d0_norm = normalized_d0[i]
            d1_norm = normalized_d1[i]
            # 使用归一化后的值计算:low_value = 0.3 * d0_norm + 0.7 * d1_norm
            low_value = 0.3 * d0_norm + 0.7 * d1_norm
            station_values[sid] = float(low_value) if np.isfinite(low_value) else np.nan
            print(f"站点 {sid}: d0_norm={d0_norm:.4f}, d1_norm={d1_norm:.4f}, low_value={low_value:.4f}")

        # 第四步：插值计算
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

        # 分级
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]
        # 执行
        classdata = classificator.execute(result["data"], class_conf)

        return {'data': classdata, 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}, 'type': '河南冬小麦晚霜冻'}

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']
        if disaster_type == 'GH':
            return self.calculate_drought(params)
        elif disaster_type == 'CJWSD':  # 晚霜冻
            return self._calculate_frost(params)
        elif disaster_type == 'GRF':  # 干热风
            return self.calculate_dry(params)
        elif disaster_type == 'LCY':  # 麦收区连阴雨
            return self.calculate_wet(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
