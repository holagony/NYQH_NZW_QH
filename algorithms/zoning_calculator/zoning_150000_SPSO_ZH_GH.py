import os
from re import S
import numpy as np
import pandas as pd
from math import pi
from osgeo import gdal
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


def calculate_cwdi(daily_data, weights, lat_deg=None, elev_m=None):
    '''
    计算内蒙古cwdi,kc使用内蒙古参数
    '''
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


def normalize_array(array):
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


class SPSO_ZH:
    '''
    内蒙古-大豆-灾害-干旱区划
    '''
    def _save_geotiff(self, data, meta, output_path, nodata=0):
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

    def _align_and_read_input(self, grid_path, target_path, result_path):
        '''
        将单个外部栅格对齐到grid_path，并读取为数组
        target_path: 要对齐的目标栅格路径
        result_path: 对齐后的临时文件存储路径
        返回: 对齐后的numpy数组（NoData已置为NaN）
        '''
        base_dir = result_path if result_path else os.path.dirname(grid_path)
        out_dir = os.path.join(base_dir, 'intermediate')
        os.makedirs(out_dir, exist_ok=True)
        temp_path = os.path.join(out_dir, 'align_temp.tif')
        if (not target_path) or (not os.path.exists(target_path)):
            gds = gdal.Open(grid_path)
            rows, cols = gds.RasterYSize, gds.RasterXSize
            gds = None
            return np.zeros((rows, cols), dtype=np.float32)
        aligned_path = LSMIDWInterpolation()._align_datasets(grid_path, target_path, temp_path)
        ds = gdal.Open(aligned_path)
        if ds is None:
            gds = gdal.Open(grid_path)
            rows, cols = gds.RasterYSize, gds.RasterXSize
            gds = None
            if os.path.exists(aligned_path):
                os.remove(aligned_path)
            return np.zeros((rows, cols), dtype=np.float32)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        ds = None
        if os.path.exists(aligned_path):
            os.remove(aligned_path)
        return arr

    def _calc_drought_station_g(self, data, config):
        '''
        计算每个站点的干旱风险性G
        '''
        df = calculate_cwdi(data, config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("lat_deg"), config.get("elev_m"))
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
        计算最终的干旱区划
        '''
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cwdi_config = algorithm_config.get('cwdi', {})
        cfg = params.get('config', {})
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')

        # 加载数据管理器
        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())

        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')
        station_values = {}

        # 逐站点获取数据 + 计算危险性G
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            g = self._calc_drought_station_g(daily, cwdi_config)  # 站点的干旱风险指数G
            station_values[sid] = float(g) if np.isfinite(g) else np.nan

        # 输出插值前站点数值范围
        vals = [v for v in station_values.values() if not np.isnan(v)]
        if vals:
            data_min = float(np.min(vals))
            data_max = float(np.max(vals))
            print(f"插值前站点数值范围: {data_min:.4f} ~ {data_max:.4f}")

        # 危险性G插值
        interp_conf = algorithm_config.get('interpolation')
        method = str(interp_conf.get('method', 'lsm_idw')).lower()
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

        if method == 'lsm_idw':  # 生成tif
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)
        print(f"插值后数值范围: {np.nanmin(result['data']):.4f} ~ {np.nanmax(result['data']):.4f}")

        # 数值设置 + tiff保存
        # result['data'] = np.maximum(result['data'], 0)
        # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])  # 将NaN也设为0
        result['data'] = normalize_array(result['data']) # 归一化

        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱危险性指数.tif")
        self._save_geotiff(result['data'], result['meta'], g_tif_path, 0)

        # 读取其他静态数据，结合危险性G，计算干旱风险
        czt_path = cfg.get('cztFilePath')
        yzhj_path = cfg.get('yzhjFilePath')
        fzjz_path = cfg.get('fzjzFilePath')
        grid_path = interp_data['grid_path']
        czt_array = self._align_and_read_input(grid_path, czt_path, cfg.get('resultPath'))
        yzhj_array = self._align_and_read_input(grid_path, yzhj_path, cfg.get('resultPath'))
        fzjz_array = self._align_and_read_input(grid_path, fzjz_path, cfg.get('resultPath'))
        czt_array = np.nan_to_num(czt_array, nan=0.0)
        yzhj_array = np.nan_to_num(yzhj_array, nan=0.0)
        fzjz_array = np.nan_to_num(fzjz_array, nan=0.0)

        # 防灾能力区域内存在nan的情况
        risk = np.nan_to_num(result['data'], nan=0.0).astype(np.float32) * 0.7 + \
               yzhj_array.astype(np.float32) * 0.1 + \
               czt_array.astype(np.float32) * 0.1 + \
               (1.0 - fzjz_array.astype(np.float32)) * 0.1
        
        print(f"综合风险指数数值范围: {np.nanmin(risk):.4f} ~ {np.nanmax(risk):.4f}")
        
        risk_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        self._save_geotiff(risk, result['meta'], risk_tif_path, 0)  # 保存干旱综合风险指数
        result['data'] = risk

        # 分级
        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            algos = params.get('algorithms', {})
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)
                class_tif_path = os.path.join(cfg.get("resultPath"), "干旱综合风险指数_分级.tif")
                self._save_geotiff(result['data'], result['meta'], class_tif_path, 0)

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

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'GH':
            return self.calculate_drought(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
