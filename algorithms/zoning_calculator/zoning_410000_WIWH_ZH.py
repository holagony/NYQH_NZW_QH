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
        
class WIWH_ZH:
    '''
    河南-冬小麦-灾害区划
    干旱区划
    晚霜冻气候区划 TODO
    麦收区连阴雨气候区划 TODO
    干热风区划 TODO
    '''
    def _calculate_continuous_rain_indicators_station(self, station_indicators,params):
        """在站点级别计算连阴雨指标"""
        continuous_rain_indicators = {}
        
        for station_id, indicators in station_indicators.items():
            
            # 获取基础指标
            Pre = indicators.get('Pre', np.nan)  # 总降水量
            SSH = indicators.get('SSH', np.nan)  # 总日照时数
            Pre_days = indicators.get('Pre_days', np.nan)  # 降水日数

            # str转字典
            Pre_df =pd.DataFrame.from_dict(Pre, orient='index')
            SSH_df =pd.DataFrame.from_dict(SSH, orient='index')
            Pre_days_df =pd.DataFrame.from_dict(Pre_days, orient='index')

            merged_df = pd.concat([Pre_df, SSH_df, Pre_days_df], axis=1)
            merged_df.columns=['Pre','SSH','Pre_days']
           
            # 按标准赋值
            merged_df['Pre']=merged_df['Pre'].apply(pre_category)
            merged_df['SSH']=merged_df['SSH'].apply(ssh_category)
            merged_df['Pre_days']=merged_df['Pre_days'].apply(pre_days_category)
            cleaned_df = merged_df.dropna()

            # 年度指数
            cleaned_df['年度指数']=cleaned_df['Pre']+cleaned_df['SSH']+cleaned_df['Pre_days']
            
            # 连阴雨程度与
            cleaned_df['连阴雨程度']=cleaned_df['年度指数'].apply(index)
            
            # 综合指数
            frequency = cleaned_df['连阴雨程度'].value_counts().sort_index()
            for level in [0, 1, 2, 3]:
                if level not in frequency:
                    frequency[level] = 0
            weighted_frequency = (0.5 * frequency.get(3, 0) + 
                                  0.3 * frequency.get(2, 0) + 
                                  0.2 * frequency.get(1, 0))
            continuous_rain_indicators[station_id] = weighted_frequency/len(cleaned_df)
        
        max_value = max(continuous_rain_indicators.values())
        max_keys = [key for key, value in continuous_rain_indicators.items() if value == max_value]
        min_value = min(continuous_rain_indicators.values())
        min_keys = [key for key, value in continuous_rain_indicators.items() if value == min_value]
        print(f'麦收区连阴雨气候区划:有效站点数据：{len(cleaned_df)}')
        print(f'麦收区连阴雨气候区划：单站最高综合指数：{max_keys}：{max_value}')
        print(f'麦收区连阴雨气候区划:单站最低综合指数：{min_keys}：{min_value}')

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'连阴雨指标_{timestamp}.csv'
        result_df = pd.DataFrame(list(continuous_rain_indicators.items()),columns=['站点ID', '连阴雨综合指数'])
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
        interpolation_data = {
            'station_values': continuous_rain_risk_station,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")
        }
        
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
    

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], 
                                indicator_name: str) -> None:
        """保存中间结果 - 各个指标的插值结果"""
        try:
            print(f"保存中间结果: {indicator_name}")
            
            # 生成中间结果文件名
            file_name = indicator_name+".tif"
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
    
    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
        """使用GDAL保存为GeoTIFF文件"""
        try:
            from osgeo import gdal, osr
            
            # 确保数据是2D的
            if len(data.shape) == 1:
                # 如果是1D数据，需要知道宽度和高度才能重塑
                if meta.get('width') and meta.get('height'):
                    data = data.reshape((meta['height'], meta['width']))
                else:
                    # 如果不知道形状，创建为1行N列
                    data = data.reshape((1, -1))
            elif len(data.shape) > 2:
                data = data.squeeze()  # 移除单维度
            
            # 获取数据形状
            height, width = data.shape
            
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
                
            # 创建GeoTIFF文件
            driver = gdal.GetDriverByName('GTiff')
            
            # 创建数据集
            dataset = driver.Create(
                str(output_path),
                width,
                height,
                1,  # 波段数
                datatype,
                ['COMPRESS=LZW']  # 使用LZW压缩
            )
            
            if dataset is None:
                raise ValueError(f"无法创建文件: {output_path}")
            
            # 设置地理变换参数
            transform = meta.get('transform')
            dataset.SetGeoTransform(transform)
            
            # 设置投影
            crs = meta.get('crs')
            if crs is not None:
                dataset.SetProjection(crs)
            else:
                print("警告: 没有坐标参考系统信息")
            
            # 获取波段并写入数据
            band = dataset.GetRasterBand(1)
            band.WriteArray(data)
            
            # 设置无数据值
            nodata = meta.get('nodata')
            if nodata is not None:
                band.SetNoDataValue(float(nodata))
            
            # 关闭数据集，确保数据写入磁盘
            dataset = None
            
            print(f"GeoTIFF文件保存成功: {output_path}")
            
        except ImportError as e:
            print(f"导入GDAL失败: {str(e)}")
        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise

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
            ranges = [(pd.Timestamp(y - 1, 8, 1), pd.Timestamp(y - 1, 10, 10)), (pd.Timestamp(y - 1, 10, 11), pd.Timestamp(y - 1, 12, 20)),
                      (pd.Timestamp(y - 1, 12, 21), pd.Timestamp(y, 2, 20)), (pd.Timestamp(y, 2, 21), pd.Timestamp(y, 3, 31)),
                      (pd.Timestamp(y, 4, 1), pd.Timestamp(y, 4, 30)), (pd.Timestamp(y, 5, 1), pd.Timestamp(y, 5, 20)),
                      (pd.Timestamp(y, 5, 21), pd.Timestamp(y, 6, 10))]

            means = []
            for s, e in ranges:
                seg = series[(series.index >= s) & (series.index <= e)]
                means.append(float(seg.mean()) if len(seg) > 0 else np.nan)

            m = np.array(means, dtype=float)
            vals.append(float(np.nansum(weights * m)))

        total = float(sum(vals))

        return total / float(len(years))

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
        
        # 输出result的数值范围
        data_min = float(np.nanmin(result['data']))
        data_max = float(np.nanmax(result['data']))
        print(f"干旱指数数值范围: {data_min:.4f} ~ {data_max:.4f}")

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
            'type': '河南冬小麦干旱'
        }

    def calculate_freeze(self, params):
        '''
        霜冻区划
        计算代码写这里
        '''
        pass

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
            continuous_rain_indicators = self._calculate_continuous_rain_indicators_station(station_indicators,config)
            
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
            result = {
                'data': classified_data,
                'meta': interpolated_risk['meta'],
                'type': 'continuous_rain_risk',
                'process': 'station_level_calculation'
            }
            print("小麦连阴雨风险计算完成")

        except Exception as e:
            print(f"小麦连阴雨风险计算失败: {str(e)}")
            result = np.nan
        return result

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        self._algorithms = params['algorithms']
        if disaster_type == 'GH':
            return self.calculate_drought(params)
        elif disaster_type == 'freeze':  # 晚霜冻
            return self.calculate_freeze(params)
        elif disaster_type == 'dry':  # 干热风
            return self.calculate_dry(params)
        elif disaster_type == 'LCY':  # 麦收区连阴雨
            return self.calculate_wet(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
