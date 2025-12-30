import numpy as np
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal


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
    黑龙江-大豆-灾害区划-大豆冷害
    '''
    def _get_algorithm(self, algorithm_name):
        """从算法注册器获取实例（插值/分类等）"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
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

        interp_data = {
            'station_values': dangerous_station,
            'station_coords': coords_used,
            'grid_path': config.get('gridFilePath'),
            'dem_path': config.get('demFilePath'),
            'area_code': config.get('areaCode'),
            'shp_path': config.get('shpFilePath')
        }

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 归一化栅格并保存中间结果tif
        result_norm = normalize_array(result['data'])
        g_tif_path = os.path.join(config.get("resultPath"), "intermediate", "低温冷害危险性指数_归一化.tif")
        self._save_geotiff_gdal(result_norm, result['meta'], g_tif_path, 0)

        # 增加分级
        class_conf = algorithmConfig.get('classification', {})
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            classifier = self._get_algorithm(f"classification.{method}")
            data_out = classifier.execute(result['data'], class_conf)
            class_tif = os.path.join(config.get("resultPath"), "低温冷害危险性指数_分级.tif")
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

        if disaster_type == 'DWLH':
            return self._calculate_DWLH(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
