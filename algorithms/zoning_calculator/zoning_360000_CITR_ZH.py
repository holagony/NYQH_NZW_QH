import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from math import exp, sqrt, pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
from osgeo import gdal
from scipy.ndimage import sobel


def _diffusion_h(a, b, m):
    # 扩散系数 h 的分段取值，基于样本极差 (b-a) 与样本数 m
    if m == 5:
        return 0.8146 * (b - a)
    if m == 6:
        return 0.5690 * (b - a)
    if m == 7:
        return 0.4560 * (b - a)
    if m == 8:
        return 0.3860 * (b - a)
    if m == 9:
        return 0.3362 * (b - a)
    if m == 10:
        return 0.2986 * (b - a)
    if m >= 11:
        return 2.6851 * (b - a) / (m - 1)
    raise ValueError("m must be >= 5")


def information_diffusion_probabilities(U, Y):
    # 空样本：直接返回零概率向量，后续指数为0
    if len(Y) == 0:
        return np.zeros(len(U), dtype=float)
    m = len(Y)
    a = min(Y)
    b = max(Y)

    # 小样本自适应带宽：用极差或论域范围估算，避免退化
    if m < 5:
        h = (b - a)
        if h <= 0:
            u_range = (max(U) - min(U)) if U else 1.0
            h = max(1e-6, u_range / max(1, len(U)))
        else:
            h = max(1e-6, 0.5 * h)
    else:
        h = _diffusion_h(a, b, m)

    # 双保险：若仍出现非正带宽，按论域范围兜底为正值
    if h <= 0:
        u_range = (max(U) - min(U)) if U else 1.0
        h = max(1e-6, u_range / max(1, len(U)))

    # 计算每个样本 y_j 在各论域点 u_i 的扩散值（高斯核）
    f = []
    for y in Y:
        row = [1.0 / (h * sqrt(2 * pi)) * exp(-((y - u)**2) / (2 * h * h)) for u in U]
        f.append(row)

    # 对每个样本行做归一化，得到 μ_{y_j}(u_i)
    f = np.array(f, dtype=float)  # (Y行,U列)
    c = f.sum(axis=1, keepdims=True)
    mu = f / c

    # 聚合所有样本得到 q(u_i)，并对全体再归一化得到 p(u_i)
    q = mu.sum(axis=0)
    Q = float(q.sum())
    p = q / Q
    return p


def hazard_index(U_low, Y_low, U_mid, Y_mid, U_high, Y_high, method):
    # 计算危险性
    Pl = information_diffusion_probabilities(U_low, Y_low)
    Pm = information_diffusion_probabilities(U_mid, Y_mid)
    Ps = information_diffusion_probabilities(U_high, Y_high)

    # p(ui)是数组，是每个站点在每个出现频次的概率，频次为0表示不出现灾害，Pl表示出现灾害的概率，所以Pl为（1减频次为0的概率）
    idx0_l = U_low.index(0) if 0 in U_low else None
    idx0_m = U_mid.index(0) if 0 in U_mid else None
    idx0_s = U_high.index(0) if 0 in U_high else None
    p0_l = float(Pl[idx0_l]) if idx0_l is not None and len(Pl) > idx0_l else 0.0
    p0_m = float(Pm[idx0_m]) if idx0_m is not None and len(Pm) > idx0_m else 0.0
    p0_s = float(Ps[idx0_s]) if idx0_s is not None and len(Ps) > idx0_s else 0.0
    occ_l = 1.0 - p0_l
    occ_m = 1.0 - p0_m
    occ_s = 1.0 - p0_s

    H = 0.2 * occ_l + 0.3 * occ_m + 0.5 * occ_s
    return float(H)


def normalize_values(values: List[float]) -> List[float]:
    """
    归一化数值到0-1范围
    """
    if not values:
        return []

    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    if not valid_values:
        return [0.0] * len(values)

    min_val = min(valid_values)
    max_val = max(valid_values)

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


class CITR_ZH:
    '''
    江西灾害区划
    果实膨大期高温热害 GSPDQGWRH
    越冬冻害（12月-次年2月） YDDH
    '''

    def _align_and_read_input(self, grid_path, target_path, result_path):
        '''
        将单个外部栅格对齐到grid_path，并读取为数组
        target_path: 要对齐的目标栅格路径
        result_path: 对齐后的临时文件存储路径
        返回: 对齐后的numpy数组（NoData已置为NaN）
        '''
        temp_path = os.path.join(result_path, 'intermediate', 'align_temp.tif')
        aligned_path = LSMIDWInterpolation()._align_datasets(grid_path, target_path, temp_path)
        ds = gdal.Open(aligned_path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        # arr = np.where(arr == nodata, 0, arr) # 设置为0，而不是np.nan
        arr = np.where(arr == nodata, np.nan, arr)
        ds = None
        os.remove(aligned_path)
        return arr

    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
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

    def _calc_GSPDQGWRH_station_H(self, data, config):
        # 站点级：按年度滑窗统计轻/中/重事件次数，基于信息扩散计算危险性 H
        tavg = data["tavg"] if "tavg" in data.columns else pd.Series(dtype=float)
        tmax = data["tmax"] if "tmax" in data.columns else pd.Series(dtype=float)
        if tavg.empty or tmax.empty:
            return np.nan

        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        year_offset = int(config.get("year_offset", 0))
        years = sorted(data.index.year.unique())
        if not years:
            return np.nan

        # 每年事件次数序列（轻/中/重），用于后续信息扩散
        Y_low, Y_mid, Y_high = [], [], []
        for y in years:
            if start_date_str and end_date_str:
                # 按配置定义年度窗口（支持跨年）
                start_dt = pd.to_datetime(f"{y}-{start_date_str}")
                end_dt = pd.to_datetime(f"{y + year_offset}-{end_date_str}")
                mask = (data.index >= start_dt) & (data.index <= end_dt)
                dfy = data.loc[mask]
            else:
                dfy = data[data.index.year == y]

            if dfy.empty:
                # 当年无数据：次数记 0
                Y_low.append(0)
                Y_mid.append(0)
                Y_high.append(0)
                continue

            # 事件判据：日平均 ≥30 且 日最高 ≥36
            cond = (dfy["tavg"] >= 30) & (dfy["tmax"] >= 36)

            # 计算连续满足判据的段长度 d（滑窗/游程统计）
            runs = []
            c = 0  # 当前连续天数计数器
            for v in cond.values:
                if v:
                    c += 1  # 满足条件则累加连续天数
                else:
                    if c > 0:
                        runs.append(c)  # 一段连续区间结束，记录其长度
                        c = 0  # 计数器归零，等待下一段
            if c > 0:
                runs.append(c)  # 序列以满足条件结束，补记最后一段

            # 分级计数：轻 3–6天，中 7–9天，重 ≥10天
            light = sum(1 for L in runs if 3 <= L <= 6)
            medium = sum(1 for L in runs if 7 <= L <= 9)
            heavy = sum(1 for L in runs if L >= 10)
            Y_low.append(light)
            Y_mid.append(medium)
            Y_high.append(heavy)

        # 论域（年度次数）按 0..max(Y) 构建，至少包含 0
        U_low = list(range(0, (max(Y_low) if Y_low else 0) + 1)) or [0]
        U_mid = list(range(0, (max(Y_mid) if Y_mid else 0) + 1)) or [0]
        U_high = list(range(0, (max(Y_high) if Y_high else 0) + 1)) or [0]

        # 信息扩散 + 加权归约，得到站点危险性指数 H
        H = hazard_index(U_low, Y_low, U_mid, Y_mid, U_high, Y_high, method='mean')

        return float(H)

    def calculate_GSPDQGWRH(self, params):
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        hazard_config = algorithm_config.get('hazard', {})
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

        # 逐站点获取数据 + 计算危险性H
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            H = self._calc_GSPDQGWRH_station_H(daily, hazard_config)
            station_values[sid] = float(H) if np.isfinite(H) else np.nan

        # 输出插值前站点数值范围
        vals = [v for v in station_values.values() if not np.isnan(v)]
        if vals:
            data_min = float(np.min(vals))
            data_max = float(np.max(vals))
            print(f"插值前站点数值范围: {data_min:.4f} ~ {data_max:.4f}")

        # 危险性H插值
        interp_conf = algorithm_config.get('interpolation')
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {'station_values': station_values, 'station_coords': station_coords, 'grid_path': cfg.get('gridFilePath'), 'dem_path': cfg.get('demFilePath'), 'area_code': cfg.get('areaCode'), 'shp_path': cfg.get('shpFilePath')}

        if method == 'lsm_idw':  # 生成tif
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 数值设置 + tiff保存
        # result['data'] = np.maximum(result['data'], 0)
        # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])  # 将NaN也设为0
        result['data'] = normalize_array(result['data'])  # 归一化
        H_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "高温热害危险性指数.tif")
        self._save_geotiff(result['data'], result['meta'], H_tif_path, 0)

        # 读取其他静态数据，结合危险性H，计算区划风险
        # czt_path = cfg.get('cztFilePath')
        # yzhj_path = cfg.get('yzhjFilePath')
        # fzjz_path = cfg.get('fzjzFilePath')
        # grid_path = interp_data['grid_path']
        # czt_array = self._align_and_read_input(grid_path, czt_path, cfg.get('resultPath'))
        # yzhj_array = self._align_and_read_input(grid_path, yzhj_path, cfg.get('resultPath'))
        # fzjz_array = self._align_and_read_input(grid_path, fzjz_path, cfg.get('resultPath'))

        # risk = result['data'].astype(np.float32) * 0.7 + \
        #        yzhj_array.astype(np.float32) * 0.1 + \
        #        czt_array.astype(np.float32) * 0.1 + \
        #        (1.0 - fzjz_array.astype(np.float32)) * 0.1

        # risk_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        # self._save_geotiff(risk, result['meta'], risk_tif_path, 0)  # 保存干旱综合风险指数
        # result['data'] = risk

        # 分级
        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            algos = params.get('algorithms', {})
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)

        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}}

    def _calc_YDDH_station_H(self, data, config):
        tmin = data["tmin"] if "tmin" in data.columns else pd.Series(dtype=float)
        if tmin.empty:
            return np.nan

        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        year_offset = int(config.get("year_offset", 0))
        years = sorted(data.index.year.unique())
        if not years:
            return np.nan

        Y_low, Y_mid, Y_high = [], [], []
        for y in years:
            if start_date_str and end_date_str:
                start_dt = pd.to_datetime(f"{y}-{start_date_str}")
                end_dt = pd.to_datetime(f"{y + year_offset}-{end_date_str}")
                mask = (data.index >= start_dt) & (data.index <= end_dt)
                dfy = data.loc[mask]
            else:
                dfy = data[data.index.year == y]

            # 极端最低气温分级统计：统计该年满足阈值的“总天数/总次数”
            if "tmin" in dfy.columns and not dfy["tmin"].empty:
                tmin_vals = dfy["tmin"].to_numpy()
                light = int(np.count_nonzero((tmin_vals > -5) & (tmin_vals <= -3)))
                medium = int(np.count_nonzero((tmin_vals > -7) & (tmin_vals <= -5)))
                heavy = int(np.count_nonzero(tmin_vals <= -7))
            else:
                light = 0
                medium = 0
                heavy = 0

            Y_low.append(light)
            Y_mid.append(medium)
            Y_high.append(heavy)

        # 论域（年度次数）按 0..max(Y) 构建，至少包含 0
        U_low = list(range(0, (max(Y_low) if Y_low else 0) + 1)) or [0]
        U_mid = list(range(0, (max(Y_mid) if Y_mid else 0) + 1)) or [0]
        U_high = list(range(0, (max(Y_high) if Y_high else 0) + 1)) or [0]

        # 信息扩散 + 加权归约，得到站点危险性指数 H
        H = hazard_index(U_low, Y_low, U_mid, Y_mid, U_high, Y_high, method='mean')

        return float(H)

    def calculate_YDDH(self, params):
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        hazard_config = algorithm_config.get('hazard', {})
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
            H = self._calc_YDDH_station_H(daily, hazard_config)
            station_values[sid] = float(H) if np.isfinite(H) else np.nan

        # 输出插值前站点数值范围
        vals = [v for v in station_values.values() if not np.isnan(v)]
        if vals:
            data_min = float(np.min(vals))
            data_max = float(np.max(vals))
            print(f"插值前站点数值范围: {data_min:.4f} ~ {data_max:.4f}")

        # 危险性H插值
        interp_conf = algorithm_config.get('interpolation')
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {'station_values': station_values, 'station_coords': station_coords, 'grid_path': cfg.get('gridFilePath'), 'dem_path': cfg.get('demFilePath'), 'area_code': cfg.get('areaCode'), 'shp_path': cfg.get('shpFilePath')}

        if method == 'lsm_idw':  # 生成tif
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)

        # 数值设置 + tiff保存
        # result['data'] = np.maximum(result['data'], 0)
        # result['data'] = np.where(np.isnan(result['data']), 0, result['data'])  # 将NaN也设为0
        result['data'] = normalize_array(result['data'])  # 归一化
        H_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "低温冷害危险性指数.tif")
        self._save_geotiff(result['data'], result['meta'], H_tif_path, 0)

        # 读取其他静态数据，结合危险性H，计算区划风险
        # czt_path = cfg.get('cztFilePath')
        # yzhj_path = cfg.get('yzhjFilePath')
        # fzjz_path = cfg.get('fzjzFilePath')
        # grid_path = interp_data['grid_path']
        # czt_array = self._align_and_read_input(grid_path, czt_path, cfg.get('resultPath'))
        # yzhj_array = self._align_and_read_input(grid_path, yzhj_path, cfg.get('resultPath'))
        # fzjz_array = self._align_and_read_input(grid_path, fzjz_path, cfg.get('resultPath'))

        # risk = result['data'].astype(np.float32) * 0.7 + \
        #        yzhj_array.astype(np.float32) * 0.1 + \
        #        czt_array.astype(np.float32) * 0.1 + \
        #        (1.0 - fzjz_array.astype(np.float32)) * 0.1

        # risk_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        # self._save_geotiff(risk, result['meta'], risk_tif_path, 0)  # 保存干旱综合风险指数
        # result['data'] = risk

        # 分级
        class_conf = algorithm_config.get('classification', {})
        if class_conf:
            algos = params.get('algorithms', {})
            key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
            if key in algos:
                result['data'] = algos[key].execute(result['data'], class_conf)

        return {'data': result['data'], 'meta': {'width': result['meta']['width'], 'height': result['meta']['height'], 'transform': result['meta']['transform'], 'crs': result['meta']['crs']}}

    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']
        if disaster_type == 'GSPDQGWRH':
            return self.calculate_GSPDQGWRH(params)
        elif disaster_type == 'YDDH':
            return self.calculate_YDDH(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")
