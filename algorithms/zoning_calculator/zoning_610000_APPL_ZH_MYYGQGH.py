import os
import json
import numpy as np
import pandas as pd
from math import pi
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def _normalize_array(array):
    if array.size == 0:
        return array
    mask = ~np.isnan(array)
    if not np.any(mask):
        return array
    valid = array[mask].astype(float)
    mn = float(np.min(valid))
    mx = float(np.max(valid))
    if mx == mn:
        out = np.full_like(array, 0.5, dtype=float)
    else:
        out = (array.astype(float) - mn) / (mx - mn)
    out[~mask] = np.nan
    return out


class APPL_ZH:
    """苹果-萌芽幼果期干旱区划计算器
    
    职责
    - 计算站点干旱指标（以少雨天数频度为核心）
    - 对站点结果进行插值生成栅格
    - 可选分类输出专题产品
    """

    def _get_algorithm(self, algorithm_name):
        """从算法注册器中获取组件"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        # 根据数组 dtype 映射 GDAL 数据类型
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
        # 使用 LZW 压缩写出单波段 GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        dataset = None

    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        """统一插值入口"""
        # 选择插值方法及其参数（默认 lsm_idw）
        interpolation = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        # 构造标准数据包交给插值组件
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")
        }
        result = interpolator.execute(data, interpolation_params)
        return result

    def _calc_station_dr(self, daily, algorithm_config):
        """计算单站点干旱指标 DR（少雨日频度比例校正）
        
        定义
        - DR = 少雨天数 × k / 累计降水量
          其中少雨阈值为 thr（毫米），k 为比例修正系数，计算月份集合为 months
        
        参数
        - daily: pandas.DataFrame，逐日数据（需包含 'precip' 或 'P'），索引为 DatetimeIndex
        - algorithm_config: dict，包含 'dry_threshold_mm'、'k'、'months'
        
        返回
        - float，跨年平均 DR；若缺数据则返回 NaN
        """
        if daily is None or len(daily) == 0:
            return np.nan
        # 少雨判定阈值与比例因子
        thr = float(algorithm_config.get('dry_threshold_mm', 0.1))
        k = float(algorithm_config.get('k', 1.41))
        # 统计月份集合（默认 3-5 月）
        months = algorithm_config.get('months', [3, 4, 5])
        # 支持两个字段名：'precip' 或 'P'
        if 'precip' in daily.columns:
            p = daily['precip']
        elif 'P' in daily.columns:
            p = daily['P']
        else:
            return np.nan
        # 按年遍历，统计少雨天数与总降水
        years = sorted(p.index.year.unique())
        vals = []
        for y in years:
            sub = p[(p.index.year == y) & (p.index.month.isin(months))]
            if sub.size == 0:
                continue
            no_rain_days = int((sub <= thr).sum())
            rain_sum = float(np.nansum(sub.values))
            # 防止分母为 0
            denom = rain_sum if rain_sum > 0 else 1e-6
            dr = float(no_rain_days * k / denom)
            vals.append(dr)
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def calculate_MYYGQGH(self, params):
        """计算苹果-萌芽幼果期干旱产品
        
        流程
        1) 读取站点逐日数据，计算 DR 指标
        2) 对站点 DR 进行空间插值，得到栅格
        3) 可选按分类方法进行分级输出
        4) 写出中间及最终 GeoTIFF
        
        返回
        - dict，包含 'data'（最终分级或原值）与 'meta'（宽、高、仿射、坐标参考）
        """
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        # 数据管理器：按配置时间窗加载站点逐日数据
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        # 站点列表：优先使用传入坐标表，否则从数据目录读取
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        # 过滤仅保留可用站点
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        # 时间范围
        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')
        # 计算站点 DR 值
        station_values = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            dr = self._calc_station_dr(daily, algorithm_config)
            station_values[sid] = float(dr) if np.isfinite(dr) else np.nan
        # 空间插值
        interp = self._interpolate(station_values, station_coords, cfg, algorithm_config)
        # 插值结果归一化
        interp['data'] = _normalize_array(interp['data'])
        # 输出路径准备
        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        # 写出中间结果（原值栅格）
        tif_path = str(inter_dir / "萌芽幼果期干旱指数.tif")
        self._save_geotiff_gdal(interp['data'].astype(np.float32), interp['meta'], tif_path, 0)
        # 分类（可选）
        class_conf = algorithm_config.get('classification', {})
        data_out = interp['data']
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            try:
                classifier = self._get_algorithm(f"classification.{method}")
                data_out = classifier.execute(interp['data'].astype(float), class_conf)
            except Exception:
                data_out = interp['data']
        # 写出最终结果（分级或原值）
        final_tif = str(out_dir / "萌芽幼果期干旱_分级.tif")
        self._save_geotiff_gdal(np.array(data_out).astype(np.float32), interp['meta'], final_tif, 0)
        return {
            'data': np.array(data_out),
            'meta': {
                'width': interp['meta']['width'],
                'height': interp['meta']['height'],
                'transform': interp['meta']['transform'],
                'crs': interp['meta']['crs']
            }
        }

    def calculate(self, params):
        """统一入口：根据 element 选择目标计算
        
        仅当 config['element'] == 'MYYGQGH' 时执行当前计算
        """
        config = params['config']
        self._algorithms = params['algorithms']
        d = config.get('element')
        if d == 'MYYGQGH':
            return self.calculate_MYYGQGH(params)
        raise ValueError(f"不支持的灾害类型: {d}")
