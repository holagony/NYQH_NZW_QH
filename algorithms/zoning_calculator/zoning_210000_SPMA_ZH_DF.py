"""辽宁春玉米大风倒伏气候区划计算器

技术路线
- 逐年统计 8–10 月日最大风速 WIN_S_Max ≥ 13.9 m/s（7级及以上）的强风发生日数
- 利用小网格推算模型校正（如 LSM-IDW）进行空间插值
- 可选分级（自然断点/自定义阈值），生成春玉米大风倒伏气候区划产品

输入
- 逐日站点数据（DataManager）索引为日期，包含字段 WIN_S_Max（日最大风速）
- 配置 params.algorithmConfig：threshold（默认13.9）、interpolation、classification
- 配置 params.config：gridFilePath、demFilePath、shpFilePath、resultPath 等

输出
- 中间产品：多年均值栅格（强风发生日数多年平均）
- 最终产品：分级后的区划栅格（若提供分类配置）
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def _strong_wind_days_by_year(df: pd.DataFrame, threshold: float = 13.9) -> dict:
    """按年统计站点强风发生日数（仅统计8–10月）
    
    参数
    - df: 逐日气象数据，索引为日期，须包含列 WIN_S_Max（日最大风速，m/s）
    - threshold: 强风判定阈值（m/s），默认 13.9

    返回
    - dict: {年份 -> 该年8–10月中符合阈值的发生日数}
    """
    if 'WIN_S_Max' not in df.columns:
        return {}
    w = pd.to_numeric(df['WIN_S_Max'], errors='coerce')
    sub = w[(w >= threshold) & (w.notna()) & (df.index.month.isin([8, 9, 10]))]
    if sub.empty:
        return {}
    counts = sub.groupby(sub.index.year).count()
    return {int(k): float(v) for k, v in counts.items()}


class SPMA_ZH:
    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存单波段 GeoTIFF（自动匹配GDAL数据类型）
        
        参数
        - data: numpy 数组（栅格数据）
        - meta: 元数据字典，包含 width/height/transform/crs
        - output_path: 输出文件路径
        - nodata: 无效值（默认 0）
        """
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
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        dataset = None

    def _get_algorithm(self, algorithm_name):
        """从算法注册器获取实例（插值/分类等）"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_params_snapshot(self, params):
        """在结果目录保存参数快照，便于复算与追溯"""
        try:
            cfg = params.get('config', {})
            rp = cfg.get('resultPath') or os.getcwd()
            outdir = Path(rp)
            outdir.mkdir(parents=True, exist_ok=True)
            fp = outdir / "params_SPMA_DF.json"
            snap = {
                'station_coords': params.get('station_coords', {}),
                'algorithmConfig': params.get('algorithmConfig', {}),
                'config': cfg,
                'startDate': params.get('startDate') or cfg.get('startDate'),
                'endDate': params.get('endDate') or cfg.get('endDate')
            }
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            return str(fp)
        except Exception:
            return ""

    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        """统一插值入口：支持 LSM-IDW 等小网格校正模型"""
        interpolation = algorithmConfig.get("interpolation", {})
        method = interpolation.get('method', 'lsm_idw')
        iparams = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{method}")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")
        }
        return interpolator.execute(data, iparams)

    def _calculate_DF(self, params):
        """主流程：站点统计 → 插值校正 → 可选分级 → 输出栅格"""
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')

        values = {}
        for sid in station_ids:
            df = dm.load_station_data(sid, start_date, end_date)
            counts = _strong_wind_days_by_year(df, threshold=float(algorithm_config.get('threshold', 13.9)))
            if not counts:
                values[sid] = np.nan
                continue
            years = sorted(counts.keys())
            values[sid] = float(np.nanmean([counts[y] for y in years])) if years else np.nan

        interp = self._interpolate(values, station_coords, cfg, algorithm_config)
        nodata = interp['meta'].get('nodata', 0)
        grid = interp['data'].astype(np.float32)
        mask = ~np.isnan(grid)
        if np.any(mask):
            mn = float(np.nanmin(grid))
            mx = float(np.nanmax(grid))
            if mx > mn:
                norm = (grid - mn) / (mx - mn)
            else:
                norm = np.zeros_like(grid, dtype=np.float32)
        else:
            norm = grid
        norm[~mask] = np.nan

        norm_tif = os.path.join(cfg.get("resultPath"), "intermediate", "大风倒伏风险.tif")
        self._save_geotiff_gdal(norm, interp['meta'], norm_tif, nodata)

        class_conf = algorithm_config.get('classification', {})
        data_out = norm
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            try:
                classifier = self._get_algorithm(f"classification.{method}")
                data_out = classifier.execute(norm.astype(float), class_conf)
            except Exception:
                data_out = norm
            class_tif = os.path.join(cfg.get("resultPath"), "大风倒伏风险_分级.tif")
            self._save_geotiff_gdal(data_out.astype(np.int16), interp['meta'], class_tif, nodata)
        meta = interp['meta']

        return {'data': data_out.astype(np.int16),
                'meta': {'width': meta['width'], 'height': meta['height'], 'transform': meta['transform'], 'crs': meta['crs']},
                'type': '辽宁春玉米大风倒伏'}

    def calculate(self, params):
        self._save_params_snapshot(params)
        return self._calculate_DF(params)
