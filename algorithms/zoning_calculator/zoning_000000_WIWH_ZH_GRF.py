import os
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def _normalize_array(array):
    """栅格归一化到 [0,1]，保留 NaN；常数场归一化为 0.5"""
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


class WIWH_ZH:
    """全国冬小麦干热风灾害区划计算器
    
    职责
    - 结合站点生育期，统计轻/中/重度干热风日数
    - 按 Hd=Σ(Wi*Di*Ni) 计算站点多年平均强度指数
    - 对结果进行归一化与可选分级，输出专题产品
    """

    def _get_algorithm(self, algorithm_name):
        """从算法注册器中获取计算组件"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存 GeoTIFF 文件（单波段，LZW 压缩）"""
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

    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        """统一插值入口：支持 IDW/LSM-IDW 等方法"""
        interpolation = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
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

    def _parse_growth_mmdd(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return {"month": int(value.month), "day": int(value.day), "year_offset": 0}
        s = str(value).strip()
        if not s:
            return None
        year_offset = -1 if "上年" in s else 0
        s = (s.replace("上年", "").replace("月", "-").replace("日", "").replace("/", "-").replace(".", "-").strip())
        ts = pd.to_datetime(s, format="%m-%d", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return {"month": int(ts.month), "day": int(ts.day), "year_offset": year_offset}

    def _load_growth_periods(self, file_path, crop_name="冬小麦"):
        df = pd.read_excel(file_path)
        if "站号" not in df.columns:
            raise ValueError("生育期Excel缺少'站号'列")
        stage_cols = [c for c in ["播种", "出苗", "拔节", "孕穗", "抽穗", "开花", "成熟"] if c in df.columns]
        if not stage_cols:
            raise ValueError("生育期Excel缺少生育期列")

        if "作物" in df.columns and crop_name:
            sub = df[df["作物"].astype(str).str.contains(str(crop_name), na=False)]
            if not sub.empty:
                df = sub

        growth = {}
        for _, row in df.iterrows():
            sid = str(row["站号"]).strip()
            if not sid:
                continue
            growth[sid] = {c: row.get(c) for c in stage_cols}
        return growth

    def _calc_dryhotwind_index(self, daily, growth_row, years, stage_start="抽穗", stage_end="成熟", thresholds=None, weights=None, marks=None):
        if daily is None or len(daily) == 0 or "tmax" not in daily.columns:
            return np.nan
        if growth_row is None:
            return np.nan

        start_raw = growth_row.get(stage_start)
        end_raw = growth_row.get(stage_end)
        start_md = self._parse_growth_mmdd(start_raw)
        end_md = self._parse_growth_mmdd(end_raw)
        if not start_md or not end_md:
            return np.nan
        if stage_start in ("播种", "出苗") and start_md.get("year_offset", 0) == 0:
            start_md["year_offset"] = -1

        thr = thresholds
        w = weights
        m = marks

        tmax = daily["tmax"]
        rhum = daily["rhum"] if "rhum" in daily.columns else pd.Series(index=daily.index, dtype=float)
        wind = daily["wind"] if "wind" in daily.columns else pd.Series(index=daily.index, dtype=float)
        hd_list = []
        for y in years:
            start_dt = pd.Timestamp(
                year=y + start_md["year_offset"],
                month=start_md["month"],
                day=start_md["day"],
            )
            end_dt = pd.Timestamp(
                year=y + end_md["year_offset"],
                month=end_md["month"],
                day=end_md["day"],
            )
            if start_dt > end_dt:
                end_dt = end_dt + pd.DateOffset(years=1)

            t_sub = tmax[(tmax.index >= start_dt) & (tmax.index <= end_dt)]
            rh_sub = rhum[(rhum.index >= start_dt) & (rhum.index <= end_dt)]
            w_sub = wind[(wind.index >= start_dt) & (wind.index <= end_dt)]
            valid = t_sub.notna() & rh_sub.notna() & w_sub.notna()
            if not valid.any():
                continue

            sev = valid & (t_sub >= float(thr["severe"]["tmax"])) & (rh_sub <= float(thr["severe"]["rhum"])) & (w_sub >= float(thr["severe"]["wind"]))
            mod = valid & (t_sub >= float(thr["moderate"]["tmax"])) & (rh_sub <= float(thr["moderate"]["rhum"])) & (w_sub >= float(
                thr["moderate"]["wind"])) & (~sev)
            light = valid & (t_sub >= float(thr["light"]["tmax"])) & (rh_sub <= float(thr["light"]["rhum"])) & (w_sub >= float(
                thr["light"]["wind"])) & (~sev) & (~mod)

            n1 = float(light.sum())
            n2 = float(mod.sum())
            n3 = float(sev.sum())
            hd = float(w.get("light", 0.0)) * float(m.get("light", 0.0)) * n1 + \
                 float(w.get("moderate", 0.0)) * float(m.get("moderate", 0.0)) * n2 + \
                 float(w.get("severe", 0.0)) * float(m.get("severe", 0.0)) * n3
            hd_list.append(hd)

        if not hd_list:
            return np.nan
        return float(np.mean(hd_list))

    def calculate_GRF(self, params):
        """全国冬小麦干热风灾害区划计算主流程"""
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
        start_year = int(str(start_date)[:4])
        end_year = int(str(end_date)[:4])
        years = list(range(start_year, end_year + 1))

        growth_path = (cfg.get("growthPeriodPath") or params.get("growthPeriodPath"))
        if not growth_path:
            raise ValueError("缺少生育期Excel路径: growthPeriodPath")
        crop_name = algorithm_config.get("crop_name", "冬小麦")
        growth_map = self._load_growth_periods(growth_path, crop_name=crop_name)

        stage_start = algorithm_config.get("grf_stage_start", "抽穗")
        stage_end = algorithm_config.get("grf_stage_end", "成熟")
        thresholds = {
            "light": {
                "tmax": 32.0,
                "rhum": 30.0,
                "wind": 2.0
            },
            "moderate": {
                "tmax": 33.0,
                "rhum": 25.0,
                "wind": 3.0
            },
            "severe": {
                "tmax": 35.0,
                "rhum": 25.0,
                "wind": 3.0
            }
        }
        weights = {"light": 0.2, "moderate": 0.3, "severe": 0.5}
        marks = {"light": 1.0, "moderate": 2.0, "severe": 3.0}

        station_values = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            gp = growth_map.get(str(sid).strip())
            hd = self._calc_dryhotwind_index(
                daily,
                gp,
                years,
                stage_start=stage_start,
                stage_end=stage_end,
                thresholds=thresholds,
                weights=weights,
                marks=marks,
            )
            station_values[sid] = float(hd) if np.isfinite(hd) else np.nan

        # 插值与栅格归一化
        interp = self._interpolate(station_values, station_coords, cfg, algorithm_config)
        interp['data'] = _normalize_array(interp['data'])
        # 输出路径与中间产品
        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "全国冬小麦干热风指数.tif")
        self._save_geotiff_gdal(interp['data'].astype(np.float32), interp['meta'], tif_path, 0)
        # 分类（可选）
        class_conf = algorithm_config.get('classification', {})
        data_out = interp['data']
        if class_conf:
            method = class_conf.get('method', 'natural_breaks')
            classifier = self._get_algorithm(f"classification.{method}")
            data_out = classifier.execute(interp['data'].astype(float), class_conf)
        # 最终产品
        final_tif = str(out_dir / "全国冬小麦干热风_分级.tif")
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
        """统一入口：根据 element 选择目标计算"""
        config = params['config']
        self._algorithms = params['algorithms']
        d = config.get('element')
        if d == 'GRF':
            return self.calculate_GRF(params)
        raise ValueError(f"不支持的灾害类型: {d}")
