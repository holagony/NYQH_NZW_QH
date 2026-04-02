import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager
from concurrent.futures import ProcessPoolExecutor, as_completed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "test" / "lgb_pz_czf.pkl"


def _compute_suitability_batch_worker(args):
    sids, input_path, station_file_path, start_date, end_date, model_path = args
    dm = DataManager(input_path, station_file_path, multiprocess=False, num_processes=1)
    out_vals = {}
    out_coords = {}
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    calc = SPSO_PZ()
    feature_cols = [
        "经度", "纬度",
        "第一个月平均最低气温", "第2月Tmin", "第3Tmin", "第4Tmin", "第5Tmin",
        "第1月Tmax", "第2Tmax", "第3Tmax", "第4Tmax", "第5Tmax"]
    for sid in sids:
        daily = dm.load_station_data(sid, start_date, end_date)
        info = dm.get_station_info(sid)
        features = calc._build_growing_season_features(daily)
        if features.empty:
            val = np.nan
        else:
            X = features[feature_cols]
            preds = model.predict(X)
            val = float(np.nanmean(preds)) if len(preds) > 0 else np.nan
        out_vals[sid] = val if np.isfinite(val) else np.nan
        out_coords[sid] = {
            "lat": float(info.get("lat", np.nan)),
            "lon": float(info.get("lon", np.nan)),
            "altitude": float(info.get("altitude", np.nan))}
    return out_vals, out_coords


class SPSO_PZ:
    def _get_algorithm(self, algorithm_name):
        """从算法注册器中获取计算组件"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存 GeoTIFF 文件（单波段，LZW 压缩）"""
        # 按 numpy dtype 映射到 GDAL 数据类型
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
        # 创建并写入单波段 GeoTIFF，使用 LZW 压缩
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
        interpolation_params['min_value'] = 0
        interpolation_params['radius_dist'] = 10
        interpolation_params['min_num'] = 5
        interpolation_params['first_size'] = 100

        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        # 组织插值所需输入，包含站点值/坐标、DEM/行政区/规则格网等路径
        grid_path = config.get("gridFilePath", "")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': grid_path,
            'area_code': config.get("areaCode", "")
        }
        result = interpolator.execute(data, interpolation_params)
        return result

    def _normalize_array(self, array):
        if array.size == 0:
            return array
        arr = array.astype(float)
        mask = ~np.isnan(arr)
        if not np.any(mask):
            return arr
        valid = arr[mask]
        mn = float(np.min(valid))
        mx = float(np.max(valid))
        if mx == mn:
            out = np.full_like(arr, 0.5)
        else:
            out = (arr - mn) / (mx - mn)
        out[~mask] = np.nan
        return out

    def _build_growing_season_features(self, daily):
        df = daily.copy()
        if df.empty:
            return pd.DataFrame()
        years = sorted(df.index.year.unique())
        rows = []
        for y in years:
            start_dt = pd.to_datetime(f"{y}-04-01")
            end_dt = pd.to_datetime(f"{y}-09-30")
            sub = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if sub.empty:
                continue
            row = {}
            month_defs = [
                (5, "第一个月平均最低气温", "第1月Tmax"),
                (6, "第2月Tmin", "第2Tmax"),
                (7, "第3Tmin", "第3Tmax"),
                (8, "第4Tmin", "第4Tmax"),
                (9, "第5Tmin", "第5Tmax"),
            ]
            for m, tmin_name, tmax_name in month_defs:
                month_data = sub[sub.index.month == m]
                if month_data.empty:
                    tmin_val = np.nan
                    tmax_val = np.nan
                else:
                    tmin_val = month_data["tmin"].mean() if "tmin" in month_data.columns else np.nan
                    tmax_val = month_data["tmax"].mean() if "tmax" in month_data.columns else np.nan
                row[tmin_name] = tmin_val
                row[tmax_name] = tmax_val
            row["经度"] = float(sub["lon"].iloc[0]) if "lon" in sub.columns else np.nan
            row["纬度"] = float(sub["lat"].iloc[0]) if "lat" in sub.columns else np.nan
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        cols = [
            "经度", "纬度",
            "第一个月平均最低气温", "第2月Tmin", "第3Tmin", "第4Tmin", "第5Tmin",
            "第1月Tmax", "第2Tmax", "第3Tmax", "第4Tmax", "第5Tmax"]
        return pd.DataFrame(rows)[cols]

    def calculate_drought(self, params):
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})

        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False, num_processes=1)
        station_ids = dm.get_all_stations()

        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')
        station_values = {}
        sel_coords = {}
        model_path = str(MODEL_PATH)
        args_common = (cfg.get('inputFilePath'), cfg.get('stationFilePath'), start_date, end_date, model_path)
        max_workers = 16
        chunk_size = max(1, len(station_ids) // (max_workers * 2) + 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in range(0, len(station_ids), chunk_size):
                chunk = station_ids[i:i + chunk_size]
                futures.append(ex.submit(_compute_suitability_batch_worker, (chunk, *args_common)))
            for fut in as_completed(futures):
                vals, coords = fut.result()
                station_values.update(vals)
                if isinstance(station_coords, dict) and station_coords:
                    for sid, coord in coords.items():
                        sel_coords[sid] = station_coords.get(sid, coord)
                else:
                    sel_coords.update(coords)

        interp = self._interpolate(station_values, sel_coords, cfg, algorithm_config)
        interp_data = interp['data'].astype(float)
        interp_data[interp_data < 0] = np.nan
        norm_data = self._normalize_array(interp_data)

        out_dir = Path(cfg.get("resultPath") or os.getcwd())
        out_dir.mkdir(parents=True, exist_ok=True)
        inter_dir = out_dir / "intermediate"
        inter_dir.mkdir(parents=True, exist_ok=True)
        tif_path = str(inter_dir / "粗脂肪指数.tif")
        self._save_geotiff_gdal(interp_data.astype(np.float32), interp['meta'], tif_path, 0)
        norm_tif = str(inter_dir / "粗脂肪指数_归一化.tif")
        self._save_geotiff_gdal(norm_data.astype(np.float32), interp['meta'], norm_tif, 0)

        class_conf = algorithm_config.get('classification', {})
        class_conf = dict(class_conf) if isinstance(class_conf, dict) else {}
        method = class_conf.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{method}")
        data_out = classifier.execute(interp_data.astype(float), class_conf)
        final_tif = str(out_dir / "粗脂肪指数_分级.tif")
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
        if d == 'CZFHL':
            return self.calculate_drought(params)
        raise ValueError(f"不支持的灾害类型: {d}")
