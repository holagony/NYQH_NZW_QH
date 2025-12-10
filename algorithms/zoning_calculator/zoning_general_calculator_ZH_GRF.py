import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import os
import datetime
from algorithms.data_manager import DataManager
from algorithms.indicators import IndicatorCalculator


class ZH_GRF:
    def _normalize_algorithm_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(cfg, dict) and 'GRF' in cfg and isinstance(cfg['GRF'], dict):
            return cfg['GRF']
        return cfg
    def _get_algorithm(self, algorithm_name: str):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name, min_value=np.nan, max_value=np.nan):
        config = params['config']
        algorithm_config = self._normalize_algorithm_config(params['algorithmConfig'])
        interpolation_config = algorithm_config.get("interpolation", {})
        method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{method}")

        print(f"开始插值: {indicator_name}, 方法: {method}")

        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value': min_value,
            'max_value': max_value,
        }

        file_name = f"intermediate_{indicator_name}.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name

        if not os.path.exists(output_path):
            print(f"执行插值计算，输出: {output_path}")
            result = interpolator.execute(data, interpolation_config.get("params", {}))
            self._save_intermediate_raster(result, output_path)
        else:
            print(f"加载已有中间栅格: {output_path}")
            result = self._load_intermediate_raster(output_path)

        return result

    def _perform_classification(self, data_interpolated, params):
        algorithm_config = self._normalize_algorithm_config(params['algorithmConfig'])
        classification = algorithm_config.get('classification', {})
        method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{method}")
        print(f"开始分级，方法: {method}")
        data = classifier.execute(data_interpolated['data'], classification)
        data_interpolated['data'] = data
        return data_interpolated

    def _build_indicator_configs_from_thresholds(self, cfg: Dict[str, Any], start_date: str = "01-01", end_date: str = "12-31") -> Dict[str, Any]:
        print(f"根据阈值生成指标配置，时间窗: {start_date} - {end_date}")
        thresholds = cfg.get('threshold', [])
        thresholds = sorted(thresholds, key=lambda x: x.get('tmax', 0))
        name_map = {1: 'GRF_light', 2: 'GRF_moderate', 3: 'GRF_severe'}
        inds = {}
        for i, th in enumerate(thresholds):
            lvl = th.get('level')
            name = name_map.get(lvl, f"LEVEL_{lvl}")
            tmax_val = th.get('tmax')
            rhummin_val = th.get('rhummin')
            wind_val = th.get('wind')
            conds = []
            if tmax_val is not None:
                conds.append({"variable": "tmax", "operator": ">=", "value": tmax_val})
                if i < len(thresholds) - 1 and thresholds[i+1].get('tmax') is not None:
                    conds.append({"variable": "tmax", "operator": "<", "value": thresholds[i+1]['tmax']})
            if rhummin_val is not None:
                conds.append({"variable": "rhummin", "operator": "<=", "value": rhummin_val})
            if wind_val is not None:
                conds.append({"variable": "wind", "operator": ">=", "value": wind_val})
            print(f"生成指标: {name}, 条件: {conds}")
            inds[name] = {
                "type": "conditional_count",
                "frequency": "yearly",
                "start_date": start_date,
                "end_date": end_date,
                "conditions": conds
            }
        return inds

    def _compute_indicators_from_thresholds(self, station_coords: Dict[str, Any], cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        start_date = cfg.get('start_date', '01-01')
        end_date = cfg.get('end_date', '12-31')
        indicator_configs = self._build_indicator_configs_from_thresholds(cfg, start_date, end_date)
        print(f"开始按阈值计算站点指标，站点数: {len(station_coords)}")
        station_ids = list(station_coords.keys())
        dm = DataManager(config['inputFilePath'], station_file=config.get('stationFilePath'), multiprocess=False)
        data_dict = {}
        start_date = config.get('startDate')
        end_date = config.get('endDate')
        print(f"加载站点数据，时间范围: {start_date} - {end_date}")
        for sid in station_ids:
            try:
                data_dict[sid] = dm.load_station_data(sid, start_date, end_date)
            except Exception:
                data_dict[sid] = pd.DataFrame()
        ic = IndicatorCalculator()
        print("执行指标计算")
        results = ic.calculate_batch(data_dict, indicator_configs)
        print("指标计算完成")
        return results

    def _save_intermediate_raster(self, result, output_path: Path):
        from osgeo import gdal
        data = result['data']
        meta = result['meta']
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(output_path), meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        ds.SetGeoTransform(meta['transform'])
        ds.SetProjection(meta['crs'])
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0)
        band.FlushCache()
        ds = None

    def _load_intermediate_raster(self, input_path: Path):
        from osgeo import gdal
        ds = gdal.Open(str(input_path))
        if ds is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        transform = ds.GetGeoTransform()
        crs = ds.GetProjection()
        meta = {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        ds = None
        return {'data': data, 'meta': meta}

    def _calculate_grf_index(self, station_indicators: pd.DataFrame, algorithm_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[int, float]:
        cfg = self._normalize_algorithm_config(algorithm_config)
        thresholds = cfg.get('threshold', [])

        name_map = {1: 'GRF_light', 2: 'GRF_moderate', 3: 'GRF_severe'}
        weights = {}
        marks = {}
        for th in thresholds:
            lvl = th.get('level')
            w = th.get('weight')
            m = th.get('mark', lvl)
            key = name_map.get(lvl, f"LEVEL_{lvl}")
            if w is not None:
                weights[key] = float(w)
            marks[key] = float(m)

        if not weights:
            weights = {name_map[1]: 0.2, name_map[2]: 0.3, name_map[3]: 0.5}
        if not marks:
            marks = {name_map[1]: 1, name_map[2]: 2, name_map[3]: 3}

        union_keys = set()
        for _, inds in station_indicators.items():
            if isinstance(inds, dict):
                union_keys.update(list(inds.keys()))

        if name_map[1] in union_keys:
            indicators_keys = [name_map[1], name_map[2], name_map[3]]
        elif 'LEVEL_1' in union_keys:
            indicators_keys = ['LEVEL_1', 'LEVEL_2', 'LEVEL_3']
        else:
            indicators_keys = [name_map[1], name_map[2], name_map[3]]

        print(f"计算综合指数，指标: {indicators_keys}")
        print(f"权重: {weights}")
        print(f"标记: {marks}")

        result = {}
        for station_id, indicators in station_indicators.items():
            dfs = []
            cols = []
            for key in indicators_keys:
                val = indicators.get(key, np.nan)
                if isinstance(val, dict):
                    df = pd.DataFrame.from_dict(val, orient='index')
                elif isinstance(val, (int, float)):
                    df = pd.DataFrame({key: [val]})
                else:
                    df = pd.DataFrame({key: [np.nan]})
                dfs.append(df)
                cols.append(key)
            merged = pd.concat(dfs, axis=1)
            merged.columns = cols
            total = 0.0
            for key in cols:
                w = float(weights.get(key, 0.0))
                m = float(marks.get(key, 0.0))
                ni = float(merged[key].mean()) if key in merged.columns else 0.0
                total += w * m * ni
            result[station_id] = total

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"干热风强度指数_{timestamp}.csv"
        df_out = pd.DataFrame(list(result.items()), columns=['站点ID', '干热风强度指数'])
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        out_path = intermediate_dir / filename
        print(f"保存指数到CSV: {out_path}")
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        return result

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        print("开始执行GRF通用计算器")
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithm_config = self._normalize_algorithm_config(params['algorithmConfig'])
        config = params['config']
        self._algorithms = params['algorithms']

        print("准备计算综合指数")
        need_compute = False
        if isinstance(station_indicators, dict):
            if len(station_indicators) == 0:
                need_compute = True
            else:
                sample = next(iter(station_indicators.values()))
                if isinstance(sample, dict) and len(sample) == 0:
                    need_compute = True
        if need_compute:
            print("站点指标为空，按阈值生成并计算站点指标")
            station_indicators = self._compute_indicators_from_thresholds(station_coords, algorithm_config, config)

        indices = self._calculate_grf_index(station_indicators, algorithm_config, config)
        print("综合指数计算完成，开始插值")
        raster = self._perform_interpolation_for_indicator(indices, station_coords, {'config': config, 'algorithmConfig': algorithm_config, 'algorithms': self._algorithms}, "GRF_risk")
        print("插值完成，开始分级")
        final_result = self._perform_classification(raster, {'config': config, 'algorithmConfig': algorithm_config, 'algorithms': self._algorithms})
        print("分级完成")
        return final_result


class WIWH_ZH(ZH_GRF):
    pass
