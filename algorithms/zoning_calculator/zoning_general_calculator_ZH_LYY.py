import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import os
import datetime
from algorithms.data_manager import DataManager
from algorithms.indicators import IndicatorCalculator


class ZH_LYY:
    def _normalize_algorithm_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(cfg, dict):
            for key in ['LCY', 'GRF']:
                if key in cfg and isinstance(cfg[key], dict):
                    return cfg[key]
        return cfg

    def _get_algorithm(self, algorithm_name: str):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _perform_interpolation_for_indicator(self, station_values, station_coords, params, indicator_name):
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
            'area_code': config.get("areaCode", "")
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

    def _build_indicator_configs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        start_date = cfg.get('start_date', '05-20')
        end_date = cfg.get('end_date', '06-10')
        print(f"生成LCY基础指标配置，时间窗: {start_date} - {end_date}")
        return {
            'Pre': {
                'type': 'period_sum',
                'frequency': 'yearly',
                'start_date': start_date,
                'end_date': end_date,
                'variable': 'precip'
            },
            'SSH': {
                'type': 'period_sum',
                'frequency': 'yearly',
                'start_date': start_date,
                'end_date': end_date,
                'variable': 'sunshine'
            },
            'Pre_days': {
                'type': 'conditional_count',
                'frequency': 'yearly',
                'start_date': start_date,
                'end_date': end_date,
                'conditions': [{ 'variable': 'precip', 'operator': '>=', 'value': 0.1 }]
            }
        }

    def _compute_base_indicators(self, station_coords: Dict[str, Any], cfg: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        indicator_configs = self._build_indicator_configs(cfg)
        station_ids = list(station_coords.keys())
        print(f"开始按LCY基础指标计算，站点数: {len(station_ids)}")
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
        print("执行LCY基础指标计算")
        results = ic.calculate_batch(data_dict, indicator_configs)
        print("LCY基础指标计算完成")
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

    def _apply_categories(self, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        def f(v):
            try:
                return float(v)
            except Exception:
                return np.nan
        ths = cfg.get('threshold', []) if isinstance(cfg, dict) else []
        pre_th = {}
        ssh_th = {}
        pday_th = {}
        for t in ths:
            lvl = t.get('level')
            if lvl is None:
                continue
            pre_th[int(lvl)] = f(t.get('precip'))
            ssh_th[int(lvl)] = f(t.get('sunshine'))
            pday_th[int(lvl)] = f(t.get('pre_days'))

        def pre_category(x):
            if all(k in pre_th for k in [1, 2, 3]):
                a1, a2, a3 = pre_th[1], pre_th[2], pre_th[3]
                if x < a1:
                    return 0
                elif a1 <= x <= a2:
                    return 1
                elif x > a2 and x <= a3:
                    return 2
                elif x > a3:
                    return 3
                else:
                    return np.nan
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
            if all(k in pday_th for k in [1, 2, 3]):
                b1, b2, b3 = pday_th[1], pday_th[2], pday_th[3]
                if x < b1:
                    return 0
                elif b1 <= x <= b2:
                    return 1
                elif x > b2 and x <= b3:
                    return 2
                elif x > b3:
                    return 3
                else:
                    return np.nan
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
            if all(k in ssh_th for k in [1, 2, 3]):
                s1, s2, s3 = ssh_th[1], ssh_th[2], ssh_th[3]
                if x > s1:
                    return 0
                elif x >= s2 and x <= s1:
                    return 1
                elif x >= s3 and x < s2:
                    return 2
                elif x < s3:
                    return 3
                else:
                    return np.nan
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

        df['Pre'] = df['Pre'].apply(pre_category)
        df['SSH'] = df['SSH'].apply(ssh_category)
        df['Pre_days'] = df['Pre_days'].apply(pre_days_category)
        return df.dropna()

    def _level_index(self, x: float, cfg: Dict[str, Any]) -> int:
        ths = cfg.get('threshold', []) if isinstance(cfg, dict) else []
        m = {}
        for t in ths:
            lvl = t.get('level')
            iy = t.get('indexyear')
            if lvl in (1, 2, 3) and iy is not None:
                try:
                    m[int(lvl)] = float(iy)
                except Exception:
                    pass
        if all(k in m for k in [1, 2, 3]):
            a, b, c = m[1], m[2], m[3]
            if x <= (a - 1):
                return 0
            elif a <= x <= b:
                return 1
            elif b < x <= c:
                return 2
            elif x > c:
                return 3
            else:
                return 0
        if x <= 2:
            return 0
        elif 3 <= x <= 4:
            return 1
        elif 5 <= x <= 6:
            return 2
        elif x > 6:
            return 3
        else:
            return 0

    def _calculate_lcy_index(self, station_indicators: Dict[str, Any], algorithm_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        print("开始计算LCY综合指数")
        # 从配置阈值读取等级权重（参考GRF格式）
        weights_cfg = {}
        for th in algorithm_config.get('threshold', []):
            lvl = th.get('level')
            w = th.get('weight')
            if lvl in (1, 2, 3) and w is not None:
                weights_cfg[int(lvl)] = float(w)
        if not weights_cfg:
            weights_cfg = {1: 0.2, 2: 0.3, 3: 0.5}
        result = {}
        all_raw = pd.DataFrame()
        for station_id, inds in station_indicators.items():
            pre = inds.get('Pre', np.nan)
            ssh = inds.get('SSH', np.nan)
            pre_days = inds.get('Pre_days', np.nan)
            if isinstance(pre, dict):
                pre_df = pd.DataFrame.from_dict(pre, orient='index')
            else:
                pre_df = pd.DataFrame({'Pre': [pre]})
            if isinstance(ssh, dict):
                ssh_df = pd.DataFrame.from_dict(ssh, orient='index')
            else:
                ssh_df = pd.DataFrame({'SSH': [ssh]})
            if isinstance(pre_days, dict):
                pre_days_df = pd.DataFrame.from_dict(pre_days, orient='index')
            else:
                pre_days_df = pd.DataFrame({'Pre_days': [pre_days]})

            merged_df = pd.concat([pre_df, ssh_df, pre_days_df], axis=1)
            merged_df.columns = ['Pre', 'SSH', 'Pre_days']
            raw_copy = merged_df.copy()
            raw_copy['站点ID'] = station_id
            all_raw = pd.concat([all_raw, raw_copy])

            cleaned = self._apply_categories(merged_df, algorithm_config)
            if cleaned.empty:
                result[station_id] = np.nan
                continue
            cleaned['年度指数'] = cleaned['Pre'] + cleaned['SSH'] + cleaned['Pre_days']
            cleaned['连阴雨程度'] = cleaned['年度指数'].apply(lambda v: self._level_index(v, algorithm_config))
            freq = cleaned['连阴雨程度'].value_counts().sort_index()
            for lv in [0, 1, 2, 3]:
                if lv not in freq:
                    freq[lv] = 0
            weighted = weights_cfg[3] * freq.get(3, 0) + weights_cfg[2] * freq.get(2, 0) + weights_cfg[1] * freq.get(1, 0)
            result[station_id] = float(weighted) / float(len(cleaned))

        if result:
            max_val = max(result.values())
            min_val = min(result.values())
            max_keys = [k for k, v in result.items() if v == max_val]
            min_keys = [k for k, v in result.items() if v == min_val]
            print(f"LCY：单站最高综合指数：{max_keys}：{max_val}")
            print(f"LCY：单站最低综合指数：{min_keys}：{min_val}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"连阴雨综合指数_{timestamp}.csv"
        df_out = pd.DataFrame(list(result.items()), columns=['站点ID', '连阴雨综合指数'])
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        out_path = intermediate_dir / filename
        print(f"保存指数到CSV: {out_path}")
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        return result

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        print("开始执行LCY通用计算器")
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithm_config = self._normalize_algorithm_config(params['algorithmConfig'])
        config = params['config']
        self._algorithms = params['algorithms']

        need_compute = False
        if isinstance(station_indicators, dict):
            if len(station_indicators) == 0:
                need_compute = True
            else:
                sample = next(iter(station_indicators.values()))
                if isinstance(sample, dict) and len(sample) == 0:
                    need_compute = True
        if need_compute:
            print("站点指标为空，生成LCY基础指标")
            station_indicators = self._compute_base_indicators(station_coords, algorithm_config, config)

        indices = self._calculate_lcy_index(station_indicators, algorithm_config, config)
        print("综合指数计算完成，开始插值")
        raster = self._perform_interpolation_for_indicator(indices, station_coords, {'config': config, 'algorithmConfig': algorithm_config, 'algorithms': self._algorithms}, "LCY_risk")
        print("插值完成，开始分级")
        final_result = self._perform_classification(raster, {'config': config, 'algorithmConfig': algorithm_config, 'algorithms': self._algorithms})
        print("分级完成")
        return final_result


# class WIWH_ZH(ZH_LCY):
#     pass
