import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
from algorithms.data_manager import DataManager
from osgeo import gdal


class WIWH_BC:
    def __init__(self):
        pass

    def _get_algorithm(self, algorithm_name: str):
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_intermediate_raster(self, result, output_path: Path):
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
        dataset = driver.Create(
            str(output_path),
            meta['width'],
            meta['height'],
            1,
            datatype,
            ['COMPRESS=LZW']
        )
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0)
        band.FlushCache()
        dataset = None

    def _load_intermediate_raster(self, input_path: Path):
        dataset = gdal.Open(str(input_path))
        if dataset is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        transform = dataset.GetGeoTransform()
        crs = dataset.GetProjection()
        meta = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        dataset = None
        return {'data': data, 'meta': meta}

    def _read_phenology_csv(self, csv_path: str) -> pd.DataFrame:
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError("未找到生育期CSV文件")
        try:
            df = pd.read_excel(csv_path, dtype=str)
        except:
            df = pd.read_excel(csv_path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if 'Station_Id_C' not in df.columns:
            col_map = {}
            if '站号' in df.columns:
                col_map['站号'] = 'Station_Id_C'
            if col_map:
                df = df.rename(columns=col_map)
        return df

    def _parse_phase_date(self, text: Any, year: int) -> pd.Timestamp:
        if text is None or pd.isna(text):
            return pd.NaT
        s = str(text).strip()
        base_year = int(year)
        if '上年' in s:
            base_year -= 1
            s = s.replace('上年', '').strip()
        if '月' in s and '日' in s:
            try:
                m_part, d_part = s.split('月', 1)
                d_part = d_part.split('日', 1)[0]
                month = int(m_part.strip())
                day = int(d_part.strip())
                return pd.Timestamp(year=base_year, month=month, day=day)
            except Exception:
                pass
        try:
            return pd.to_datetime(f"{base_year}年{s}")
        except Exception:
            return pd.NaT

    def _count_txb_days(self, daily, sowing_text: Any, mature_text: Any) -> float:
        if daily is None or len(daily) == 0:
            return np.nan
        cnt = 0
        ok_all = (daily['tavg'] >= 9) & (daily['precip'] > 0.1)
        for year in daily.index.year.unique():
            start_date = self._parse_phase_date(sowing_text, int(year))
            end_date = self._parse_phase_date(mature_text, int(year))
            if pd.isna(start_date) or pd.isna(end_date):
                continue
            mask = (daily.index >= start_date) & (daily.index <= end_date)
            cnt += int(ok_all[mask].sum())
        return cnt

    def _compute_station_values(self, station_coords: Dict[str, Any], cfg: Dict[str, Any], algo_cfg: Dict[str, Any]) -> Dict[str, float]:
        csv_path =  cfg.get('growthPeriodPath')
        csv_df = self._read_phenology_csv(csv_path)
        wins_map = csv_df['Station_Id_C'].astype(str)
        wins_set = set(wins_map.tolist())
        sow_col = None
        for c in ['播种', '播种期', 'sowing', 'bozhong']:
            if c in csv_df.columns:
                sow_col = c
                break
        mature_col = None
        for c in ['成熟', '成熟期', 'mature', 'chengshu']:
            if c in csv_df.columns:
                mature_col = c
                break
        if sow_col is None or mature_col is None:
            raise ValueError("生育期CSV中缺少播种或成熟列")
        pheno_by_station: Dict[str, Tuple[Any, Any]] = {}
        for _, row in csv_df.iterrows():
            sid = str(row['Station_Id_C'])
            pheno_by_station[sid] = (row[sow_col], row[mature_col])

        dm = DataManager(cfg.get('inputFilePath'), cfg.get('stationFilePath'), multiprocess=False)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date_cfg = cfg.get('startDate')
        end_date = cfg.get('endDate')
        start_date = start_date_cfg
        if start_date_cfg:
            start_dt = pd.to_datetime(start_date_cfg, format='%Y%m%d', errors='coerce')
            if pd.notna(start_dt):
                has_prev_year = any(('上年' in str(p[0])) for p in pheno_by_station.values())
                if has_prev_year:
                    adj_dt = start_dt - pd.Timedelta(days=366)
                    start_date = adj_dt.strftime('%Y%m%d')
        values: Dict[str, float] = {}
        missing_ids = [sid for sid in station_ids if str(sid) not in wins_set]
        for sid in missing_ids:
            values[str(sid)] = np.nan
        usable_ids = [sid for sid in station_ids if str(sid) in wins_set]
        for sid in usable_ids:
            sid_str = str(sid)
            daily = dm.load_station_data(sid_str, start_date, end_date)
            if len(daily) == 0:
                values[sid_str] = np.nan
                continue
            ph = pheno_by_station.get(sid_str)
            if ph is None:
                values[sid_str] = np.nan
                continue
            sowing_text, mature_text = ph
            yc = self._count_txb_days(daily, sowing_text, mature_text)
            values[sid_str] = yc
        return values

    def _interpolate(self, station_values: Dict[str, float], station_coords: Dict[str, Any], params: Dict[str, Any]):
        cfg = params['config']
        algo_cfg = params['algorithmConfig']
        interp_cfg = algo_cfg.get('interpolation', {})
        method = interp_cfg.get('method', 'idw')
        interpolator = self._get_algorithm(f"interpolation.{method}")
        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': cfg.get("demFilePath", ""),
            'shp_path': cfg.get("shpFilePath", ""),
            'grid_path': cfg.get("gridFilePath", ""),
            'area_code': cfg.get("areaCode", "")
        }
        intermediate_dir = Path(cfg["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        out_tif = intermediate_dir / "intermediate_TXB_days.tif"
        if not os.path.exists(out_tif):
            res = interpolator.execute(data, interp_cfg.get("params", {}))
            self._save_intermediate_raster(res, out_tif)
        else:
            res = self._load_intermediate_raster(out_tif)
        return res

    def _maybe_classify(self, result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        algo_cfg = params.get('algorithmConfig', {})
        cls = algo_cfg.get('classification')
        if not cls:
            return result
        method = cls.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{method}")
        data = classifier.execute(result['data'], cls)
        result['data'] = data
        return result

    def calculate_TXB(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algo_cfg = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        print("开始计算冬小麦赤霉病发病适宜日数并插值")
        station_values = self._compute_station_values(station_coords, cfg, algo_cfg)
        print("站点多年平均适宜日数统计完成")
        try:
            intermediate_dir = Path(cfg["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            out_csv = intermediate_dir / "intermediate_WIWH_TXB.csv"
            df_station = pd.DataFrame(
                [{"Station_Id_C": k, "TXB_days": v} for k, v in station_values.items()]
            )
            df_station.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"站点TXB适宜日数已保存到: {out_csv}")
        except Exception as e:
            print(f"保存站点TXB适宜日数CSV失败: {e}")
        interp_res = self._interpolate(station_values, station_coords, params)
        print("IDW插值完成")
        final_res = self._maybe_classify(interp_res, params)
        return {
            'data': final_res['data'],
            'meta': {
                'width': final_res['meta']['width'],
                'height': final_res['meta']['height'],
                'transform': final_res['meta']['transform'],
                'crs': final_res['meta']['crs']
            }
        }

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cfg = params['config']
        element = cfg.get('element')
        if element == 'TXB':
            return self.calculate_TXB(params)
        raise ValueError(f"不支持的灾害类型: {element}")
