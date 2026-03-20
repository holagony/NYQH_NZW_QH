import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
from multiprocessing import Pool
from algorithms.data_manager import DataManager
from osgeo import gdal


class SPSO_ZH:
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

    def _compute_station_values(self, station_coords: Dict[str, Any], cfg: Dict[str, Any], algo_cfg: Dict[str, Any]) -> Dict[str, float]:
        input_path = cfg.get('inputFilePath')
        station_path = cfg.get('stationFilePath')
        dm = DataManager(input_path, station_path, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys()) or dm.get_all_stations()
        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')

        values: Dict[str, float] = {}
        if station_ids:
            args = [
                (sid, input_path, station_path, start_date, end_date)
                for sid in station_ids
            ]
            
            # 使用多进程并行计算以提高速度
            with Pool(16) as pool:
                results = pool.map(_spso_zh_worker, args)
                
            for sid_str, val in results:
                values[sid_str] = val
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
        out_tif = intermediate_dir / "intermediate_SPSO_ZH_days.tif"
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

    def calculate_SPSO_ZH(self, params: Dict[str, Any]) -> Dict[str, Any]:

        #%% 测试用
        cfg = params.get('config', {})
        result_path = cfg.get('resultPath', '')
        params_file = Path(result_path) / "intermediate" / "params_SPSO_ZH_KHQDW.json"
        #%%
        if params_file.exists():
            print(f"从文件加载已保存的params: {params_file}")
            with open(params_file, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
            params['station_coords'] = loaded_params.get('station_coords', {})
            params['algorithmConfig'] = loaded_params.get('algorithmConfig', {})
            params['config'] = loaded_params.get('config', {})
            print("params加载成功")
        
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algo_cfg = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        print("开始计算春大豆低温冷害日数并插值")
        station_values = self._compute_station_values(station_coords, cfg, algo_cfg)
        print("站点多年平均低温冷害日数统计完成")
        try:
            intermediate_dir = Path(cfg["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            out_csv = intermediate_dir / "intermediate_SPSO_ZH_KHQDW.csv"
            df_station = pd.DataFrame(
                [{"Station_Id_C": k, "Cold_Damage_Days": v} for k, v in station_values.items()]
            )
            df_station.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"站点低温冷害日数已保存到: {out_csv}")
        except Exception as e:
            print(f"保存站点CSV失败: {e}")
        
        # 过滤掉无效值后再进行插值
        valid_station_values = {k: v for k, v in station_values.items() if not np.isnan(v)}
        if not valid_station_values:
            print("警告: 所有站点计算结果均为NaN")
            raise ValueError("没有有效的站点冷害计算结果，无法进行插值")
            
        print(f"有效站点数量: {len(valid_station_values)}")
        interp_res = self._interpolate(valid_station_values, station_coords, params)
        print("插值完成")
        final_res = self._maybe_classify(interp_res, params)
        
        # 处理 NaN 值并转换数据类型，防止 TIFF 显示异常
        data = final_res['data']
        # 将 NaN 替换为 255 (NoData)，并将数据转换为 uint8
        if np.issubdtype(data.dtype, np.floating):
            data = np.nan_to_num(data, nan=255)
        data = data.astype(np.uint8)
        
        #%% 测试用：保存params到文件，供下次运行使用
        try:
            result_path = cfg.get('resultPath', '')
            params_file = Path(result_path) / "intermediate" / "params_SPSO_ZH_KHQDW.json"
            params_to_save = {
                'station_coords': station_coords,
                'algorithmConfig': algo_cfg,
                'config': cfg
            }
            params_file.parent.mkdir(parents=True, exist_ok=True)
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
            print(f"params已保存到: {params_file}")
        except Exception as e:
            print(f"保存params失败: {e}")
        #%%
        return {
            'data': data,
            'meta': {
                'width': final_res['meta']['width'],
                'height': final_res['meta']['height'],
                'transform': final_res['meta']['transform'],
                'crs': final_res['meta']['crs']
            }
        }

    def calculate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.calculate_SPSO_ZH(params)


_dm_cache: Dict[Tuple[str, str], DataManager] = {}


def _get_dm(input_path: str, station_path: str) -> DataManager:
    key = (input_path, station_path)
    dm = _dm_cache.get(key)
    if dm is None:
        dm = DataManager(input_path, station_path, multiprocess=False, num_processes=1)
        _dm_cache[key] = dm
    return dm


def _spso_zh_worker(args):
    sid, input_path, station_path, start_date_global, end_date_global = args
    dm = _get_dm(input_path, station_path)
    sid_str = str(sid)
    
    # 获取站点省份信息
    # info格式示例: {'station_name': '北极村', 'station_id': '50137', 'lon': 122.38331, 'lat': 53.46671, 'altitude': 296.0, 'county_code': '232701', 'PAC': '232701', 'county': '漠河市', 'province': '黑龙江省', 'city': '大兴安岭地区', 'PAC_prov': '230000', 'PAC_city': '232700'}
    info = dm.get_station_info(sid_str)
    province = info.get('province', '')
    
    if not province:
        # print(f"站点 {sid_str} 缺少省份信息")
        return sid_str, np.nan
    
    # 定义区域省份列表
    north_provinces = ['内蒙古', '黑龙江', '吉林', '辽宁', '河北', '北京', '山东', '山西', '陕西', '甘肃', '河南']
    south_provinces = ['湖北', '安徽', '江苏', '四川', '重庆', '云南', '贵州', '湖南', '江西', '浙江', '福建', '广东', '广西']
    
    start_md, end_md = None, None
    found = False
    
    # 模糊匹配省份名称
    for p in north_provinces:
        if p in province:
            start_md, end_md = '07-01', '07-20'
            found = True
            break
    if not found:
        for p in south_provinces:
            if p in province:
                start_md, end_md = '06-01', '06-30'
                found = True
                break
    
    if not found:
        # 尝试去掉"省"字再匹配，或者是数据中 province 字段可能包含空格
        province_clean = province.strip().replace('省', '').replace('市', '').replace('自治区', '')
        for p in north_provinces:
            if p in province_clean:
                start_md, end_md = '07-01', '07-20'
                found = True
                break
        if not found:
            for p in south_provinces:
                if p in province_clean:
                    start_md, end_md = '06-01', '06-30'
                    found = True
                    break
    
    if not start_md:
        # print(f"站点 {sid_str} 省份 {province} 未匹配到南北方区域")
        return sid_str, np.nan

    # 加载站点数据
    daily = dm.load_station_data(sid_str, start_date_global, end_date_global)
    if len(daily) == 0:
        # print(f"站点 {sid_str} 无气象数据")
        return sid_str, np.nan

    years = daily.index.year.unique()
    total_cold_damage_days = 0
    valid_years = 0
    
    for year in years:
        try:
            sy = pd.Timestamp(f"{year}-{start_md}")
            ey = pd.Timestamp(f"{year}-{end_md}")
            
            mask = (daily.index >= sy) & (daily.index <= ey)
            sub_df = daily[mask]
            
            if len(sub_df) == 0:
                continue
                
            if 'tavg' not in sub_df.columns:
                continue
                
            tavg = sub_df['tavg']
            
            # 识别低温日数：日平均气温 < 20℃
            cond = (tavg < 20).values
            
            # 统计低温日数
            year_cold_damage_days = np.sum(cond)
            
            total_cold_damage_days += year_cold_damage_days
            valid_years += 1
            
        except Exception:
            continue
        
    if valid_years == 0:
        return sid_str, np.nan
        
    avg_cold_damage_days = total_cold_damage_days / valid_years
    
    return sid_str, float(avg_cold_damage_days)
