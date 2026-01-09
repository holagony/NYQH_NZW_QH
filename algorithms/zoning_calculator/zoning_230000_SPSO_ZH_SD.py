# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:57:35 2025

@author: HTHT
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from math import pi
from algorithms.data_manager import DataManager
from algorithms.interpolation.idw import IDWInterpolation
from algorithms.interpolation.lsm_idw import LSMIDWInterpolation
import os
from osgeo import gdal

def calculate_R(daily_data):
    """
    计算霜冻日数
    返回: R   （0.2D_l+0.3D_m+0.5D_s）年平均轻中重天数

    """
    df = daily_data.copy()
    TMIN = df["tmin"]
    
    # 原有霜冻指标计算
    condition_spring = ((TMIN.index.month == 5) & (TMIN.index.day >= 16)) | ((TMIN.index.month == 6) & (TMIN.index.day <= 14))
    condition_autumn = ((TMIN.index.month == 8) & (TMIN.index.day >= 27)) | ((TMIN.index.month == 9) & (TMIN.index.day <= 22))
    
    frost1 = TMIN[(condition_spring & (TMIN < 2) & (TMIN >= 0)) | (condition_autumn & (TMIN >= 0) & (TMIN < 2))]  # 轻霜冻 
    frost2 = TMIN[(condition_spring & (TMIN < 0) & (TMIN >= (-2))) | (condition_autumn & (TMIN >= (-2)) & (TMIN < 0))]  # 中霜冻
    frost3 = TMIN[(condition_spring & (TMIN < (-2)) & (TMIN >= (-4))) | (condition_autumn & (TMIN >= (-4)) & (TMIN < (-2)))]  # 重霜冻
        
    # 统计霜冻频次
    frost1_num = len(frost1)
    frost2_num = len(frost2)
    frost3_num = len(frost3)
    #年平均霜冻天数
    frost1_mean_d = frost1_num / len(frost1.index.year.unique()) if len(frost1.index.year.unique()) > 0 else 0
    frost2_mean_d = frost2_num / len(frost2.index.year.unique()) if len(frost2.index.year.unique()) > 0 else 0
    frost3_mean_d = frost3_num / len(frost3.index.year.unique()) if len(frost3.index.year.unique()) > 0 else 0
    
    # 计算R
    R = 0.2 * frost1_mean_d + 0.3 * frost2_mean_d + 0.5 * frost3_mean_d
    
    return R

def normalize_grid_values(grid_data: np.ndarray) -> np.ndarray:
    """
    对栅格数据进行归一化到0-1范围
    """
    if grid_data.size == 0:
        return grid_data
    
    # 创建掩码，排除无效值
    valid_mask = ~np.isnan(grid_data)
    
    if not np.any(valid_mask):
        return np.zeros_like(grid_data)
    
    # 获取有效值
    valid_values = grid_data[valid_mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    
    # 初始化结果数组
    normalized_grid = np.zeros_like(grid_data)
    
    # 处理所有值相同的情况
    if max_val == min_val:
        normalized_grid[valid_mask] = 0.5
    else:
        # 归一化计算：最小值→0，最大值→1，中间值按比例计算
        normalized_grid[valid_mask] = (valid_values - min_val) / (max_val - min_val)
    
    # 无效值保持为0
    normalized_grid[~valid_mask] = 0.0
    
    return normalized_grid
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
class SPSO_ZH:
    '''
    黑龙江-大豆-灾害区划
    干旱和霜冻
    干旱--目前是生成危险性G的tif，其他风险区划的样例数据暂未提供
    霜冻
    '''
    def _align_and_read_input(self, grid_path, target_path, result_path):
        '''
        将单个外部栅格对齐到grid_path，并读取为数组
        target_path: 要对齐的目标栅格路径
        result_path: 对齐后的临时文件存储路径
        返回: 对齐后的numpy数组（NoData已置为NaN）
        '''
        base_dir = result_path if result_path else os.path.dirname(grid_path)
        out_dir = os.path.join(base_dir, 'intermediate')
        os.makedirs(out_dir, exist_ok=True)
        temp_path = os.path.join(out_dir, 'align_temp.tif')
        if (not target_path) or (not os.path.exists(target_path)):
            gds = gdal.Open(grid_path)
            rows, cols = gds.RasterYSize, gds.RasterXSize
            gds = None
            return np.zeros((rows, cols), dtype=np.float32)
        aligned_path = LSMIDWInterpolation()._align_datasets(grid_path, target_path, temp_path)
        ds = gdal.Open(aligned_path)
        if ds is None:
            gds = gdal.Open(grid_path)
            rows, cols = gds.RasterYSize, gds.RasterXSize
            gds = None
            if os.path.exists(aligned_path):
                os.remove(aligned_path)
            return np.zeros((rows, cols), dtype=np.float32)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        ds = None
        if os.path.exists(aligned_path):
            os.remove(aligned_path)
        return arr
    def _calculate_frost(self, params):
        """霜冻灾害风险指数模型 - 先插值后归一化"""
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cfg = params.get('config', {})
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')

        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())
        if not station_ids:
            station_ids = dm.get_all_stations()
        available = set(dm.get_all_stations())

        station_ids = [sid for sid in station_ids if sid in available]
        start_date = cfg.get('startDate')
        end_date = cfg.get('endDate')
        #计算致灾因子危险性指数---------------------------------------------------------
        # 第一步：收集所有站点的R值（原始值）
        station_R_values = []
        station_data_map = {}  # 存储站点数据以便后续使用
        
        print("收集所有站点的R值...")
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            station_data_map[sid] = daily
            R = calculate_R(daily)
            station_R_values.append(R)
            print(f"站点 {sid}: R={R:.4f}")
        
        # 第二步：使用原始R值进行插值
        station_values: Dict[str, float] = {}
        for i, sid in enumerate(station_ids):
            R_value = station_R_values[i]
            station_values[sid] = float(R_value) if np.isfinite(R_value) else np.nan
        
        # 插值配置
        interp_conf = algorithm_config.get('interpolation', {})
        method = str(interp_conf.get('method', 'idw')).lower()
        iparams = interp_conf.get('params', {})

        if 'var_name' not in iparams:
            iparams['var_name'] = 'value'

        interp_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'grid_path': cfg.get('gridFilePath'),
            'dem_path': cfg.get('demFilePath'),
            'area_code': cfg.get('areaCode'),
            'shp_path': cfg.get('shpFilePath')
        }

        if method == 'lsm_idw':
            result = LSMIDWInterpolation().execute(interp_data, iparams)
        else:
            result = IDWInterpolation().execute(interp_data, iparams)
        
        # 第三步：对插值后的格点数据进行归一化
        raw_grid_data = result['data']
        print(f"插值后栅格数据统计 - 最小值: {np.nanmin(raw_grid_data):.4f}, 最大值: {np.nanmax(raw_grid_data):.4f}")
        
        # 归一化处理
        normalized_grid = normalize_grid_values(raw_grid_data)
        print(f"归一化后栅格数据统计 - 最小值: {np.nanmin(normalized_grid):.4f}, 最大值: {np.nanmax(normalized_grid):.4f}")
        
        # 第四步：保存归一化后的结果
        result_path = cfg.get("resultPath")
        intermediate_dir = os.path.join(result_path, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        R_tif_path = os.path.join(intermediate_dir, "霜冻致灾因子危险性指数归一化结果.tif")   
        self._save_geotiff(normalized_grid, result['meta'], R_tif_path, 0)
        
        print(f"霜冻致灾因子危险性指数归一化结果已保存至: {R_tif_path}")
        #计算承载体暴露性指数------------------------------------------------------------
        #大豆收获面积
        #ZZ_percent_path = os.path.dirname(interp_data["grid_path"])[:-4] + "/黑龙江大豆收获面积.tif"
        ZZ_percent_path=cfg.get("cropgainFilePath")
        C_array=self._align_and_read_input(interp_data["grid_path"], ZZ_percent_path, cfg.get('resultPath'))

        LC_percent_path =cfg.get("landuseFilePath")
        LC_array=self._align_and_read_input(interp_data["grid_path"], LC_percent_path, cfg.get('resultPath'))

        ###重分类值
        lulc_array= LC_array.copy()
        lulc_array = np.where(LC_array == 1, 3, lulc_array)   #耕地
        lulc_array = np.where((LC_array>1)&(LC_array<=4), 2, lulc_array)    #林地 草地
        lulc_array = np.where((LC_array>4), 1, lulc_array)     #水域 建筑用地 未利用土地
        ###归一化
        lulc_array_norm=normalize_array(lulc_array)
        FCV=0.6*C_array+0.4*lulc_array_norm        
        FCV_tif_path = os.path.join(intermediate_dir, "霜冻承载体暴露性指数.tif")   
        self._save_geotiff(FCV, result['meta'], FCV_tif_path, 0)
        print(f"霜冻承灾体暴露性指数已保存至: {FCV_tif_path}")
        #计算孕灾环境脆弱性指数--------------------------------------------------------------
        #FCV重分类值，进行归一化
        FCV_array= FCV.copy()
        FCV_array = np.where(FCV <0.2, 5, FCV_array)
        FCV_array = np.where((FCV>=0.2)&(FCV<0.4), 4, FCV_array)
        FCV_array = np.where((FCV>=0.4)&(FCV<0.6), 3, FCV_array)
        FCV_array = np.where((FCV>=0.6)&(FCV<0.8), 2, FCV_array)
        FCV_array = np.where((FCV>=0.8)&(FCV<=1), 1, FCV_array)
        FCV_array_norm=normalize_array(FCV_array)
        #地形因子重分类然后归一化
        alti_array_norm=self.calculate_alti(params)
        
        #孕灾环境评价公式
        S=0.5*FCV_array_norm+0.5*alti_array_norm
        #防灾减灾能力计算（人均GDP）---------------------------------------------------------------
        GDP_path = cfg.get("GDPFilePath")
        GDP_array=self._align_and_read_input(interp_data["grid_path"], GDP_path, cfg.get('resultPath'))
 
        ##归一化
        GDP_array_norm=normalize_array(GDP_array)
        
        #综合计算---------------------------------------------------------------------
        ##风险因子危险性、暴露度、脆弱性、防灾减灾能力综合权重比例0.53:0.14:0.26:0.07
        result_array=normalized_grid*0.53+FCV*0.14+S*0.26+GDP_array_norm*0.07
        # 分级--------------------------------------------------------------------------
        class_conf = algorithm_config.get('classification', {})
        key = f"classification.{class_conf.get('method', 'custom_thresholds')}"
        classificator = params.get('algorithms', {})[key]    
        # 执行
        classdata = classificator.execute(result_array, class_conf) 
      
        return {
            'data': classdata,
            'meta': {
                'width': result['meta']['width'],
                'height': result['meta']['height'],
                'transform': result['meta']['transform'],
                'crs': result['meta']['crs']
            },
            'type': '黑龙江大豆霜冻'
        }        
        
        
    def calculate_alti(self,params):
        config = params['config']
        dem_path = config.get("demFilePath", "")
        grid_path = config.get("gridFilePath", "")
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp.tif')
        aligned_dem = LSMIDWInterpolation()._align_datasets(grid_path, dem_path, temp_path)  
        in_ds_dem = gdal.Open(aligned_dem)
        gtt = in_ds_dem.GetGeoTransform()
        alti_array = in_ds_dem.GetRasterBand(1).ReadAsArray()  # 读取波段数据
        Nodata = in_ds_dem.GetRasterBand(1).GetNoDataValue()
        # 海拔
        alti_array = np.where(alti_array == Nodata, np.nan, alti_array)  
        alti_array_=alti_array.copy()
        alti_array_=np.where(alti_array<500,1,alti_array_)
        alti_array_=np.where((alti_array>=500)&(alti_array<1000),2,alti_array_)
        alti_array_=np.where((alti_array>=1000)&(alti_array<2000),3,alti_array_)
        alti_array_=np.where((alti_array>=2000)&(alti_array<4000),4,alti_array_)
        alti_array_=np.where((alti_array>=4000),5,alti_array_)
        alti_array_norm=normalize_array(alti_array_)
        return alti_array_norm

        
    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str, nodata=0):
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
        dataset = driver.Create(
            output_path,
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
        band.SetNoDataValue(nodata)
        
        band.FlushCache()
        dataset = None                      
        
    def calculate(self, params):
        config = params['config']
        disaster_type = config['element']

        if disaster_type == 'SD':
            return self._calculate_frost(params)
        else:
            raise ValueError(f"不支持的灾害类型: {disaster_type}")