
#from core.base_classes import BaseAlgorithm
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from osgeo import gdal, osr
import os
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LSMInterpolation():
    """LSM插值算法（最小二乘法）"""
    
    def execute(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Any:
        """执行LSM插值"""
        if params is None:
            params = {}
        
        try:
            # 提取数据
            station_values = data['station_values']
            station_coords = data['station_coords']
            dem_path = data['dem_path']
            grid_path = data['grid_path']
            
            # 获取参数
            var_name = params.get('var_name', 'value')
            min_value = params.get('min_value', np.nan)
            max_value = params.get('max_value', np.nan)
            block_size = params.get('block_size', 256)
            grid_nodata = params.get('grid_nodata', 0)
            dem_nodata = params.get('dem_nodata', -32768)
            nodata = params.get('nodata', -32768)
            
            print(f"开始LSM插值，站点数: {len(station_values)}")
            
            # 准备站点数据
            stations_data = self._prepare_station_data(station_values, station_coords, var_name)
            
            # 对齐DEM和网格文件
            temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_lsm.tif')
            aligned_dem = self._align_datasets(grid_path, dem_path, temp_path)
            
            if stations_data.empty:
                raise ValueError("没有有效的站点数据进行插值")
            
            print(f"有效站点数据: {len(stations_data)} 条")
            
            # 执行LSM插值
            result = self._lsm_interpolation(
                stations_data=stations_data,
                dem_path=aligned_dem,
                grid_path=grid_path,
                var_name=var_name,
                min_value=min_value,
                max_value=max_value,
                block_size=block_size,
                grid_nodata=grid_nodata,
                dem_nodata=dem_nodata,
                nodata=nodata
            )
            
            # 清理临时文件
            if os.path.exists(aligned_dem):
                os.remove(aligned_dem)
            
            print("LSM插值完成")
            return result
            
        except Exception as e:
            print(f"LSM插值失败: {str(e)}")
            raise
    
    def _prepare_station_data(self, station_values: Dict[str, Any], 
                            station_coords: Dict[str, Any], 
                            var_name: str) -> pd.DataFrame:
        """准备站点数据"""
        data_list = []
        
        for station_id, values in station_values.items():
            if station_id in station_coords:
                coords = station_coords[station_id]
                
                # 获取指标值
                if isinstance(values, dict):
                    value = values.get(var_name, np.nan)
                else:
                    value = values
                
                # 检查坐标和值是否有效
                lon = coords.get('lon', np.nan)
                lat = coords.get('lat', np.nan)
                alti = coords.get('altitude', np.nan)
                
                if (not np.isnan(value) and not np.isnan(lon) and 
                    not np.isnan(lat) and not np.isnan(alti)):
                    data_list.append({
                        'station_id': station_id,
                        'Lon': lon,
                        'Lat': lat,
                        'Alti': alti,
                        'value': value
                    })
        
        return pd.DataFrame(data_list)
    
    def _lsm_interpolation(self, stations_data: pd.DataFrame, dem_path: str, 
                          grid_path: str, var_name: str, min_value: Optional[float],
                          max_value: Optional[float], block_size: int, 
                          grid_nodata: float, dem_nodata: float, nodata: float) -> Dict[str, Any]:
        """执行LSM插值"""
        # 第一步：使用最小二乘法建立多元线性回归模型
        print("建立线性回归模型")
        model = self._build_linear_model(stations_data)
        
        # 第二步：使用模型直接预测得到最终结果
        print("进行模型预测")
        final_grid = self._predict_with_model(
            model, dem_path, grid_path,
            min_value, max_value, block_size, grid_nodata, dem_nodata
        )
        
        # 获取地理参考信息
        geotransform, projection = self._get_geo_reference(grid_path)
        
        return {
            'data': final_grid,
            'meta': {
                'transform': geotransform,
                'crs': projection,
                'height': final_grid.shape[0],
                'width': final_grid.shape[1],
                'dtype': final_grid.dtype,
                'nodata': nodata
            }
        }
    
    def _build_linear_model(self, data: pd.DataFrame):
        """建立多元线性回归模型"""
        try:
            X = data[['Lon', 'Lat', 'Alti']].values
            y = data['value'].values
            
            # 检查数据有效性
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                raise ValueError("输入数据包含NaN值")
            
            # 添加常数项
            X = sm.add_constant(X)
            
            # 拟合线性模型
            model = sm.OLS(y, X).fit()
            
            print(f"模型R²: {model.rsquared:.4f}")
            print(f"模型系数: {model.params}")
            print(f"模型P值: {model.pvalues}")
            
            return model
            
        except Exception as e:
            print(f"建立线性模型失败: {str(e)}")
            raise
    
    def _predict_with_model(self, model, dem_path: str, grid_path: str,
                          min_value: Optional[float], max_value: Optional[float], 
                          block_size: int, grid_nodata: float, dem_nodata: float) -> np.ndarray:
        """使用模型进行预测"""
        dataset = None
        grid_data = None
        
        try:
            dataset = gdal.Open(dem_path, gdal.GA_ReadOnly)
            if dataset is None:
                raise ValueError(f"无法打开DEM文件: {dem_path}")
            
            # 获取地理信息
            geotransform = dataset.GetGeoTransform()
            rows, cols = dataset.RasterYSize, dataset.RasterXSize
            
            print(f"DEM尺寸: {cols}x{rows}, 地理变换: {geotransform}")
            
            # 初始化结果数组
            final_grid = np.full((rows, cols), np.nan, dtype=np.float32)
            
            print(f"开始分块预测，网格大小: {cols}x{rows}, 块大小: {block_size}")
            
            # 分块处理
            for i in range(0, rows, block_size):
                for j in range(0, cols, block_size):
                    row_start = i
                    row_end = min(i + block_size, rows)
                    col_start = j
                    col_end = min(j + block_size, cols)
                    
                    block_rows = row_end - row_start
                    block_cols = col_end - col_start
                    
                    # 读取当前块的高程数据
                    dem_block = dataset.GetRasterBand(1).ReadAsArray(
                        col_start, row_start, block_cols, block_rows
                    )
                    
                    if dem_block is None or dem_block.size == 0:
                        print(f"块({i},{j}): 无法读取DEM数据")
                        continue
                    
                    # 检查是否全部为无效值
                    dem_block_valid = ~((dem_block == dem_nodata) | np.isnan(dem_block))
                    if not np.any(dem_block_valid):
                        continue
                    
                    # 计算每个像素的中心坐标
                    x_coords = np.zeros(block_cols)
                    y_coords = np.zeros(block_rows)
                    
                    for col in range(block_cols):
                        x_coords[col] = (geotransform[0] + 
                                    (col_start + col + 0.5) * geotransform[1] + 
                                    (row_start + 0.5) * geotransform[2])
                    
                    for row in range(block_rows):
                        y_coords[row] = (geotransform[3] + 
                                    (col_start + 0.5) * geotransform[4] + 
                                    (row_start + row + 0.5) * geotransform[5])
                    
                    # 创建坐标网格
                    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
                    
                    # 准备预测数据
                    block_data = pd.DataFrame({
                        'Lon': lon_grid.flatten(),
                        'Lat': lat_grid.flatten(),
                        'Alti': dem_block.flatten()
                    })
                    
                    # 过滤无效值
                    valid_mask = (~np.isnan(block_data['Alti'])) & (block_data['Alti'] != dem_nodata)
                    valid_count = np.sum(valid_mask)
                    
                    if valid_count == 0:
                        continue
                    
                    valid_data = block_data[valid_mask].copy()
                    
                    try:
                        # 添加常数项并进行预测
                        X_pred = sm.add_constant(valid_data[['Lon', 'Lat', 'Alti']], has_constant='add')
                        y_pred = model.predict(X_pred)
                        
                        # 检查预测结果
                        if np.all(np.isnan(y_pred)):
                            print(f"块({i},{j}): 模型预测结果全为NaN")
                            continue
                        
                        # 应用值域限制
                        if (min_value is not None) and (not np.isnan(min_value)):
                            y_pred = np.maximum(y_pred, min_value)
                        if (max_value is not None) and (not np.isnan(max_value)):
                            y_pred = np.minimum(y_pred, max_value)
                        
                        # 创建当前块的预测结果数组
                        pred_block = np.full((block_rows, block_cols), np.nan, dtype=np.float32)
                        pred_flat = np.full(block_data.shape[0], np.nan, dtype=np.float32)
                        pred_flat[valid_mask] = y_pred
                        pred_block = pred_flat.reshape((block_rows, block_cols))
                        
                        # 检查最终结果
                        final_valid_count = np.sum(~np.isnan(pred_block))
                        
                        # 将结果填入最终网格
                        final_grid[row_start:row_end, col_start:col_end] = pred_block
                        
                    except Exception as e:
                        print(f"块({i},{j})处理失败: {str(e)}")
                        continue
            
            # 检查最终网格的有效性
            final_valid = np.sum(~np.isnan(final_grid))
            print(f"最终网格有效点数: {final_valid}/{final_grid.size}")
            
            # 应用掩膜
            grid_data = gdal.Open(grid_path, gdal.GA_ReadOnly)
            if grid_data:
                grid_array = grid_data.GetRasterBand(1).ReadAsArray()
                dem_array = dataset.GetRasterBand(1).ReadAsArray()
                
                # 创建掩膜
                mask = (grid_array != grid_nodata) & (dem_array != dem_nodata) & ~np.isnan(dem_array)
                final_grid[~mask] = np.nan
                
                print(f"应用掩膜后有效点数: {np.sum(~np.isnan(final_grid))}")
            
            print("模型预测完成")
            return final_grid
            
        except Exception as e:
            print(f"模型预测失败: {str(e)}")
            raise
        finally:
            # 清理资源
            if dataset:
                dataset = None
            if grid_data:
                grid_data = None
    
    def _align_datasets(self, reference_path: str, source_path: str, 
                       output_path: Optional[str] = None) -> str:
        """对齐数据集到参考栅格"""
        ref_ds = None
        src_ds = None
        
        try:
            ref_ds = gdal.Open(reference_path)
            src_ds = gdal.Open(source_path)
            
            if ref_ds is None:
                raise ValueError(f"无法打开参考数据集: {reference_path}")
            if src_ds is None:
                raise ValueError(f"无法打开源数据集: {source_path}")
            
            if output_path is None:
                temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_lsm.tif')
            else:
                temp_path = output_path
            
            # 执行对齐
            gdal.Warp(
                temp_path,
                src_ds,
                format="GTiff",
                width=ref_ds.RasterXSize,
                height=ref_ds.RasterYSize,
                dstSRS=ref_ds.GetProjection(),
                outputBounds=self._get_bounds(ref_ds),
                resampleAlg=gdal.GRA_NearestNeighbour
            )
            
            return temp_path
            
        finally:
            if ref_ds:
                ref_ds = None
            if src_ds:
                src_ds = None
    
    def _get_cols_rows(self, dataset_path: str) -> tuple:
        """获取栅格的行列数和边界范围"""
        ds = gdal.Open(dataset_path)
        if ds is None:
            raise ValueError(f"无法打开数据集: {dataset_path}")
        
        try:
            geo = ds.GetGeoTransform()
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            
            x_min = geo[0]
            x_max = geo[0] + cols * geo[1]
            y_max = geo[3]
            y_min = geo[3] + rows * geo[5]
            
            return (x_min, y_min, x_max, y_max), cols, rows
        finally:
            ds = None
    
    def _get_epsg(self, dataset_path: str) -> int:
        """获取数据集的EPSG代码"""
        ds = gdal.Open(dataset_path)
        if ds is None:
            return 4326  # 默认WGS84
        
        try:
            proj = ds.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj)
            
            # 尝试获取EPSG代码
            try:
                if srs.IsProjected():
                    code = srs.GetAuthorityCode("PROJCS")
                else:
                    code = srs.GetAuthorityCode("GEOGCS")
                
                if code:
                    return int(code)
            except:
                pass
            
            return 4326  # 默认WGS84
        finally:
            ds = None
    
    def _get_bounds(self, dataset: gdal.Dataset) -> tuple:
        """获取数据集的边界范围"""
        geo = dataset.GetGeoTransform()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        
        x_min = geo[0]
        x_max = geo[0] + cols * geo[1]
        y_max = geo[3]
        y_min = geo[3] + rows * geo[5]
        
        return (x_min, y_min, x_max, y_max)
    
    def _get_geo_reference(self, dataset_path: str) -> tuple:
        """获取地理参考信息"""
        ds = gdal.Open(dataset_path)
        if ds is None:
            raise ValueError(f"无法打开数据集: {dataset_path}")
        
        try:
            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()
            return geotransform, projection
        finally:
            ds = None