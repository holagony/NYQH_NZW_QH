# from core.base_classes import BaseAlgorithm
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

class LSMIDWInterpolation():
    """LSM-IDW混合插值算法（最小二乘法+反距离权重）"""
    
    def execute(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Any:
        """执行LSM-IDW插值"""
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
            min_value = params.get('min_value',np.nan)
            max_value = params.get('max_value',np.nan)
            block_size = params.get('block_size', 256)
            radius_dist = params.get('radius_dist', 2.0)
            min_num = params.get('min_num', 20)
            first_size = params.get('first_size', 200)
            grid_nodata = params.get('grid_nodata', 0)
            dem_nodata = params.get('dem_nodata', -32768)
            nodata = params.get('nodata', -32768)
            
            print(f"开始LSM-IDW插值，站点数: {len(station_values)}")
            
            # 准备站点数据
            stations_data = self._prepare_station_data(station_values, station_coords, var_name)
            # 对齐DEM和网格文件
            temp_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),'temp.tif')
            aligned_dem = self._align_datasets(grid_path, dem_path, temp_path)
            
            if stations_data.empty:
                raise ValueError("没有有效的站点数据进行插值")
            
            print(f"有效站点数据: {len(stations_data)} 条")
            
            # 执行LSM-IDW插值
            result = self._lsm_idw_interpolation(
                stations_data=stations_data,
                dem_path=aligned_dem,
                grid_path=grid_path,
                var_name=var_name,
                min_value=min_value,
                max_value=max_value,
                block_size=block_size,
                radius_dist=radius_dist,
                min_num=min_num,
                first_size=first_size,
                grid_nodata=grid_nodata,
                dem_nodata=dem_nodata,
                nodata = nodata
            )
            os.remove(aligned_dem)
            print("LSM-IDW插值完成")
            return result
            
        except Exception as e:
            print(f"LSM-IDW插值失败: {str(e)}")
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
    
    def _lsm_idw_interpolation(self, stations_data: pd.DataFrame, dem_path: str, 
                              grid_path: str, var_name: str, min_value: Optional[float],
                              max_value: Optional[float], block_size: int, radius_dist: float,
                              min_num: int, first_size: int, grid_nodata: float, 
                              dem_nodata: float,nodata:float) -> Dict[str, Any]:
        """执行LSM-IDW混合插值"""
        # 第一步：使用最小二乘法建立多元线性回归模型
        print("建立线性回归模型")
        model, residuals = self._build_linear_model(stations_data)
        
        # 第二步：对残差进行IDW插值
        print("对残差进行IDW插值")
        residuals_grid = self._interpolate_residuals(
            stations_data, residuals, grid_path, dem_path,
            radius_dist, min_num, first_size,nodata
        )
        
        # 第三步：使用模型预测+残差修正得到最终结果
        print("进行模型预测和残差修正")
        final_grid = self._predict_with_model(
            model, dem_path, grid_path, residuals_grid,
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
            residuals = model.resid
            
            print(f"模型R²: {model.rsquared:.4f}")
            return model, residuals
            
        except Exception as e:
            print(f"建立线性模型失败: {str(e)}")
            raise
        
    def _interpolate_residuals(self, stations_data: pd.DataFrame, residuals: pd.Series,
                            grid_path: str, dem_path: str, radius_dist: float,
                            min_num: int, first_size: int,nodata:float) -> np.ndarray:
        """对残差进行IDW插值"""
        # temp_path = None
        # aligned_dem = None
        idw_ds = None
        
        try:
            # # 创建临时文件用于对齐
            # fd, temp_path = tempfile.mkstemp(suffix='.tif')
            # os.close(fd)
            
            # # 对齐DEM和网格文件
            # aligned_dem = self._align_datasets(grid_path, dem_path, temp_path)
            
            # 获取边界范围和行列数
            boundary_range, cols, rows = self._get_cols_rows(dem_path)
            
            # 执行IDW插值
            residuals_data = np.array(residuals.astype(float))
            lat_data = np.array(stations_data['Lat'].astype(float))
            lon_data = np.array(stations_data['Lon'].astype(float))
            
            # 获取投影信息
            dst_epsg = self._get_epsg(grid_path)
            
            idw_ds = self._idw_interpolation(
                data=residuals_data,
                latdata=lat_data,
                londata=lon_data,
                boundary_range=boundary_range,
                dst_epsg=dst_epsg,
                dst_rows=rows,
                dst_cols=cols,
                radius_dist=radius_dist,
                min_num=min_num,
                first_size=first_size,
                nodata=nodata
            )
            
            result = idw_ds.ReadAsArray()
            return result
            
        except Exception as e:
            print(f"残差插值失败: {str(e)}")
            raise
        finally:
            # 清理资源
            if idw_ds:
                idw_ds = None
            # if aligned_dem and aligned_dem != dem_path and os.path.exists(aligned_dem):
            #     try:
            #         os.unlink(aligned_dem)
            #     except:
            #         pass
            # if temp_path and os.path.exists(temp_path):
            #     try:
            #         os.unlink(temp_path)
            #     except:
            #         pass
    
    def _predict_with_model(self, model, dem_path: str, grid_path: str,
                        residuals_grid: np.ndarray, min_value: Optional[float],
                        max_value: Optional[float], block_size: int, 
                        grid_nodata: float, dem_nodata: float) -> np.ndarray:
        """使用模型进行预测并应用残差修正"""
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
            
            # 检查残差网格尺寸是否匹配
            if residuals_grid.shape != (rows, cols):
                raise ValueError(f"残差网格尺寸不匹配: 期望({rows}, {cols}), 实际{residuals_grid.shape}")
            
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
                        # print(f"块({i},{j}): 全部为无效高程值")
                        continue
                    
                    # 修复：正确的坐标计算
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
                        print(f"块({i},{j}): 无有效数据点")
                        continue
                    
                    # print(f"块({i},{j}): 有效点数 {valid_count}/{block_rows*block_cols}")
                    
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
                        if (min_value is not None) and (min_value is not np.nan):
                            y_pred = np.maximum(y_pred, min_value)
                        if (max_value is not None) and (max_value is not np.nan):
                            y_pred = np.minimum(y_pred, max_value)
                        
                        # 创建当前块的预测结果数组
                        pred_block = np.full((block_rows, block_cols), np.nan, dtype=np.float32)
                        pred_flat = np.full(block_data.shape[0], np.nan, dtype=np.float32)
                        pred_flat[valid_mask] = y_pred
                        pred_block = pred_flat.reshape((block_rows, block_cols))
                        
                        # 加上残差修正
                        residuals_block = residuals_grid[row_start:row_end, col_start:col_end]
                        
                        # # 检查残差数据
                        # if np.all(np.isnan(residuals_block)):
                        #     print(f"块({i},{j}): 残差数据全为NaN")
                        
                        final_block = pred_block + residuals_block
                        
                        # 检查最终结果
                        final_valid_count = np.sum(~np.isnan(final_block))
                        # print(f"块({i},{j}): 最终有效点数 {final_valid_count}")
                        
                        # 将结果填入最终网格
                        final_grid[row_start:row_end, col_start:col_end] = final_block
                        
                    except Exception as e:
                        print(f"块({i},{j})处理失败: {str(e)}")
                        continue
            
            # 检查最终网格的有效性
            final_valid = np.sum(~np.isnan(final_grid))
            # print(f"最终网格有效点数: {final_valid}/{final_grid.size}")
            
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
    
    
    def _idw_interpolation(self, data: np.ndarray, latdata: np.ndarray, londata: np.ndarray,
                        boundary_range: tuple, dst_epsg: int, dst_rows: int, dst_cols: int,
                        radius_dist: float, min_num: int, first_size: int,nodata:float) -> gdal.Dataset:
        """反距离权重插值"""
        # 调整输出大小以避免内存溢出
        if (dst_cols > first_size) or (dst_rows > first_size):
            if dst_cols > dst_rows:
                cols = first_size
                rows = int(np.ceil(dst_rows / (dst_cols / first_size)))
            else:
                rows = first_size
                cols = int(np.ceil(dst_cols / (dst_rows / first_size)))
        else:
            cols = dst_cols
            rows = dst_rows
        
        x_min, y_min, x_max, y_max = boundary_range
        
        # 修复：正确计算网格单元大小
        x_cell = (x_max - x_min) / cols
        y_cell = (y_max - y_min) / rows
        
        # 修复：正确计算网格中心坐标
        x_centers = np.linspace(x_min + x_cell/2, x_max - x_cell/2, cols)
        y_centers = np.linspace(y_max - y_cell/2, y_min + y_cell/2, rows)  # 注意：y_max在顶部，y_min在底部
        
        # 创建输出数组
        out_data = np.full((rows, cols), np.nan, dtype=np.float32)
        
        # 创建内存数据集
        driver = gdal.GetDriverByName("MEM")
        out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, -y_cell))  # y_cell为负值
        
        # 设置投影
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(dst_epsg)
        out_ds.SetProjection(srs.ExportToWkt())
        
        # 执行IDW插值
        search_radius = radius_dist ** 2
        
        print(f"开始IDW插值，网格: {cols}x{rows}, 点数: {len(data)}")
        
        # 修复：正确的循环顺序
        for i in range(rows):  # 行循环
            y = y_centers[i]
            for j in range(cols):  # 列循环
                x = x_centers[j]
                
                # 计算距离
                dx2 = (londata - x) ** 2
                dy2 = (latdata - y) ** 2
                distances = dx2 + dy2
                
                # 找到搜索半径内的点
                in_radius = distances < search_radius
                valid_indices = in_radius & ~np.isnan(data)
                
                valid_count = np.sum(valid_indices)
                
                if valid_count < min_num:
                    # 如果点数不足，选择最近的min_num个点
                    dist_copy = distances.copy()
                    dist_copy[~valid_indices] = np.inf
                    if np.any(np.isfinite(dist_copy)):
                        idx = np.argpartition(dist_copy, min(min_num-1, len(dist_copy)-1))[:min_num]
                        valid_indices = np.zeros_like(distances, dtype=bool)
                        valid_indices[idx] = True
                        valid_count = min_num
                    else:
                        out_data[i, j] = np.nan
                        continue
                
                if valid_count == 0:
                    out_data[i, j] = np.nan
                    continue
                
                # 提取有效数据
                valid_values = data[valid_indices]
                valid_distances = distances[valid_indices]
                
                # 检查是否所有值相同
                if np.all(valid_values == valid_values[0]):
                    out_data[i, j] = valid_values[0]
                    continue
                
                # 计算IDW权重
                with np.errstate(divide='ignore', invalid='ignore'):
                    weights = 1.0 / (valid_distances + 1e-12)  # 使用距离平方的倒数
                
                weights_sum = np.sum(weights)
                
                if weights_sum > 0:
                    idw_value = np.sum(weights * valid_values) / weights_sum
                    out_data[i, j] = idw_value
                else:
                    out_data[i, j] = np.nan
        
        # 写入数据
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(out_data)
        out_band.SetNoDataValue(nodata)
        
        # 重采样到目标尺寸
        if cols != dst_cols or rows != dst_rows:
            resampled_ds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            out_ds = None
            print(f"IDW插值完成，重采样从 {cols}x{rows} 到 {dst_cols}x{dst_rows}")
            return resampled_ds
        
        print("IDW插值完成")
        return out_ds
    
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
                # 创建临时文件
                temp_path =  os.path.join(os.path.dirname(os.path.abspath(__file__)),'temp.tif')
                # fd, temp_path = tempfile.mkstemp(suffix='.tif')
                # os.close(fd)
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