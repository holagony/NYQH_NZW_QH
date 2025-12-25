# from core.base_classes import BaseAlgorithm
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import os
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class IDWInterpolation():
    """反距离权重插值算法 - 基于LSM-IDW的改进版本"""
    
    def execute(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Any:
        """执行IDW插值"""
        if params is None:
            params = {}
        
        try:
            # 提取数据
            station_values = data['station_values']
            station_coords = data['station_coords']
            grid_path = data['grid_path']
            area_code = data['area_code']
            shp_path = data.get('shp_path')
            
            # 获取参数
            var_name = params.get('var_name', 'value')
            min_value = params.get('min_value')
            max_value = params.get('max_value')
            radius_dist = params.get('radius_dist', 2.0)
            min_num = params.get('min_num', 10)
            first_size = params.get('first_size', 200)
            fill_nans = params.get('fill_nans', True)
            grid_nodata = params.get('grid_nodata', 0)
            
            logger.info(f"开始IDW插值，站点数: {len(station_values)}")
            
            # 准备站点数据
            stations_data = self._prepare_station_data(station_values, station_coords, var_name)
            
            if stations_data.empty:
                raise ValueError("没有有效的站点数据进行插值")
            
            logger.info(f"有效站点数据: {len(stations_data)} 条")
            
            # 执行IDW插值
            result = self._idw_interpolation_improved(
                stations_data=stations_data,
                grid_path=grid_path,
                shp_path=shp_path,
                area_code=area_code,
                var_name=var_name,
                radius_dist=radius_dist,
                min_num=min_num,
                first_size=first_size,
                fill_nans=fill_nans,
                grid_nodata=grid_nodata
            )
            
            logger.info("IDW插值完成")
            return result
            
        except Exception as e:
            logger.error(f"IDW插值失败: {str(e)}")
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
    
    def _idw_interpolation_improved(self, stations_data: pd.DataFrame, grid_path: str,
                                  shp_path: str, area_code: str, var_name: str,
                                  radius_dist: float, min_num: int, first_size: int,
                                  fill_nans: bool, grid_nodata: float) -> Dict[str, Any]:
        """改进的IDW插值算法"""
        temp_path = None
        aligned_grid = None
        
        try:
            # 如果提供了矢量文件，先裁剪网格
            aligned_grid = grid_path
            
            # 获取边界范围和行列数
            boundary_range, cols, rows = self._get_cols_rows(aligned_grid)
            
            # 执行IDW插值
            data_values = np.array(stations_data['value'].astype(float))
            lat_data = np.array(stations_data['Lat'].astype(float))
            lon_data = np.array(stations_data['Lon'].astype(float))
            
            # 获取投影信息
            dst_epsg = self._get_epsg(aligned_grid)
            
            idw_ds = self._idw_interpolation_core(
                data=data_values,
                latdata=lat_data,
                londata=lon_data,
                boundary_range=boundary_range,
                dst_epsg=dst_epsg,
                dst_rows=rows,
                dst_cols=cols,
                radius_dist=radius_dist,
                min_num=min_num,
                first_size=first_size,
                mask_grid_path=grid_path,
                grid_nodata=grid_nodata
            )

            result_array = idw_ds.ReadAsArray()
            
            # 填充NaN值
            if fill_nans:
                result_array = self._fill_remaining_nans(result_array, data_values, lat_data, lon_data)
            
            # 获取地理参考信息
            geotransform, projection = self._get_geo_reference(aligned_grid)
            
            return {
                'data': result_array,
                'meta': {
                    'transform': geotransform,
                    'crs': projection,
                    'height': result_array.shape[0],
                    'width': result_array.shape[1],
                    'dtype': result_array.dtype,
                    'nodata': np.nan
                }
            }
            
        except Exception as e:
            logger.error(f"IDW插值核心算法失败: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            if aligned_grid and aligned_grid != grid_path and os.path.exists(aligned_grid):
                try:
                    os.unlink(aligned_grid)
                except:
                    pass
    
    # def _idw_interpolation_core(self, data: np.ndarray, latdata: np.ndarray, londata: np.ndarray,
    #                           boundary_range: tuple, dst_epsg: int, dst_rows: int, dst_cols: int,
    #                           radius_dist: float, min_num: int, first_size: int) -> gdal.Dataset:
    #     """IDW插值核心算法"""
    #     # 调整输出大小以避免内存溢出
    #     if (dst_cols > first_size) or (dst_rows > first_size):
    #         if dst_cols > dst_rows:
    #             cols = first_size
    #             rows = int(np.ceil(dst_rows / (dst_cols / first_size)))
    #         else:
    #             rows = first_size
    #             cols = int(np.ceil(dst_cols / (dst_rows / first_size)))
    #     else:
    #         cols = dst_cols
    #         rows = dst_rows
    #
    #     x_min, y_min, x_max, y_max = boundary_range
    #
    #     # 计算网格单元大小
    #     x_cell = (x_max - x_min) / cols
    #     y_cell = (y_max - y_min) / rows
    #
    #     # 计算网格中心坐标
    #     x_centers = np.linspace(x_min + x_cell/2, x_max - x_cell/2, cols)
    #     y_centers = np.linspace(y_max - y_cell/2, y_min + y_cell/2, rows)
    #
    #     # 创建输出数组
    #     out_data = np.full((rows, cols), np.nan, dtype=np.float32)
    #
    #     # 创建内存数据集
    #     driver = gdal.GetDriverByName("MEM")
    #     out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    #     out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, -y_cell))
    #
    #     # 设置投影
    #     srs = osr.SpatialReference()
    #     srs.ImportFromEPSG(dst_epsg)
    #     out_ds.SetProjection(srs.ExportToWkt())
    #
    #     # 执行IDW插值
    #     search_radius = radius_dist ** 2
    #
    #     logger.info(f"开始IDW插值，网格: {cols}x{rows}, 站点数: {len(data)}")
    #
    #     # 使用向量化计算提高性能
    #     for i in range(rows):
    #         y = y_centers[i]
    #         dy2 = (latdata - y) ** 2
    #
    #         for j in range(cols):
    #             x = x_centers[j]
    #             dx2 = (londata - x) ** 2
    #             distances = dx2 + dy2
    #
    #             # 找到搜索半径内的点
    #             in_radius = distances < search_radius
    #             valid_indices = in_radius & ~np.isnan(data)
    #
    #             valid_count = np.sum(valid_indices)
    #
    #             # 渐进式搜索：如果没有找到足够点，逐步扩大搜索半径
    #             current_radius = search_radius
    #             max_radius_multiplier = 4.0
    #             radius_multiplier = 1.0
    #
    #             while valid_count < min_num and radius_multiplier <= max_radius_multiplier:
    #                 radius_multiplier *= 2.0
    #                 current_radius = search_radius * radius_multiplier
    #                 in_radius = distances < current_radius
    #                 valid_indices = in_radius & ~np.isnan(data)
    #                 valid_count = np.sum(valid_indices)
    #
    #             if valid_count < min_num:
    #                 # 如果仍然不足，选择最近的所有点
    #                 dist_copy = distances.copy()
    #                 dist_copy[~valid_indices] = np.inf
    #                 if np.any(np.isfinite(dist_copy)):
    #                     idx = np.argpartition(dist_copy, min(min_num-1, len(dist_copy)-1))[:min_num]
    #                     valid_indices = np.zeros_like(distances, dtype=bool)
    #                     valid_indices[idx] = True
    #                     valid_count = min_num
    #                 else:
    #                     out_data[i, j] = np.nan
    #                     continue
    #
    #             if valid_count == 0:
    #                 out_data[i, j] = np.nan
    #                 continue
    #
    #             # 提取有效数据
    #             valid_values = data[valid_indices]
    #             valid_distances = distances[valid_indices]
    #
    #             # 检查是否所有值相同
    #             if np.all(valid_values == valid_values[0]):
    #                 out_data[i, j] = valid_values[0]
    #                 continue
    #
    #             # 计算IDW权重
    #             with np.errstate(divide='ignore', invalid='ignore'):
    #                 weights = 1.0 / (valid_distances + 1e-8)
    #
    #                 # 对距离为0的点（同一位置）给予最大权重
    #                 zero_dist_mask = valid_distances == 0
    #                 if np.any(zero_dist_mask):
    #                     weights[zero_dist_mask] = 1e6
    #
    #             weights_sum = np.sum(weights)
    #
    #             if weights_sum > 0:
    #                 idw_value = np.sum(weights * valid_values) / weights_sum
    #                 out_data[i, j] = idw_value
    #             else:
    #                 # 如果权重和仍然为0，使用最近点的值
    #                 min_idx = np.argmin(valid_distances)
    #                 out_data[i, j] = valid_values[min_idx]
    #
    #     # 写入数据
    #     out_band = out_ds.GetRasterBand(1)
    #     out_band.WriteArray(out_data)
    #     out_band.SetNoDataValue(np.nan)
    #
    #     # 重采样到目标尺寸
    #     if cols != dst_cols or rows != dst_rows:
    #         resampled_ds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
    #                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
    #         out_ds = None
    #         logger.info(f"IDW插值完成，重采样从 {cols}x{rows} 到 {dst_cols}x{dst_rows}")
    #         return resampled_ds
    #
    #     logger.info("IDW插值完成")
    #     return out_ds

    def _idw_interpolation_core(self, data: np.ndarray, latdata: np.ndarray, londata: np.ndarray,
                                boundary_range: tuple, dst_epsg: int, dst_rows: int, dst_cols: int,
                                radius_dist: float, min_num: int, first_size: int,
                                mask_grid_path: str = None, grid_nodata: float = -9999) -> gdal.Dataset:
        """IDW插值核心算法 - 包含掩膜边界处理"""
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

        # 计算网格单元大小
        x_cell = (x_max - x_min) / cols
        y_cell = (y_max - y_min) / rows

        # 计算网格中心坐标
        x_centers = np.linspace(x_min + x_cell / 2, x_max - x_cell / 2, cols)
        y_centers = np.linspace(y_max - y_cell / 2, y_min + y_cell / 2, rows)

        # 创建输出数组
        out_data = np.full((rows, cols), np.nan, dtype=np.float32)

        # 创建内存数据集
        driver = gdal.GetDriverByName("MEM")
        out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, -y_cell))

        # 设置投影
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(dst_epsg)
        out_ds.SetProjection(srs.ExportToWkt())

        # 执行IDW插值
        search_radius = radius_dist ** 2

        logger.info(f"开始IDW插值，网格: {cols}x{rows}, 站点数: {len(data)}")

        # 使用向量化计算提高性能
        for i in range(rows):
            y = y_centers[i]
            dy2 = (latdata - y) ** 2

            for j in range(cols):
                x = x_centers[j]
                dx2 = (londata - x) ** 2
                distances = dx2 + dy2

                # 找到搜索半径内的点
                in_radius = distances < search_radius
                valid_indices = in_radius & ~np.isnan(data)

                valid_count = np.sum(valid_indices)

                # 渐进式搜索：如果没有找到足够点，逐步扩大搜索半径
                current_radius = search_radius
                max_radius_multiplier = 4.0
                radius_multiplier = 1.0

                while valid_count < min_num and radius_multiplier <= max_radius_multiplier:
                    radius_multiplier *= 2.0
                    current_radius = search_radius * radius_multiplier
                    in_radius = distances < current_radius
                    valid_indices = in_radius & ~np.isnan(data)
                    valid_count = np.sum(valid_indices)

                if valid_count < min_num:
                    # 如果仍然不足，选择最近的所有点
                    dist_copy = distances.copy()
                    dist_copy[~valid_indices] = np.inf
                    if np.any(np.isfinite(dist_copy)):
                        idx = np.argpartition(dist_copy, min(min_num - 1, len(dist_copy) - 1))[:min_num]
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
                    weights = 1.0 / (valid_distances + 1e-8)

                    # 对距离为0的点（同一位置）给予最大权重
                    zero_dist_mask = valid_distances == 0
                    if np.any(zero_dist_mask):
                        weights[zero_dist_mask] = 1e6

                weights_sum = np.sum(weights)

                if weights_sum > 0:
                    idw_value = np.sum(weights * valid_values) / weights_sum
                    out_data[i, j] = idw_value
                else:
                    # 如果权重和仍然为0，使用最近点的值
                    min_idx = np.argmin(valid_distances)
                    out_data[i, j] = valid_values[min_idx]

        # 写入数据
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(out_data)
        out_band.SetNoDataValue(np.nan)

        # 第一步：重采样到目标尺寸（如果需要）
        if cols != dst_cols or rows != dst_rows:
            resampled_ds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                     resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            out_ds = None
            logger.info(f"IDW插值完成，重采样从 {cols}x{rows} 到 {dst_cols}x{dst_rows}")
            out_ds = resampled_ds

        # 第二步：应用掩膜边界处理
        if mask_grid_path and os.path.exists(mask_grid_path):
            logger.info(f"应用掩膜边界处理: {mask_grid_path}")

            # 读取掩膜栅格数据
            mask_data = gdal.Open(mask_grid_path, gdal.GA_ReadOnly)
            if mask_data:
                try:
                    # 确保掩膜数据与插值数据具有相同的空间参考和尺寸
                    mask_ds = gdal.Warp("", mask_data, format="MEM",
                                        width=dst_cols, height=dst_rows,
                                        outputBounds=boundary_range,
                                        resampleAlg=gdal.GRA_NearestNeighbour)

                    mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
                    result_array = out_ds.GetRasterBand(1).ReadAsArray()

                    # 创建掩膜
                    mask = (mask_array != grid_nodata) & ~np.isnan(result_array)

                    # 应用掩膜
                    result_array[~mask] = np.nan

                    # 更新输出数据集
                    out_band = out_ds.GetRasterBand(1)
                    out_band.WriteArray(result_array)
                    out_band.SetNoDataValue(np.nan)

                    valid_count = np.sum(~np.isnan(result_array))
                    total_count = result_array.size
                    valid_ratio = valid_count / total_count * 100

                    logger.info(f"掩膜应用完成 - 有效点数: {valid_count}/{total_count} ({valid_ratio:.1f}%)")

                    # 清理临时数据集
                    mask_ds = None
                    mask_data = None

                except Exception as e:
                    logger.warning(f"掩膜处理失败: {str(e)}")
            else:
                logger.warning(f"无法打开掩膜文件: {mask_grid_path}")
        else:
            logger.info("未提供掩膜文件，跳过掩膜处理")

        # 第三步：边界裁剪
        logger.info("执行边界裁剪...")

        # 使用Warp进行精确的边界裁剪
        clipped_ds = gdal.Warp("", out_ds, format="MEM",
                               width=dst_cols,
                               height=dst_rows,
                               outputBounds=boundary_range,
                               resampleAlg=gdal.GRA_Bilinear,
                               dstNodata=np.nan)

        # 清理临时数据集
        out_ds = None

        # 验证最终结果
        final_band = clipped_ds.GetRasterBand(1)
        final_array = final_band.ReadAsArray()
        final_valid_count = np.sum(~np.isnan(final_array))
        final_total_count = final_array.size

        logger.info(f"边界处理完成 - 最终有效点数: {final_valid_count}/{final_total_count}")
        logger.info(f"IDW插值全部完成")

        return clipped_ds

    def _fill_remaining_nans(self, grid_data: np.ndarray, station_data: np.ndarray,
                           station_lats: np.ndarray, station_lons: np.ndarray) -> np.ndarray:
        """填充剩余的NaN值"""
        nan_mask = np.isnan(grid_data)
        nan_count = np.sum(nan_mask)
        
        if nan_count == 0:
            return grid_data
        
        logger.info(f"填充 {nan_count} 个剩余NaN像素")
        
        # 方法1: 使用最近邻插值填充
        from scipy.spatial import cKDTree
        
        # 创建有效站点的KD树
        valid_station_mask = ~np.isnan(station_data)
        if np.any(valid_station_mask):
            valid_points = np.column_stack([station_lons[valid_station_mask], 
                                          station_lats[valid_station_mask]])
            valid_values = station_data[valid_station_mask]
            
            if len(valid_points) > 0:
                tree = cKDTree(valid_points)
                
                # 找到所有NaN位置的坐标
                nan_coords = []
                nan_indices = []
                for i in range(grid_data.shape[0]):
                    for j in range(grid_data.shape[1]):
                        if np.isnan(grid_data[i, j]):
                            nan_coords.append([j, i])  # 注意：这里使用像素坐标
                            nan_indices.append((i, j))
                
                if nan_coords:
                    # 将像素坐标转换为地理坐标
                    # 这里需要地理变换信息，简化处理：使用最近邻
                    distances, indices = tree.query(nan_coords, k=1)
                    
                    # 用最近站点的值填充
                    for idx, (i, j) in enumerate(nan_indices):
                        if indices[idx] < len(valid_values):
                            grid_data[i, j] = valid_values[indices[idx]]
        
        # 方法2: 如果还有NaN，使用网格数据的平均值填充
        remaining_nans = np.isnan(grid_data)
        if np.any(remaining_nans) and np.any(~np.isnan(grid_data)):
            mean_value = np.nanmean(grid_data)
            grid_data[remaining_nans] = mean_value
            logger.info(f"使用平均值 {mean_value:.4f} 填充剩余 {np.sum(remaining_nans)} 个NaN")
        
        return grid_data
        
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
