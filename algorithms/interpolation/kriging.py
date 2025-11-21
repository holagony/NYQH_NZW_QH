from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import os
import tempfile
from pathlib import Path
import logging
from pykrige.ok import OrdinaryKriging

logger = logging.getLogger(__name__)

class KrigingInterpolation():
    """克里金插值算法 - 基于PyKrige的实现"""
    
    def execute(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Any:
        """执行克里金插值"""
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
            variogram_model = params.get('variogram_model', 'spherical')
            coordinates_type = params.get('coordinates_type', 'geographic')
            nlags = params.get('nlags', 6)
            
            logger.info(f"开始克里金插值，站点数: {len(station_values)}")
            
            # 准备站点数据
            stations_data = self._prepare_station_data(station_values, station_coords, var_name)
            
            if stations_data.empty:
                raise ValueError("没有有效的站点数据进行插值")
            
            logger.info(f"有效站点数据: {len(stations_data)} 条")
            
            # 执行克里金插值
            result = self._kriging_interpolation(
                stations_data=stations_data,
                grid_path=grid_path,
                shp_path=shp_path,
                area_code=area_code,
                var_name=var_name,
                radius_dist=radius_dist,
                min_num=min_num,
                first_size=first_size,
                fill_nans=fill_nans,
                min_value=min_value,
                max_value=max_value,
                variogram_model=variogram_model,
                coordinates_type=coordinates_type,
                nlags=nlags
            )
            
            logger.info("克里金插值完成")
            return result
            
        except Exception as e:
            logger.error(f"克里金插值失败: {str(e)}")
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
    
    def _kriging_interpolation(self, stations_data: pd.DataFrame, grid_path: str,
                             shp_path: str, area_code: str, var_name: str,
                             radius_dist: float, min_num: int, first_size: int,
                             fill_nans: bool, min_value: Optional[float], 
                             max_value: Optional[float], variogram_model: str,
                             coordinates_type: str, nlags: int) -> Dict[str, Any]:
        """克里金插值主函数"""
        temp_path = None
        aligned_grid = None
        
        try:
            # 如果提供了矢量文件，先裁剪网格
            aligned_grid = grid_path
            
            # 获取边界范围和行列数
            boundary_range, cols, rows = self._get_cols_rows(aligned_grid)
            
            # 执行克里金插值
            data_values = np.array(stations_data['value'].astype(float))
            lat_data = np.array(stations_data['Lat'].astype(float))
            lon_data = np.array(stations_data['Lon'].astype(float))
            
            # 获取投影信息
            dst_epsg = self._get_epsg(aligned_grid)
            
            krige_ds = self._kriging_interpolation_core(
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
                min_value=min_value,
                max_value=max_value,
                variogram_model=variogram_model,
                coordinates_type=coordinates_type,
                nlags=nlags
            )

            result_array = krige_ds.ReadAsArray()
            
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
            logger.error(f"克里金插值核心算法失败: {str(e)}")
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
    
    def _kriging_interpolation_core(self, data: np.ndarray, latdata: np.ndarray, londata: np.ndarray,
                                  boundary_range: tuple, dst_epsg: int, dst_rows: int, dst_cols: int,
                                  radius_dist: float, min_num: int, first_size: int,
                                  min_value: Optional[float], max_value: Optional[float],
                                  variogram_model: str, coordinates_type: str, nlags: int) -> gdal.Dataset:
        """克里金插值核心算法"""
        
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
        x_centers = np.linspace(x_min + x_cell/2, x_max - x_cell/2, cols)
        y_centers = np.linspace(y_max - y_cell/2, y_min + y_cell/2, rows)

        logger.info(f"开始克里金插值，网格: {cols}x{rows}, 站点数: {len(data)}")
        logger.info(f"变差函数模型: {variogram_model}, 坐标类型: {coordinates_type}")

        try:
            # 创建克里金插值模型
            OK = OrdinaryKriging(
                londata,
                latdata,
                data,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
                coordinates_type=coordinates_type,
                nlags=nlags,
                weight=True
            )

            # 执行克里金插值
            outdata, ss = OK.execute('grid', x_centers, y_centers)

            # 应用上下限值
            if min_value is not None:
                outdata[outdata < min_value] = min_value
            if max_value is not None:
                outdata[outdata > max_value] = max_value

            # 创建输出数据集
            driver = gdal.GetDriverByName("MEM")
            out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, -y_cell))

            # 设置投影
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(srs.ExportToWkt())

            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outdata)
            out_band.SetNoDataValue(np.nan)

            # 重采样到目标尺寸
            if cols != dst_cols or rows != dst_rows:
                resampled_ds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                       resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
                out_ds = None
                logger.info(f"克里金插值完成，重采样从 {cols}x{rows} 到 {dst_cols}x{dst_rows}")
                return resampled_ds

            logger.info("克里金插值完成")
            return out_ds

        except Exception as e:
            logger.error(f"克里金插值过程出错: {str(e)}")
            # 如果克里金插值失败，回退到IDW插值
            logger.info("尝试使用IDW插值作为备选方案")
            return self._fallback_idw_interpolation(
                data, latdata, londata, boundary_range, dst_epsg, 
                dst_rows, dst_cols, radius_dist, min_num, first_size
            )
    
    def _fallback_idw_interpolation(self, data: np.ndarray, latdata: np.ndarray, londata: np.ndarray,
                                  boundary_range: tuple, dst_epsg: int, dst_rows: int, dst_cols: int,
                                  radius_dist: float, min_num: int, first_size: int) -> gdal.Dataset:
        """备用的IDW插值方法（当克里金插值失败时使用）"""
        logger.info("使用IDW插值作为备选方案")
        
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
        x_centers = np.linspace(x_min + x_cell/2, x_max - x_cell/2, cols)
        y_centers = np.linspace(y_max - y_cell/2, y_min + y_cell/2, rows)

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

                # 渐进式搜索
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

                # 计算IDW权重
                with np.errstate(divide='ignore', invalid='ignore'):
                    weights = 1.0 / (valid_distances + 1e-8)

                weights_sum = np.sum(weights)

                if weights_sum > 0:
                    idw_value = np.sum(weights * valid_values) / weights_sum
                    out_data[i, j] = idw_value
                else:
                    min_idx = np.argmin(valid_distances)
                    out_data[i, j] = valid_values[min_idx]

        # 写入数据
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(out_data)
        out_band.SetNoDataValue(np.nan)

        # 重采样到目标尺寸
        if cols != dst_cols or rows != dst_rows:
            resampled_ds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                   resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            out_ds = None
            return resampled_ds

        return out_ds
    
    def _fill_remaining_nans(self, grid_data: np.ndarray, station_data: np.ndarray,
                           station_lats: np.ndarray, station_lons: np.ndarray) -> np.ndarray:
        """填充剩余的NaN值"""
        nan_mask = np.isnan(grid_data)
        nan_count = np.sum(nan_mask)
        
        if nan_count == 0:
            return grid_data
        
        logger.info(f"填充 {nan_count} 个剩余NaN像素")
        
        # 使用最近邻插值填充
        from scipy.spatial import cKDTree
        
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
                rows, cols = grid_data.shape
                x_min, y_min, x_max, y_max = 0, 0, cols, rows  # 像素坐标范围
                
                for i in range(rows):
                    for j in range(cols):
                        if np.isnan(grid_data[i, j]):
                            # 转换为相对坐标进行最近邻搜索
                            rel_x = j / cols
                            rel_y = i / rows
                            nan_coords.append([rel_x, rel_y])
                            nan_indices.append((i, j))
                
                if nan_coords:
                    distances, indices = tree.query(nan_coords, k=1)
                    
                    for idx, (i, j) in enumerate(nan_indices):
                        if indices[idx] < len(valid_values):
                            grid_data[i, j] = valid_values[indices[idx]]
        
        # 如果还有NaN，使用网格数据的平均值填充
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