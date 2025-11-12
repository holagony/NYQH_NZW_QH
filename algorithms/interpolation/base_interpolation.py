from core.base_classes import BaseAlgorithm
from typing import Dict, Any
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path

class BaseInterpolation(BaseAlgorithm):
    """插值算法基类"""
    
    def execute(self, data: Dict[str, Any], params: Dict[str, Any] = None) -> Any:
        """执行插值"""
        # 提取站点数据
        station_values = data['station_values']
        dem_path = data['dem_path']
        lulc_path = data['lulc_path']
        shp_path = data['shp_path']
        grid_path = data['grid_path']
        area_code = data['area_code']
        
        # 读取全国格网
        with rasterio.open(grid_path) as src:
            grid = src.read(1)
            grid_meta = src.meta
        
        # 读取省份矢量
        shp_files = list(Path(shp_path).glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"在 {shp_path} 目录下找不到矢量文件")
        
        gdf = gpd.read_file(shp_files[0])
        # 根据area_code筛选省份
        province_gdf = gdf[gdf['PAC'] == area_code]
        
        if province_gdf.empty:
            raise ValueError(f"未找到行政区划代码为 {area_code} 的区域")
        
        # 确保坐标系一致
        if province_gdf.crs != src.crs:
            province_gdf = province_gdf.to_crs(src.crs)
        
        # 使用省份矢量裁剪全国格网
        out_image, out_transform = mask(src, province_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # 执行具体的插值算法
        result = self._interpolate(station_values, out_image, out_meta, dem_path, lulc_path)
        
        return result
    
    def _interpolate(self, station_values: Dict[str, Any], grid: np.ndarray, 
                    grid_meta: Dict[str, Any], dem_path: str, lulc_path: str) -> Any:
        """具体的插值算法实现，由子类重写"""
        raise NotImplementedError("子类必须实现_interpolate方法")