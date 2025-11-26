#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块 - 处理依赖数据的检验和裁剪
"""
import os
import shutil
from pathlib import Path
from osgeo import ogr, gdal
from typing import Dict, List, Optional, Tuple
import numpy as np

class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, config, fjson, rjson):
        self.config = config
        self.fjson = fjson
        self.rjson = rjson
        self.area_code = config["areaCode"]
        self.depend_dir = Path(config.get("dependDir"))

    def preprocess_depend_data(self):
        """预处理所有依赖数据 - 统一处理流程"""
        try:
            self.fjson.log("开始预处理依赖数据")

            # 阶段1: 首先处理矢量文件（必须首先完成）
            if not self._process_vector_files():
                raise Exception("矢量文件处理失败")

            # 阶段2: 处理格网文件（使用矢量文件裁剪）
            if not self._process_grid_file():
                raise Exception("格网文件处理失败")

            # 阶段3: 处理DEM文件（使用格网文件掩膜）
            if not self._process_dem_file():
                raise Exception("DEM文件处理失败")

            # 阶段4: 处理其他可选栅格文件（使用格网文件掩膜）
            self._process_optional_raster_files()

            # 阶段5: 处理参数文件  
            self._process_parameter_files()

            # 阶段6: 为QGIS准备省级矢量文件（不再创建临时市县级矢量文件）
            # 删除原有的 _create_temp_shp_for_subregion 调用

            # 阶段7: 最终一致性检查 - 确保所有栅格文件与gridFilePath完全一致
            self._final_consistency_check()

            self.fjson.log("依赖数据预处理完成")
            return True

        except Exception as e:
            self.fjson.log(f"依赖数据预处理失败: {str(e)}", codeId='1')
            return False

    def _final_consistency_check(self):
        """最终一致性检查 - 确保所有栅格文件与gridFilePath完全一致"""
        self.fjson.log("执行最终栅格数据一致性检查")
        
        grid_path = self.config.get('gridFilePath')
        if not grid_path or not Path(grid_path).exists():
            self.fjson.log("格网文件不存在，无法进行最终一致性检查")
            return
        
        # 检查所有栅格文件
        raster_files = {
            'DEM文件': self.config.get('demFilePath'),
            '防灾减灾能力文件': self.config.get('dpamFilePath'),
            '承载体脆弱性文件': self.config.get('vulFilePath'),
            '孕灾环境敏感性文件': self.config.get('sensFilePath'),
            '土地利用文件': self.config.get('landuseFilePath'),
            'GDP文件': self.config.get('GDPFilePath'),
            '收获面积文件': self.config.get('cropgainFilePath')
        }
        
        for desc, file_path in raster_files.items():
            if file_path and file_path.strip() and Path(file_path).exists():
                if self._are_rasters_consistent(grid_path, file_path):
                    self.fjson.log(f"✓ {desc}与格网文件一致: {file_path}")
                else:
                    self.fjson.log(f"✗ {desc}与格网文件不一致，进行强制重采样: {file_path}")
                    self._force_resample_to_match(file_path, grid_path, desc)

    def _force_resample_to_match(self, target_path: str, ref_path: str, file_desc: str):
        """强制重采样目标文件以匹配参考文件"""
        try:
            # 创建临时文件
            temp_path = Path(target_path).with_suffix('.consistent.tif')
            
            # 重采样
            success = self._resample_raster_to_match(target_path, str(temp_path), ref_path)
            
            if success and temp_path.exists():
                # 备份原文件
                backup_path = Path(target_path).with_suffix('.backup.tif')
                shutil.copy(target_path, backup_path)
                self.fjson.log(f"已备份原文件: {backup_path}")
                
                # 替换原文件
                shutil.move(str(temp_path), target_path)
                self.fjson.log(f"已重采样并替换{file_desc}: {target_path}")
                
                # 验证重采样后的文件
                # if self._are_rasters_consistent(ref_path, target_path):
                #     self.fjson.log(f"✓ {file_desc}重采样后与格网文件一致")
                # else:
                #     self.fjson.log(f"⚠ {file_desc}重采样后仍与格网文件不一致")
            else:
                self.fjson.log(f"✗ {file_desc}重采样失败")
                
        except Exception as e:
            self.fjson.log(f"强制重采样{file_desc}异常: {str(e)}")

    def _process_vector_files(self):
        """处理矢量文件 - 必须首先执行"""
        shp_path = self.config.get('shpFilePath')
        if not shp_path:
            raise ValueError("矢量文件路径为空，这是必要参数")

        if not Path(shp_path).exists():
            new_path = self._extract_province_shp(shp_path)
            if new_path:
                self.config['shpFilePath'] = new_path
                self.fjson.log(f"成功提取省份矢量文件: {new_path}")
                return True
            else:
                raise FileNotFoundError(f"无法找到或生成矢量文件: {shp_path}")
        else:
            # 即使文件存在，也进行基本验证
            if self._validate_shapefile(shp_path):
                self.fjson.log(f"矢量文件已验证: {shp_path}")
                return True
            else:
                raise ValueError(f"矢量文件验证失败: {shp_path}")

    def _extract_province_shp(self, target_path: str) -> Optional[str]:
        """从全国矢量文件中提取省份矢量"""
        try:
            target_path = Path(target_path)
            province_code = self.area_code[:2] + "0000"

            # 尝试多个可能的全国矢量文件
            national_files = [
                self.depend_dir / "shp" / "000000_prov.shp", # 优先使用省级
                self.depend_dir / "shp" / "000000_city.shp",
                self.depend_dir / "shp" / "000000_county.shp",  # 优先使用县级
            ]

            national_file = None
            for nf in national_files:
                if nf.exists():
                    national_file = nf
                    break

            if not national_file:
                self.fjson.log(f"全国矢量文件都不存在")
                return None

            # 创建输出目录
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 使用矢量提取函数
            DataPreprocessor.shp_region_select_PAC(
                in_shp=str(national_file),
                outfile=str(target_path),
                filed_name="省级码",
                region_list=[province_code]
            )

            if target_path.exists():
                return str(target_path)
            else:
                return None

        except Exception as e:
            self.fjson.log(f"提取省份矢量文件异常: {str(e)}")
            return None

    def _validate_shapefile(self, shp_path: str) -> bool:
        """验证矢量文件是否有效"""
        try:
            ds = ogr.Open(shp_path)
            if ds is None:
                return False
            layer = ds.GetLayer()
            if layer is None:
                return False
            feature_count = layer.GetFeatureCount()
            ds = None
            return feature_count > 0
        except:
            return False

    def _process_grid_file(self):
        """处理格网文件 - 使用矢量文件裁剪"""
        grid_path = self.config.get('gridFilePath')
        if not grid_path:
            raise ValueError("格网文件路径为空，这是必要参数")

        if not Path(grid_path).exists():
            new_path = self._clip_grid_with_shp(grid_path)
            if new_path:
                self.config['gridFilePath'] = new_path
                self.fjson.log(f"成功裁剪格网文件: {new_path}")
                return True
            else:
                raise FileNotFoundError(f"无法找到或生成格网文件: {grid_path}")
        else:
            # 格网文件是参考基准，只需要验证有效性
            if self._validate_raster_file(grid_path):
                self.fjson.log(f"格网文件已验证: {grid_path}")
                return True
            else:
                raise ValueError(f"格网文件验证失败: {grid_path}")

    def _validate_raster_file(self, raster_path: str) -> bool:
        """验证栅格文件是否有效"""
        try:
            ds = gdal.Open(raster_path)
            if ds is None:
                return False
            band = ds.GetRasterBand(1)
            if band is None:
                return False
            ds = None
            return True
        except:
            return False

    def _process_dem_file(self):
        """处理DEM文件 - 使用格网文件掩膜，并确保一致性"""
        dem_path = self.config.get('demFilePath')
        if not dem_path:
            raise ValueError("DEM文件路径为空，这是必要参数")

        grid_path = self.config.get('gridFilePath')
        if not grid_path or not Path(grid_path).exists():
            raise FileNotFoundError("格网文件不存在，无法处理DEM文件")

        if not Path(dem_path).exists():
            new_path = self._mask_raster_with_grid('dem', dem_path)
            if new_path:
                self.config['demFilePath'] = new_path
                self.fjson.log(f"成功掩膜DEM文件: {new_path}")
                # 立即验证一致性
                if self._are_rasters_consistent(grid_path, new_path):
                    self.fjson.log(f"✓ DEM文件与格网文件一致")
                else:
                    self.fjson.log(f"⚠ DEM文件与格网文件不一致，将进行重采样")
                    self._force_resample_to_match(new_path, grid_path, "DEM文件")
                return True
            else:
                raise FileNotFoundError(f"无法找到或生成DEM文件: {dem_path}")
        else:
            self.fjson.log(f"DEM文件已存在: {dem_path}")
            # 检查并确保与格网文件一致
            if self._are_rasters_consistent(grid_path, dem_path):
                self.fjson.log(f"✓ DEM文件与格网文件一致")
                return True
            else:
                self.fjson.log(f"⚠ DEM文件与格网文件不一致，将进行重采样")
                self._force_resample_to_match(dem_path, grid_path, "DEM文件")
                return True

    def _process_optional_raster_files(self):
        """处理可选栅格文件 - 使用格网文件掩膜，并确保一致性"""
        optional_files = {
            'dpam': self.config.get('dpamFilePath'),
            'vul': self.config.get('vulFilePath'),
            'sens': self.config.get('sensFilePath'),
            'landuse': self.config.get('landuseFilePath'),
            'gdp': self.config.get('GDPFilePath'),
            'cropgain': self.config.get('cropgainFilePath')
        }

        grid_path = self.config.get('gridFilePath')
        if not grid_path or not Path(grid_path).exists():
            self.fjson.log("格网文件不存在，无法处理可选栅格文件")
            return

        for file_type, file_path in optional_files.items():
            if file_path and file_path.strip():  # 非空路径
                if not Path(file_path).exists():
                    new_path = self._mask_raster_with_grid(file_type, file_path)
                    if new_path:
                        self._set_optional_file_path(file_type, new_path)
                        # 验证一致性
                        if self._are_rasters_consistent(grid_path, new_path):
                            self.fjson.log(f"✓ {file_type}文件与格网文件一致")
                        else:
                            self.fjson.log(f"⚠ {file_type}文件与格网文件不一致，将进行重采样")
                            self._force_resample_to_match(new_path, grid_path, f"{file_type}文件")
                    else:
                        self.fjson.log(f"无法找到或生成{file_type}文件，将使用空路径")
                        self._set_optional_file_path(file_type, "")
                else:
                    self.fjson.log(f"{file_type}文件已存在: {file_path}")
                    # 检查并确保一致性
                    if self._are_rasters_consistent(grid_path, file_path):
                        self.fjson.log(f"✓ {file_type}文件与格网文件一致")
                    else:
                        self.fjson.log(f"⚠ {file_type}文件与格网文件不一致，将进行重采样")
                        self._force_resample_to_match(file_path, grid_path, f"{file_type}文件")
            else:
                self.fjson.log(f"{file_type}文件路径为空，跳过处理")
                self._set_optional_file_path(file_type, "")

    # 其他方法保持不变，但确保所有栅格处理都包含一致性检查和必要的重采样

    def _are_rasters_consistent(self, ref_raster_path: str, target_raster_path: str) -> bool:
        """检查两个栅格文件是否一致（坐标系、范围、分辨率、网格数）"""
        try:
            ref_ds = gdal.Open(ref_raster_path)
            target_ds = gdal.Open(target_raster_path)

            if ref_ds is None or target_ds is None:
                return False

            # 检查坐标系
            ref_proj = ref_ds.GetProjection()
            target_proj = target_ds.GetProjection()
            if ref_proj != target_proj:
                return False

            # 检查地理变换参数
            ref_geo = ref_ds.GetGeoTransform()
            target_geo = target_ds.GetGeoTransform()

            # 比较地理变换参数（允许很小的浮点数误差）
            tolerance = 1e-10
            if (abs(ref_geo[0] - target_geo[0]) > tolerance or  # 左上角X
                abs(ref_geo[3] - target_geo[3]) > tolerance or  # 左上角Y
                abs(ref_geo[1] - target_geo[1]) > tolerance or  # X方向分辨率
                abs(ref_geo[5] - target_geo[5]) > tolerance):   # Y方向分辨率
                return False

            # 检查网格数
            ref_width = ref_ds.RasterXSize
            ref_height = ref_ds.RasterYSize
            target_width = target_ds.RasterXSize
            target_height = target_ds.RasterYSize

            if ref_width != target_width or ref_height != target_height:
                return False

            # 检查范围
            ref_ulx = ref_geo[0]
            ref_uly = ref_geo[3]
            ref_lrx = ref_ulx + ref_geo[1] * ref_width
            ref_lry = ref_uly + ref_geo[5] * ref_height

            target_ulx = target_geo[0]
            target_uly = target_geo[3]
            target_lrx = target_ulx + target_geo[1] * target_width
            target_lry = target_uly + target_geo[5] * target_height

            if (abs(ref_ulx - target_ulx) > tolerance or
                abs(ref_uly - target_uly) > tolerance or
                abs(ref_lrx - target_lrx) > tolerance or
                abs(ref_lry - target_lry) > tolerance):
                return False

            ref_ds = None
            target_ds = None

            return True

        except Exception as e:
            self.fjson.log(f"检查栅格一致性异常: {str(e)}")
            return False

    # 其余方法保持不变...
    def _process_parameter_files(self):
        """处理参数文件"""
        # 光合生产潜力参数文件
        photo_path = self.config.get('photosyntheticParamsPath')
        if photo_path and photo_path.strip():
            if not Path(photo_path).exists():         
                self.config['photosyntheticParamsPath'] = ""
        else:
            self.config['photosyntheticParamsPath'] = ""
        
        # 生长期参数文件
        growth_path = self.config.get('growthPeriodPath')
        if growth_path and growth_path.strip():
            if not Path(growth_path).exists():
                self.config['growthPeriodPath'] = ""       
        else:
            self.config['growthPeriodPath'] = ""

    # def _create_temp_shp_for_subregion(self):
    #     """为市县级区域创建临时矢量文件"""
    #     area_code = self.area_code  # 使用原始区域编码
        
    #     # 检查是否为市或县级编码
    #     if len(area_code) == 6:
    #         if area_code.endswith('0000'):  # 省级
    #             return
    #         elif area_code.endswith('00'):  # 市级
    #             region_type = 'city'
    #             national_file = self.depend_dir / "shp" / "000000_city.shp"
    #             field_name = "地级码"
    #         else:  # 县级
    #             region_type = 'county'
    #             national_file = self.depend_dir / "shp" / "000000_county.shp"
    #             field_name = "县级码"
            
    #         if national_file.exists():
    #             # 创建临时矢量文件
    #             temp_shp_path = Path(self.config['resultPath']) / "temp.shp"
    #             DataPreprocessor.shp_region_select_PAC(
    #                 in_shp=str(national_file),
    #                 outfile=str(temp_shp_path),
    #                 filed_name=field_name,
    #                 region_list=[area_code]
    #             )
                
    #             if temp_shp_path.exists():
    #                 self.config['tempShpFilePath'] = str(temp_shp_path)
    #                 self.fjson.log(f"创建临时{region_type}矢量文件: {temp_shp_path}")
                
    
    def prepare_qgis_shp_files(self):
        """准备QGIS画图所需的矢量文件"""
        try:
            qgis_shp_dir = Path("qgis_src/auxpath/shp")
            qgis_shp_dir.mkdir(parents=True, exist_ok=True)
            
            province_code = self.area_code[:2] + "0000"
            shp_types = {
                'sheng': ('000000_prov.shp', '省级码', f"{province_code}_sheng.shp"),
                'shi': ('000000_city.shp', '地级码', f"{province_code}_shi.shp"), 
                'xian': ('000000_county.shp', '县级码', f"{province_code}_xian.shp")
            }
            
            for shp_type, (national_file, field_name, output_file) in shp_types.items():
                output_path = qgis_shp_dir / output_file
                
                if not output_path.exists():
                    national_path = self.depend_dir / "shp" / national_file
                    if national_path.exists():
                        DataPreprocessor.shp_region_select_PAC(
                            in_shp=str(national_path),
                            outfile=str(output_path),
                            filed_name=field_name,
                            region_list=[province_code]
                        )
                        if output_path.exists():
                            self.fjson.log(f"生成QGIS {shp_type}矢量文件: {output_path}")
                    # else:
                    #     self.fjson.log(f"全国{shp_type}文件不存在: {national_path}")
                        
        except Exception as e:
            self.fjson.log(f"准备QGIS矢量文件失败: {str(e)}")

    def _mask_raster_with_grid(self, file_type: str, target_path: str) -> Optional[str]:
        """使用格网文件掩膜其他栅格文件 - 确保一致性"""
        try:
            target_path = Path(target_path)
            province_code = self.area_code[:2] + "0000"
            
            # 确定全国文件路径
            if file_type == 'dem':
                national_file = self.depend_dir / "dem" / "000000_dem.tif"
            elif file_type == 'dpam':
                national_file = self.depend_dir / "dpam" / "000000_dpam.tif"
            elif file_type == 'vul':
                national_file = self.depend_dir / "vulnerability" / "000000_vulnerability.tif"
            elif file_type == 'sens':
                national_file = self.depend_dir / "dpam" / "000000_sensitivity.tif"
            elif file_type == 'landuse':
                national_file = self.depend_dir / "landuse" / "000000_landuse.tif"
            elif file_type == 'gdp':
                national_file = self.depend_dir / "gdp" / "000000_GDP.tif"
            elif file_type == 'cropgain':
                # 收获面积文件需要作物代码
                crop_code = self.config.get('cropCode', '')
                if not crop_code:
                    self.fjson.log(f"无法获取作物代码，无法处理收获面积文件")
                    return None
                national_file = self.depend_dir / "cropgain" / f"000000_{crop_code}_gain.tif"
            else:
                return None
            
            # 检查全国文件是否存在
            if not national_file.exists():
                # self.fjson.log(f"全国{file_type}文件不存在: {national_file}")
                return None
            
            # 获取格网文件
            grid_file = self.config.get('gridFilePath')
            if not grid_file or not Path(grid_file).exists():
                self.fjson.log(f"格网文件不存在，无法进行栅格掩膜: {grid_file}")
                return None
            
            # 创建输出目录
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用格网文件进行掩膜
            success = self._resample_raster_to_match(
                str(national_file), 
                str(target_path), 
                str(grid_file)
            )
            
            if success and target_path.exists():
                self.fjson.log(f"成功掩膜{file_type}文件: {target_path}")
                return str(target_path)
            else:
                self.fjson.log(f"掩膜{file_type}文件失败")
                return None
                
        except Exception as e:
            self.fjson.log(f"掩膜{file_type}文件异常: {str(e)}")
            return None

    def _set_optional_file_path(self, file_type: str, path: str):
        """设置可选文件路径"""
        if file_type == 'landuse':
            self.config['landuseFilePath'] = path
        elif file_type == 'gdp':
            self.config['GDPFilePath'] = path
        elif file_type == 'cropgain':
            self.config['cropgainFilePath'] = path
        else:
            self.config[f'{file_type}FilePath'] = path

    def _clip_grid_with_shp(self, target_path: str) -> Optional[str]:
        """使用矢量文件裁剪格网文件"""
        try:
            target_path = Path(target_path)
            
            # 全国格网文件
            national_grid = self.depend_dir / "grid" / "000000_30grid.tif"
            if not national_grid.exists():
                self.fjson.log(f"全国格网文件不存在: {national_grid}")
                return None
            
            # 省份矢量文件
            shp_file = self.config.get('shpFilePath')
            if not shp_file or not Path(shp_file).exists():
                self.fjson.log(f"省份矢量文件不存在，无法裁剪格网")
                return None
            
            # 创建输出目录
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用矢量文件裁剪格网
            
            warp_parameters = gdal.WarpOptions(format='GTiff',
                                               cutlineDSName=str(shp_file),
                                               cropToCutline=True,
                                               dstNodata=0)
            gdal.Warp(str(target_path), str(national_grid), options=warp_parameters)
            
            if target_path.exists():
                return str(target_path)
            else:
                return None
                
        except Exception as e:
            self.fjson.log(f"裁剪格网文件异常: {str(e)}")
            return None

    def _resample_raster_to_match(self, input_path: str, output_path: str, ref_path: str, resample_alg='nearest') -> bool:
        """将输入栅格重采样以匹配参考栅格"""
        try:
            # 打开参考栅格
            ref_ds = gdal.Open(ref_path)
            if ref_ds is None:
                self.fjson.log(f"无法打开参考栅格: {ref_path}")
                return False
            
            # 获取参考栅格信息
            ref_geo = ref_ds.GetGeoTransform()
            ref_proj = ref_ds.GetProjection()
            ref_width = ref_ds.RasterXSize
            ref_height = ref_ds.RasterYSize
            
            # 计算参考范围
            ulx = ref_geo[0]
            uly = ref_geo[3]
            lrx = ulx + ref_geo[1] * ref_width
            lry = uly + ref_geo[5] * ref_height

            # 重采样算法映射
            resample_alg_map = {
                'nearest': gdal.GRA_NearestNeighbour,
                'bilinear': gdal.GRA_Bilinear,
                'cubic': gdal.GRA_Cubic,
                'cubicspline': gdal.GRA_CubicSpline,
                'lanczos': gdal.GRA_Lanczos,
                'average': gdal.GRA_Average,
                'mode': gdal.GRA_Mode
            }
            
            resample_method = resample_alg_map.get(resample_alg, gdal.GRA_Bilinear)
            
            # 重采样选项
            warp_options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=[ulx, lry, lrx, uly],
                width=ref_width,
                height=ref_height,
                dstSRS=ref_proj,
                resampleAlg=gdal.GRA_Bilinear,  # 双线性重采样
                creationOptions=['COMPRESS=LZW', 'TILED=YES']
            )
            
            # 执行重采样
            result_ds = gdal.Warp(output_path, input_path, options=warp_options)
            if result_ds is None:
                self.fjson.log(f"重采样失败: {input_path} -> {output_path}")
                return False
            
            result_ds = None
            ref_ds = None
            
            self.fjson.log(f"重采样成功: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.fjson.log(f"重采样异常: {str(e)}")
            return False


    @staticmethod
    def maskRasterByRaster(infile, maskfile, outfile, mask_nodata=None, dst_nodata=None, srs_nodata=None):
        """
        数据掩膜
        :param infile: 文件或文件对象
        :param maskfile: 文件或文件对象
        :param outfile: str,结果文件
        :param mask_nodata: float， 掩膜的无效值
        :param dst_nodata: float, 输出的无效值
        :return:
        """
        try:
            try:
                inds = gdal.Open(infile)
            except:
                inds = infile
            indata = inds.ReadAsArray()
            cols = inds.RasterXSize
            rows = inds.RasterYSize
            geo = inds.GetGeoTransform()
            proj = inds.GetProjection()
            band = inds.GetRasterBand(1)
            if srs_nodata is not None:
                pass
            else:
                srs_nodata = band.GetNoDataValue()
            indata[indata == srs_nodata] = dst_nodata
            try:
                maskds = gdal.Open(maskfile)
            except:
                maskds = maskfile
            maskdata = maskds.ReadAsArray()
            band_mask = maskds.GetRasterBand(1)
            if mask_nodata is not None:
                pass
            else:
                mask_nodata = band_mask.GetNoDataValue()
            indata[maskdata==mask_nodata] = dst_nodata
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                out_ds = driver.Create("", cols, rows, 1, band.DataType)
            else:
                driver = gdal.GetDriverByName("Gtiff")
                out_ds = driver.Create(outfile, cols, rows, 1, band.DataType)
            out_ds.SetGeoTransform(geo)
            out_ds.SetProjection(proj)
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(indata)
            out_band.SetNoDataValue(dst_nodata)
            return out_ds
        finally:
            inds =None
            maskds = None    

    @staticmethod
    def shp_region_select_PAC(in_shp, outfile, filed_name, region_list): 
        # in_shp为输入shp文件；outfile输出文件；filed_name提取字段的名称；region_list区域列表
        gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
        shp = ogr.Open(in_shp, 1)   # 打开shp文件
        lyr = shp.GetLayer()
        lydefn = lyr.GetLayerDefn()
        spatialref = lyr.GetSpatialRef()  # 获取空间坐标系
        geomtype = lydefn.GetGeomType()   # 文件类型（point，polyline，polygon等）
        a = []  # 初始化列表
        
        for i, fea in enumerate(lyr):
            feat = lyr.GetFeature(i)
            fid = feat.GetField(filed_name) 
            if ((str(fid)[0:2] + "0000") in region_list) | ((str(fid)[0:4] + "00") in region_list) | (str(fid) in region_list):
                a.append(fid)  # 获取字段的属性值

        driver = ogr.GetDriverByName("ESRI Shapefile")   # 创建shp驱动
        out_shp = driver.CreateDataSource(outfile)    # 创建文件，文件命名为字段属性值+输入的文件名。
        
        outlayer = out_shp.CreateLayer('0', srs=spatialref, geom_type=geomtype, options=("ENCODING=UTF-8", ))
        
        # 复制原始字段
        for k in range(0, lydefn.GetFieldCount()):
            fieldDefn = lydefn.GetFieldDefn(k)
            fieldType = fieldDefn.GetType()
            ret = outlayer.CreateField(fieldDefn, fieldType)
        
        # 新增：判断是否为省级文件，如果是则增加"code"字段
        is_provincial_file = (filed_name == "省级码")
        if is_provincial_file:
            # 创建新的code字段
            code_field = ogr.FieldDefn("code", ogr.OFTString)
            outlayer.CreateField(code_field)
        
        outlayerDefn = outlayer.GetLayerDefn()
        
        for i in range(0, lyr.GetFeatureCount()):
            feat = lyr.GetFeature(i)
            fid = feat.GetField(filed_name)
            
            for j in range(len(a)): 
                if fid == a[j]:    # 判断属性值等于其中某一个值，提取相应的图层
                    outFeature = ogr.Feature(outlayerDefn)
                    geom = feat.GetGeometryRef()
                    outFeature.SetGeometry(geom)
                    
                    # 复制原始字段
                    for k in range(0, outlayerDefn.GetFieldCount()):
                        field_name = outlayerDefn.GetFieldDefn(k).GetName()
                        # 如果是省级文件且当前字段不是新增的code字段，复制原始值
                        if field_name != "code" or not is_provincial_file:
                            outFeature.SetField(field_name, feat.GetField(k))
                    
                    # 新增：如果是省级文件，设置code字段的值
                    if is_provincial_file:
                        # 获取省级码的值并设置到code字段
                        provincial_code = feat.GetField(filed_name)
                        outFeature.SetField("code", str(provincial_code))
                    
                    outlayer.CreateFeature(outFeature)
                    outFeature = None
        
        out_shp = None


    def _clip_raster_to_region(self, raster_path: str, area_code: str) -> str:
        """
        使用GDAL将栅格数据裁剪到指定区域
        
        Args:
            raster_path: 输入栅格文件路径
            area_code: 区域代码
        
        Returns:
            str: 裁剪后的文件路径（如果裁剪失败则返回原路径）
        """
        try:
            # 根据区域代码确定使用的矢量文件和查询字段
            if len(area_code) == 6:
                if area_code.endswith('0000'):  # 省级，不需要裁剪
                    self.fjson.log(f"省级区域 {area_code} 不需要裁剪")
                    return raster_path
                elif area_code.endswith('00'):  # 市级
                    shp_file = "000000_city.shp"
                    # sql_field = "地级码"
                    # region_type = "市级"
                else:  # 县级
                    shp_file = "000000_county.shp" 
                    # sql_field = "县级码"
                    # region_type = "县级"
            else:
                self.fjson.log(f"未知区域代码格式: {area_code}，不进行裁剪")
                return raster_path
            
            # 构建完整矢量文件路径
            # depend_dir = Path(self.config.get("dependDir", "/mnt/d/2024_NYQH_NCC/05_Data/03_Depend"))
            shp_path = self.depend_dir / "shp" / shp_file
            
            if not shp_path.exists():
                # self.fjson.log(f"矢量文件不存在: {shp_path}，无法进行{region_type}裁剪")
                return raster_path
                
            # self.fjson.log(f"使用GDAL裁剪{region_type}结果: {shp_path}, 区域代码: {area_code}")
            
            # 创建临时文件路径
            temp_raster_path = raster_path.replace('.tif', f'_temp_{os.getpid()}.tif')
            
            # 构建SQL查询条件
            layer_name = os.path.splitext(os.path.basename(shp_path))[0]
            # sql_query = f"SELECT * FROM {layer_name} WHERE code = '{area_code}'"
                    # 方法1: 使用双引号引用中文字段名
            sql_query = f'SELECT * FROM "{layer_name}" WHERE code = \'{area_code}\''
            
            self.fjson.log(f"执行裁剪: {area_code}, SQL: {sql_query}")

            # 如果临时文件已存在，先删除
            if os.path.exists(temp_raster_path):
                try:
                    os.remove(temp_raster_path)
                except Exception as e:
                    self.fjson.log(f"删除已存在的临时文件失败: {str(e)}")

            # 使用gdal.Warp进行裁剪
            warp_options = gdal.WarpOptions(
                format='GTiff',
                cutlineDSName=str(shp_path),
                cutlineSQL=sql_query,
                cropToCutline=True,
                dstNodata=0,
                creationOptions=['COMPRESS=LZW', 'BIGTIFF=IF_SAFER']
            )
            
            self.fjson.log(f"开始GDAL裁剪操作: {raster_path} -> {temp_raster_path}")
            
            # 执行裁剪
            result = gdal.Warp(temp_raster_path, raster_path, options=warp_options)
            
            if result is None:
                self.fjson.log("GDAL裁剪操作返回None")
                # 检查是否生成了文件
                if not os.path.exists(temp_raster_path):
                    self.fjson.log("GDAL裁剪失败，未生成文件，将使用未裁剪结果")
                    return raster_path
                else:
                    self.fjson.log("文件已生成但GDAL返回None，继续处理")
            else:
                # 立即关闭结果数据集
                result = None
            
            # 如果裁剪成功生成文件，替换原文件
            if os.path.exists(temp_raster_path):
                # 删除原文件
                if os.path.exists(raster_path):
                    os.remove(raster_path)
                # 重命名临时文件
                shutil.move(temp_raster_path, raster_path)
                self.fjson.log(f"结果GDAL裁剪完成")
                return raster_path
            else:
                self.fjson.log("GDAL裁剪失败，临时文件未生成，使用未裁剪结果")
                return raster_path
                
        except Exception as e:
            self.fjson.log(f"GDAL裁剪异常: {str(e)}")
            # 清理临时文件
            if 'temp_raster_path' in locals() and os.path.exists(temp_raster_path):
                try:
                    os.remove(temp_raster_path)
                except:
                    pass
            self.fjson.log("裁剪失败，使用未裁剪结果")
            return raster_path
        