#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版区划处理器 - 支持逐日数据
"""
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import importlib
from typing import List,Dict,Any,Optional,Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .data_manager import DataManager
from configs.base_config import BaseConfig
from .preprocess import DataPreprocessor

class ZoningProcessor:
    """区划处理器"""
    
    def __init__(self, config, fjson, rjson):
        self.config = config
        self.fjson = fjson
        self.rjson = rjson

        # 初始化配置管理器 - 使用绝对路径
        configs_dir = Path(__file__).parent.parent / "configs"
        self.config_manager = BaseConfig(str(configs_dir))        
        
        # 初始化数据预处理器并执行预处理
        self.preprocessor = DataPreprocessor(config, fjson, rjson)
        
        # # 检查依赖文件存在性
        # self.preprocessor.check_depend_files_existence()
        
        # 执行数据预处理
        if not self.preprocessor.preprocess_depend_data():
            raise Exception("依赖数据预处理失败")
        
        # # 再次检查预处理后的文件存在性
        # self.preprocessor.check_depend_files_existence()
        
        # 初始化参数
        self.area_code = self.config["areaCode"]  # 原始区域编码
        
        # 新增：计算省级编码和区域名称
        self.province_code, self.area_name = self._get_province_code_and_name()
        
        self.crop_code = self.config["cropCode"]
        self.zoning_type = self.config["zoningType"]
        self.element = self.config.get("element", "")

        # 算法来源参数
        self.algo_from = self.config.get("algoFrom", "")

        # 时间参数
        self.start_date = self.config.get("startDate", "19910101")
        self.end_date = self.config.get("endDate", "20201231")
        
        # 根据 algoFrom 确定配置键
        if self.algo_from:
            config_key = self.algo_from
        else:
            config_key = f"zoning_{self.area_code}_{self.crop_code}_{self.zoning_type}_{self.element}"
        
        # 合并配置
        user_algorithm_config = self.config.get('algorithmConfig')
        self.algorithm_config = self.config_manager.merge_configs(
            user_algorithm_config, config_key, self.crop_code, self.zoning_type,self.element
        )
        
        # 获取指标配置
        if config_key and "zoning_general_calculator" in config_key:
            # 通用计算器模式：不按element提取配置
            intermediate_filename = f"intermediate_{self.crop_code}_general.csv"
        else:
            # 特定元素模式：按element提取配置
            # if self.element:
            #     intermediate_filename = f"intermediate_{self.crop_code}_{self.element}.csv"
            # else:
            #     intermediate_filename = f"intermediate_{self.crop_code}.csv"
            intermediate_filename = f"intermediate_{self.crop_code}_{self.element}.csv"

        self.indicator_configs = self.algorithm_config.get("indicators")
        # 文件路径
        self.result_path = self.config.get("resultPath", "./results")
        Path(self.result_path).mkdir(parents=True, exist_ok=True)

        # 设置中间结果输出路径
        intermediate_dir = Path(self.result_path) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_output = intermediate_dir / intermediate_filename
                
        # 初始化数据管理器
        self.data_manager = DataManager(config['inputFilePath'],
                    station_file=config.get('stationFilePath'),
                    multiprocess=config.get('multiprocess', True),
                    num_processes=config.get('num_processes',16)
                    )
        # 插值和分类方法初始化
        self._algorithms = self._load_algorithms()
        

    def process(self):
        """执行简化的区划处理流程 - 支持逐日数据"""
        try:
            self.fjson.log("获取站点信息")
            
            # 1. 获取站点列表 - 使用省级编码获取站点
            station_ids = self.data_manager.get_stations_by_province(self.province_code)
            self.fjson.log(f"找到 {len(station_ids)} 个站点，使用省级编码: {self.province_code}")
            
            # 2. 批量计算指标
            # 检查是否有daily频率的指标
            has_daily_indicators = any(
                config.get("frequency") == "daily" 
                for config in self.indicator_configs.values()
            )
            
            if not os.path.exists(self.intermediate_output):
                # 计算指标，返回常规结果和逐日结果
                station_indicators_df, daily_indicators_df = self.data_manager.calculate_indicators_for_stations(
                    station_ids, self.indicator_configs, self.start_date, self.end_date, self.intermediate_output
                )
                
                # 如果有逐日数据，记录信息
                if daily_indicators_df is not None and not daily_indicators_df.empty:
                    self.fjson.log(f"生成逐日数据，包含 {len(daily_indicators_df)} 条记录")
            else:
                # 读取已有的结果
                station_indicators_df = pd.read_csv(self.intermediate_output)
                daily_indicators_df = None
                
                # 尝试读取逐日数据文件
                daily_output_path = Path(self.intermediate_output).with_name(f"daily_{Path(self.intermediate_output).name}")
                if daily_output_path.exists():
                    try:
                        daily_indicators_df = pd.read_csv(daily_output_path, encoding='gbk')
                    except:
                        daily_indicators_df = pd.read_csv(daily_output_path, encoding='utf-8')
                    self.fjson.log(f"读取逐日数据，包含 {len(daily_indicators_df)} 条记录")
            
            self.fjson.info("指标计算完成")     
            
            # 3. 准备站点数据
            # 如果有daily指标且需要传入区划计算，则使用daily数据
            if has_daily_indicators and daily_indicators_df is not None and not daily_indicators_df.empty:
                self.fjson.log("使用逐日数据进行区划计算")
                station_indicators = daily_indicators_df
                station_coords = self._prepare_station_coords_from_daily(daily_indicators_df)
            else:
                self.fjson.log("使用逐年或多年数据进行区划计算")
                station_indicators, station_coords = self._prepare_station_data(self.intermediate_output, self.indicator_configs)
          
            # 4. 执行区划计算
            zoning_result = self._execute_zoning_calculation(station_indicators, station_coords)
            self.fjson.info("区划计算完成") 
            
            # 5. 保存结果
            self._save_results(zoning_result)
            self.fjson.info("结果输出完成")
            
            return True
            
        except Exception as e:
            self.fjson.info(f"区划处理失败: {str(e)}", codeId='1')
            import traceback
            self.fjson.log(f"详细错误: {traceback.format_exc()}")
            return False

    def _prepare_station_coords_from_daily(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """从逐日数据中提取站点坐标信息"""
        station_coords = {}
        
        # 按站点分组，获取每个站点的坐标信息
        for station_id, group in daily_df.groupby('station_id'):
            first_row = group.iloc[0]
            station_coords[station_id] = {
                'lat': float(first_row.get('lat', np.nan)),
                'lon': float(first_row.get('lon', np.nan)),
                'altitude': float(first_row.get('altitude', np.nan))
            }
        
        return station_coords

    def _prepare_station_data(self, station_indicators_df, indicator_configs):
        """准备站点数据 - 适配 yearly、lta 和 daily 数据类型"""
        try:
            # 读取CSV文件
            station_indicators_df = pd.read_csv(station_indicators_df, dtype=str, encoding="gbk")
        except:
            station_indicators_df = pd.read_csv(station_indicators_df, dtype=str, encoding="utf-8")
        
        print("准备站点数据...")
        
        # 检查是否是逐日数据的DataFrame格式
        if 'datetime' in station_indicators_df.columns:
            # 这是逐日数据格式，直接返回DataFrame
            self.fjson.log("检测到逐日数据格式")
            station_coords = self._prepare_station_coords_from_daily(station_indicators_df)
            return station_indicators_df, station_coords
        
        # 原有的处理逻辑（用于yearly和lta数据）
        # 使用配置中的指标名称
        indicator_names = list(indicator_configs.keys())
        
        # 检查DataFrame中实际存在的列
        actual_columns = set(station_indicators_df.columns)
        
        # 找出匹配的指标列
        matched_indicators = [col for col in indicator_names if col in actual_columns]
        
        # 排除的列（坐标信息和元数据）
        exclude_columns = ['station_id', 'lat', 'lon', 'altitude', 'province', 'city', 'county', 'error', 'station_name']
        
        station_indicators = {}
        station_coords = {}
        
        # 批量处理
        for _, row in station_indicators_df.iterrows():
            station_id = row['station_id']
            
            # 提取坐标信息
            station_coords[station_id] = {
                'lat': float(row.get('lat', np.nan)),
                'lon': float(row.get('lon', np.nan)),
                'altitude': float(row.get('altitude', np.nan))
            }
            
            # 提取指标数据 - 只提取配置中定义的指标
            station_data = {}
            for indicator_name in matched_indicators:
                value = row.get(indicator_name)
                
                if pd.isna(value) or value == '':
                    station_data[indicator_name] = np.nan
                    continue
                
                try:
                    # 尝试解析为数值
                    numeric_value = float(value)
                    station_data[indicator_name] = numeric_value
                except (ValueError, TypeError):
                    # 如果不是数值，尝试解析为字典（yearly数据）
                    try:
                        if value.startswith('{') and value.endswith('}'):
                            # 简单的字典格式解析
                            dict_str = value.strip('{}')
                            pairs = dict_str.split(',')
                            yearly_dict = {}
                            for pair in pairs:
                                if ':' in pair:
                                    year_str, val_str = pair.split(':', 1)
                                    year = int(year_str.strip().strip('"').strip("'"))
                                    val = float(val_str.strip())
                                    yearly_dict[year] = val
                            station_data[indicator_name] = yearly_dict
                        else:
                            # 尝试JSON解析
                            import json
                            yearly_dict = json.loads(value)
                            if isinstance(yearly_dict, dict):
                                # 转换值为浮点数
                                station_data[indicator_name] = {int(k): float(v) for k, v in yearly_dict.items()}
                            else:
                                station_data[indicator_name] = np.nan
                    except:
                        print(f"警告: 无法解析指标 {indicator_name} 的值: {value}")
                        station_data[indicator_name] = np.nan
            
            station_indicators[station_id] = station_data
        
        # 统计有效数据
        valid_stations = 0
        yearly_indicators = {}
        lta_indicators = {}
        
        for station_id, data in station_indicators.items():
            has_valid_data = False
            for indicator_name, value in data.items():
                if isinstance(value, dict) and value:  # yearly 数据
                    has_valid_data = True
                    if indicator_name not in yearly_indicators:
                        yearly_indicators[indicator_name] = 0
                    yearly_indicators[indicator_name] += 1
                elif isinstance(value, (int, float)) and not np.isnan(value):  # lta 数据
                    has_valid_data = True
                    if indicator_name not in lta_indicators:
                        lta_indicators[indicator_name] = 0
                    lta_indicators[indicator_name] += 1
            
            if has_valid_data:
                valid_stations += 1
        
        print(f"成功准备 {valid_stations}/{len(station_indicators)} 个站点的有效数据")
        
        return station_indicators, station_coords

    def _execute_zoning_calculation(self, station_indicators: Any, station_coords: Dict[str, Any]) -> Dict[str, Any]:
        """执行区划计算 - 支持DataFrame格式的逐日数据"""
        
        # 根据 algoFrom 确定计算器名称
        if self.algo_from:
            if "zoning_general_calculator" in self.algo_from:
                calculator_name = self.algo_from
                class_name = self.algo_from.replace("zoning_general_calculator_", "", 1)
                if not class_name or class_name == "zoning_general_calculator":
                    class_name = "GenericZoningCalculator"
                self.fjson.log(f"使用配置算法: {calculator_name}")
            else:
                calculator_name = self.algo_from
                # class_name = f"{self.crop_code}_{self.zoning_type}_{self.element}"
                class_name = "ProvZongingCalculator"
                self.fjson.log(f"使用配置算法: {calculator_name}")
        else:
            calculator_name = "zoning_general_calculator"
            class_name = "GenericZoningCalculator"
            self.fjson.log(f"使用默认算法: {calculator_name}")
        
        module_path = f"algorithms.zoning_calculator.{calculator_name}"
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            self.fjson.log(f"模块导入成功: {module_path}")
            
            # 获取计算器类
            if not hasattr(module, class_name):
                class_name = f"{self.crop_code}_{self.zoning_type}"
                
            if hasattr(module, class_name):
                calculator_class = getattr(module, class_name)
                self.fjson.log(f"找到计算器类: {calculator_class}")
                
                # 准备计算参数 - 包含所有可能的依赖文件路径
                calc_params = {
                    'station_indicators': station_indicators,
                    'station_coords': station_coords,
                    'algorithmConfig': self.algorithm_config,
                    'algorithms': self._algorithms,
                    'config': self.config,
                    # 可选依赖文件路径
                    'dpamFilePath': self.config.get('dpamFilePath', ''),
                    'vulFilePath': self.config.get('vulFilePath', ''),
                    'sensFilePath': self.config.get('sensFilePath', ''),
                    'landuseFilePath': self.config.get('landuseFilePath', ''),
                    'GDPFilePath': self.config.get('GDPFilePath', ''),
                    'cropgainFilePath': self.config.get('cropgainFilePath', ''),
                    'photosyntheticParamsPath': self.config.get('photosyntheticParamsPath', ''),
                    'growthPeriodPath': self.config.get('growthPeriodPath', '')
                }
                
                # 如果是DataFrame格式，添加标记
                if isinstance(station_indicators, pd.DataFrame):
                    calc_params['data_type'] = 'daily'
                    self.fjson.log("传入逐日DataFrame数据到区划计算器")
                else:
                    calc_params['data_type'] = 'station'
                    self.fjson.log("传入逐年或多年站点数据到区划计算器")
                
                # 执行计算
                calculator_instance = calculator_class()
                result = calculator_instance.calculate(calc_params)
                self.fjson.log("区划计算执行成功")
                
                return result
                
            else:
                self.fjson.log(f"模块 {module_path} 中没有找到类 {class_name}")
                raise AttributeError(f"类 {class_name} 不存在")
                
        except ImportError as e:
            self.fjson.log("尝试使用通用计算器")
            return self._fallback_to_generic_calculator(station_indicators, station_coords)
            
        except Exception as e:
            self.fjson.log(f"计算器执行异常: {str(e)}")
            raise

    def _fallback_to_generic_calculator(self, station_indicators: Any, station_coords: Dict[str, Any]) -> Dict[str, Any]:
        """通用算法 - 支持DataFrame格式"""
        try:
            from algorithms.zoning_calculator.zoning_general_calculator import GenericZoningCalculator
            
            calc_params = {
                'station_indicators': station_indicators,
                'station_coords': station_coords,
                'grid_path': self.config['gridFilePath'],
                'dem_path': self.config.get('demFilePath'),
                'algorithmConfig': self.algorithm_config
            }
            
            # 如果是DataFrame格式，添加标记
            if isinstance(station_indicators, pd.DataFrame):
                calc_params['data_type'] = 'daily'
                self.fjson.log("通用计算器处理逐日DataFrame数据")
            else:
                calc_params['data_type'] = 'station'
                self.fjson.log("通用计算器处理常规站点数据")
            
            calculator = GenericZoningCalculator()
            result = calculator.calculate(calc_params)
            self.fjson.log("通用计算器执行成功")
            return result
            
        except Exception as e:
            self.fjson.log(f"通用计算器也失败: {str(e)}")
            raise


    # def _save_results(self, zoning_result: Dict[str, Any]):
    #     """保存结果 - 使用GDAL直接裁剪市县级数据"""
    #     # 生成输出文件名
    #     tif_filename = self._generate_output_filename("tif")
    #     tif_path = os.path.join(self.result_path, tif_filename)
        
    #     # 保存TIFF文件
    #     self._save_geotiff(zoning_result['data'], zoning_result['meta'], tif_path, 0)
        
    #     from algorithms.common_tool.raster_tool import RasterTool
    #     RasterTool.maskRasterByRaster(tif_path, self.config['gridFilePath'], tif_path,
    #                                 mask_nodata=0, dst_nodata=0, srs_nodata=0)
        
    #     # 如果是市县级区域，使用GDAL直接裁剪结果
    #     temp_shp_path = self.config.get('tempShpFilePath')
    #     if temp_shp_path and Path(temp_shp_path).exists():
    #         tif_path = DataPreprocessor.clip_raster_to_region(tif_path, temp_shp_path, self.area_code)
        
    #     # 写入结果到rjson
    #     if tif_path:
    #         self.rjson.info("result", [tif_path, "NYQH_NZW", "农作物气候区划", "QH", "TIFF"])
                
    #     # 生成PNG专题图
    #     png_filename = self._generate_output_filename("png")
    #     png_path = os.path.join(self.result_path, png_filename)
        
    #     # 准备QGIS矢量文件并生成专题图
    #     self.preprocessor.prepare_qgis_shp_files()
    #     self._generate_qgis_map(tif_path, png_path)
    #     self.fjson.log(f"专题图已生成: {png_path}")

    #     if png_path:
    #         self.rjson.info("result", [png_path, "NYQH_NZW", "农作物气候区划专题图", "QH", "PNG"])

             
    def _save_results(self, zoning_result: Dict[str, Any]):
        """保存结果 - 使用GDAL直接裁剪市县级数据"""
        # 生成输出文件名
        tif_filename = self._generate_output_filename("tif")
        tif_path = os.path.join(self.result_path, tif_filename)
        
        # 保存TIFF文件
        self._save_geotiff(zoning_result['data'], zoning_result['meta'], tif_path, 0)
        
        self.preprocessor.maskRasterByRaster(tif_path, self.config['gridFilePath'], tif_path,
                                    mask_nodata=0, dst_nodata=0, srs_nodata=0)
        
        # 如果是市县级区域，使用GDAL直接裁剪结果
        if not self.area_code.endswith('0000'):
            tif_path = self.preprocessor._clip_raster_to_region(tif_path, self.area_code)
        
        # 写入结果到rjson
        if tif_path:
            self.rjson.info("result", [tif_path, "NYQH_NZW", "农作物气候区划", "QH", "TIFF"])

        # sys.exit(0)
        # 生成PNG专题图
        png_filename = self._generate_output_filename("png")
        png_path = os.path.join(self.result_path, png_filename)
        
        # 准备QGIS矢量文件并生成专题图
        self.preprocessor.prepare_qgis_shp_files()
        self._generate_qgis_map(tif_path, png_path)
        self.fjson.log(f"专题图已生成: {png_path}")

        if png_path:
            self.rjson.info("result", [png_path, "NYQH_NZW", "农作物气候区划专题图", "QH", "PNG"])
        
    def _generate_output_filename(self, file_type: str) -> str:
        """生成输出文件名"""
        crop_code = self.crop_code.upper()[:4]
        factor_code = self._get_factor_code(self.crop_code,self.zoning_type,self.element)

        if file_type == "tif":
            base_name = f"Q_PR_{crop_code}-{self.zoning_type}_{self.area_code}_{factor_code}"
            return f"{base_name}.tif"
        elif file_type == "png":
            base_name = f"Q_PP_{crop_code}-{self.zoning_type}_{self.area_code}_{factor_code}"
            return f"{base_name}.png"
        # elif file_type == "json":
        #     return f"{base_name}.json"
        else:
            return f"{base_name}.{file_type}"
    
    def _get_factor_code(self,crop_code: str,zoning_type: str,element: str) -> str:
        cropfile=os.path.dirname(self.config.get('stationFilePath'))+"/crop.csv"
        QHfile=os.path.dirname(self.config.get('stationFilePath'))+"/QH.csv"
        # cropfile="D:/project/农业气候资源普查和区划/code/china/depend_data/crop.csv"
        # QHfile="D:/project/农业气候资源普查和区划/code/china/depend_data/QH.csv"
        try:
            crop_ds = pd.read_csv(cropfile, dtype=str, encoding="utf-8")
        except:
            crop_ds = pd.read_csv(cropfile, dtype=str, encoding="gbk") 
        try:
            QH_ds = pd.read_csv(QHfile, dtype=str, encoding="utf-8")
        except:
            QH_ds = pd.read_csv(QHfile, dtype=str, encoding="gbk")       
        crop_class=crop_ds[crop_ds["代码"]==crop_code]["类型"].iloc[0]    #作物类型 
        factor_code=QH_ds[(QH_ds["作物名称"]==crop_class)&(QH_ds["区划类型代码"]==zoning_type)&(QH_ds["英文编码"]==element)]["区划类别代码"].iloc[0]   #区划类别代码
        factor_code=str(factor_code).zfill(3)
        """获取因子代码"""
        # factor_codes = {
        #     "protein": "001",
        #     "fat": "002",
        #     "sclerotinia": "101",
        #     "bean_moth": "102", 
        #     "high_temperature": "201"
        # }
        return factor_code

    def _get_province_code_and_name(self) -> Tuple[str, str]:
        """获取省级编码和区域名称"""
        area_code = self.area_code
        admin_code_file = self.config.get("adminCodeFile")
        
        # 如果没有提供行政区划编码文件，使用简单逻辑
        if not admin_code_file:
            return self._get_province_code_and_name_fallback(area_code)
        
        try:
            # 读取行政区划编码文件
            admin_df = pd.read_csv(admin_code_file, dtype=str, encoding='utf-8')
        except:
            try:
                admin_df = pd.read_csv(admin_code_file, dtype=str, encoding='gbk')
            except Exception as e:
                self.fjson.log(f"无法读取行政区划编码文件: {str(e)}，使用备用逻辑")
                return self._get_province_code_and_name_fallback(area_code)
        
        # 根据areaCode长度和格式判断区域级别
        if len(area_code) == 6:
            if area_code.endswith('0000'):  # 省级
                # 匹配省代码
                province_match = admin_df[admin_df['省代码'] == area_code]
                if not province_match.empty:
                    province_name = province_match.iloc[0]['省']
                    return area_code, province_name
            elif area_code.endswith('00'):  # 市级
                # 匹配市代码
                city_match = admin_df[admin_df['市代码'] == area_code]
                if not city_match.empty:
                    city_row = city_match.iloc[0]
                    province_code = city_row['省代码']
                    province_name = city_row['省']
                    city_name = city_row['市']
                    area_name = f"{province_name}{city_name}"
                    return province_code, area_name
            else:  # 县级
                # 匹配PAC（县级编码）
                county_match = admin_df[admin_df['PAC'] == area_code]
                if not county_match.empty:
                    county_row = county_match.iloc[0]
                    province_code = county_row['省代码']
                    province_name = county_row['省']
                    city_name = county_row['市']
                    county_name = county_row['县']
                    area_name = f"{province_name}{city_name}{county_name}"
                    return province_code, area_name
        
        # 如果没有匹配到，使用备用逻辑
        self.fjson.log(f"在行政区划编码文件中未找到匹配的区域代码: {area_code}，使用备用逻辑")
        return self._get_province_code_and_name_fallback(area_code)

    def _get_province_code_and_name_fallback(self, area_code: str) -> Tuple[str, str]:
        """备用逻辑：当无法从行政区划编码文件获取时使用"""
        # 如果是省级编码，直接使用
        if len(area_code) == 6 and area_code.endswith('0000'):
            return area_code, self.config["areaName"]
        
        # 如果是市级或县级编码，提取省级部分
        if len(area_code) == 6:
            province_code = area_code[:2] + "0000"
            return province_code, self.config["areaName"]
        
        # 其他情况，直接使用原编码和名称
        return area_code, self.config["areaName"]
    
    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str,nodata=0):
        """保存GeoTIFF文件"""
        # valid_count = np.sum(~np.isnan(data))
        # total_count = data.size
        # if valid_count > 0:
        #     min_val = np.nanmin(data)
        #     max_val = np.nanmax(data)
        #     mean_val = np.nanmean(data)
        #     print(f"最终结果: {valid_count}/{total_count} 有效像素, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")

        from osgeo import gdal

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
        
        self.fjson.log(f"GeoTIFF文件已保存: {output_path}")
    
    def _generate_qgis_map(self, resultfile: str, output_png: str) -> bool:
        """调用QGIS绘制专题图"""
        try:

            rasterStyle=os.path.join(os.getcwd(),"qgis_src","auxpath","colorbar",str(self.zoning_type+".qml"))
            template=os.path.join(os.getcwd(),"qgis_src","auxpath","template","template.qgs")
            cropfile=os.path.dirname(self.config.get('stationFilePath'))+"/crop.csv"
            QHfile=os.path.dirname(self.config.get('stationFilePath'))+"/QH.csv"
            province_vector=os.path.join(os.getcwd(),"qgis_src","auxpath","shp",str(self.area_code[:2]+"0000_sheng.shp"))
            city_vector=os.path.join(os.getcwd(),"qgis_src","auxpath","shp",str(self.area_code[:2]+"0000_shi.shp"))
            county_vector=os.path.join(os.getcwd(),"qgis_src","auxpath","shp",str(self.area_code[:2]+"0000_xian.shp"))
            city_copy_vector=os.path.join(os.getcwd(),"qgis_src","auxpath","shp",str(self.area_code[:2]+"0000_shi.shp"))
            county_copy_vector=os.path.join(os.getcwd(),"qgis_src","auxpath","shp",str(self.area_code[:2]+"0000_xian.shp"))
            style_dir=os.getcwd()+"/qgis_src/auxpath/shp"
            # cropfile="D:/project/农业气候资源普查和区划/code/china/depend_data/crop.csv"
            # QHfile="D:/project/农业气候资源普查和区划/code/china/depend_data/QH.csv"
            try:
                crop_ds = pd.read_csv(cropfile, dtype=str, encoding="utf-8")
            except:
                crop_ds = pd.read_csv(cropfile, dtype=str, encoding="gbk") 
            try:
                QH_ds = pd.read_csv(QHfile, dtype=str, encoding="utf-8")
            except:
                QH_ds = pd.read_csv(QHfile, dtype=str, encoding="gbk")          
            cropname_chinese=crop_ds[crop_ds["代码"]==self.crop_code]["名称"].iloc[0]    #作物类型 
            element_chinese=QH_ds[(QH_ds["区划类型代码"]==self.zoning_type)&(QH_ds["英文编码"]==self.element)]["区划类别名称"].iloc[0]
            json_params = {
                "resultfile": resultfile,
                "output": output_png,
                "rasterStyle": rasterStyle,
                "template":template,
                "areacode": self.area_code,
                "areaname": self.area_name,
                "cropname": self.crop_code,
                "cropname_chinese": cropname_chinese,
                "element_chinese":element_chinese,
                "startdate": self.start_date,
                "enddate": self.end_date,
            }
            from qgis_src.qgis_plot import main as main_plot
            main_plot(json_params["resultfile"],json_params["output"],json_params["rasterStyle"],json_params["template"],json_params["areacode"],json_params["areaname"],json_params["cropname_chinese"],json_params["element_chinese"],json_params["startdate"],json_params["enddate"],province_vector, city_vector,county_vector, city_copy_vector,county_copy_vector,style_dir)
            return True
        except Exception as e:
            self.fjson.log(f"调用QGIS绘图失败: {str(e)}")
            return False
    
    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _load_algorithms(self) -> Dict[str, Any]:
        """加载所有算法"""
        algorithms = {}
        
        # 加载插值算法
        try:
            from .interpolation.idw import IDWInterpolation
            algorithms["interpolation.idw"] = IDWInterpolation()
        except ImportError:
            print("警告: 无法加载IDW插值算法")
        
        try:
            from .interpolation.kriging import KrigingInterpolation
            algorithms["interpolation.krige"] = KrigingInterpolation()
        except ImportError:
            print("警告: 无法加载Kriging插值算法")
        
        try:
            from .interpolation.rbf import RBFInterpolation
            algorithms["interpolation.RBF"] = RBFInterpolation()
        except ImportError:
            print("警告: 无法加载RBF插值算法")
        
        try:
            from .interpolation.rf import RFInterpolation
            algorithms["interpolation.RF"] = RFInterpolation()
        except ImportError:
            print("警告: 无法加载RF插值算法")
        
        try:
            from .interpolation.lsm import LSMInterpolation
            algorithms["interpolation.LSM"] = LSMInterpolation()
        except ImportError:
            print("警告: 无法加载LSM插值算法")
        
        try:
            from .interpolation.mlp import MLPInterpolation
            algorithms["interpolation.MLP"] = MLPInterpolation()
        except ImportError:
            print("警告: 无法加载MLP插值算法")
        
        try:
            from .interpolation.lsm_idw import LSMIDWInterpolation
            algorithms["interpolation.lsm_idw"] = LSMIDWInterpolation()
        except ImportError as e:
            print(f"警告: 无法加载LSM-IDW插值算法: {str(e)}")
        
        # 加载分类算法
        try:
            from .classification.equal_interval import EqualIntervalClassification
            algorithms["classification.equal_interval"] = EqualIntervalClassification()
        except ImportError:
            print("警告: 无法加载等间隔分类算法")
        
        try:
            from .classification.natural_breaks import NaturalBreaksClassification
            algorithms["classification.natural_breaks"] = NaturalBreaksClassification()
        except ImportError:
            print("警告: 无法加载自然断点分类算法")

        try:
            from .classification.custom_thresholds import CustomClassification
            algorithms["classification.custom_thresholds"] = CustomClassification()
        except ImportError:
            print("警告: 无法加载自然断点分类算法")
                    
        return algorithms
