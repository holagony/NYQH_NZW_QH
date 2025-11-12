#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版区划处理器 - 直接高效的流程
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import importlib
from typing import List,Dict,Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from .data_manager import DataManager
from configs.base_config import BaseConfig


class ZoningProcessor:
    """区划处理器"""
    
    def __init__(self, config, fjson, rjson):
        self.config = config
        self.fjson = fjson
        self.rjson = rjson
        
        # 初始化配置管理器
        self.config_manager = BaseConfig("configs")
        
        # 初始化参数
        self._init_parameters()
        
        # 初始化数据管理器
        self.data_manager =DataManager(config['inputFilePath'],
                    station_file=config.get('stationFilePath'),
                    multiprocess=config.get('multiprocess', True),
                    num_processes=config.get('num_processes',16)
                    )
        # 插值和分类方法初始化
        self._algorithms = self._load_algorithms()
        
    def _init_parameters(self):
        """初始化参数"""
        self.area_code = self.config.get("areaCode", "150000")
        self.area_name = self.config.get("areaName", "内蒙古自治区")
        self.crop_code = self.config.get("cropCode", "soybean")
        self.zoning_type = self.config.get("zoningType", "BH")
        self.element = self.config.get("element", "sclerotinia")

        # 新增：算法来源参数，用于控制算法模块和配置文件
        self.algo_from = self.config.get("algoFrom", None)
        
        # 时间参数
        self.start_date = self.config.get("startDate", "19910101")
        self.end_date = self.config.get("endDate", "20201231")
        
        # 根据 algoFrom 确定配置键
        if self.algo_from:
            # 使用 algoFrom 指定的算法配置
            config_key = self.algo_from
            print(f"使用跨区域算法配置: {config_key}")
        else:
            # 使用默认的区域-作物-区划类型配置
            config_key = f"zoning_{self.area_code}_{self.crop_code}_{self.zoning_type}"
            print(f"使用默认算法配置: {config_key}")
        
        # 合并配置
        user_algorithm_config = self.config.get('algorithmConfig')
        self.algorithm_config = self.config_manager.merge_configs(
            user_algorithm_config, config_key, self.crop_code, self.zoning_type
        )
        
        # # 合并配置
        # user_algorithm_config = self.config.get('algorithmConfig')
        # self.algorithm_config = self.config_manager.merge_configs(
        #     user_algorithm_config, self.area_code, self.crop_code, self.zoning_type
        # )
        # 获取指标配置
        if self.element:
            self.algorithm_config = self.algorithm_config[self.element]

        self.indicator_configs = self.algorithm_config.get("indicators")
        # 文件路径
        self.result_path = self.config.get("resultPath", "./results")
        Path(self.result_path).mkdir(parents=True, exist_ok=True)
    
        # 设置中间结果输出路径
        intermediate_dir = Path(self.result_path) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_output = intermediate_dir / "intermediate.csv"    
        
        self.fjson.log(f"配置加载完成: {config_key}")
    
    def process(self):
        """执行简化的区划处理流程"""
        try:
            self.fjson.info("开始数据准备")
            
            # 1. 获取站点列表
            station_ids = self.data_manager.get_stations_by_province(self.area_code)
            self.fjson.log(f"找到 {len(station_ids)} 个站点")
            
            # 2. 批量计算指标
            self.fjson.info("开始批量计算指标")
            # 计算所有站点的指标值
            station_indicators_df = self.data_manager.calculate_indicators_for_stations(
                station_ids, self.indicator_configs, self.start_date, self.end_date, self.intermediate_output
            )  
            # 读取站点数据
            station_indicators, station_coords = self._prepare_station_data(self.intermediate_output, self.indicator_configs)
          
            # 3. 执行区划计算
            self.fjson.info("开始区划计算")
            
            zoning_result = self._execute_zoning_calculation(station_indicators,station_coords)
            
            # 4. 保存结果
            self.fjson.info("开始输出结果")
            self._save_results(zoning_result)
            
            self.fjson.info("区划处理完成")
            return True
            
        except Exception as e:
            self.fjson.info(f"区划处理失败: {str(e)}", codeId='1')
            import traceback
            self.fjson.log(f"详细错误: {traceback.format_exc()}")
            return False


    def _prepare_station_data(self, station_indicators_df, indicator_configs):
        """准备站点数据"""
        # if station_indicators_df.empty:
        #     return {}, {}
        
        try:           # 读取CSV文件
            station_indicators_df = pd.read_csv(station_indicators_df, dtype=str, encoding="gbk")
        except:
            station_indicators_df = pd.read_csv(station_indicators_df, dtype=str, encoding="utf-8")
        
        print("准备站点数据...")
        
        # 使用配置中的指标名称
        indicator_names = list(indicator_configs.keys())
        print(f"期望的指标名称: {indicator_names}")
        
        # 检查DataFrame中实际存在的列
        actual_columns = set(station_indicators_df.columns)
        print(f"DataFrame中的列: {list(actual_columns)}")
        
        # 找出匹配的指标列
        matched_indicators = [col for col in indicator_names if col in actual_columns]
        print(f"匹配的指标列: {matched_indicators}")
        
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
                if value:
                    station_data[indicator_name] = float(value)
                else:
                    station_data[indicator_name] = np.nan
                # if pd.notna(value) and isinstance(value, (int, float)):
                #     station_data[indicator_name] = float(value)
                # else:
                #     station_data[indicator_name] = np.nan
            
            station_indicators[station_id] = station_data
        
        # 统计有效数据
        valid_stations = 0
        for station_id, data in station_indicators.items():
            if any(not np.isnan(val) for val in data.values()):
                valid_stations += 1
        
        print(f"成功准备 {valid_stations}/{len(station_indicators)} 个站点的有效数据")
        return station_indicators, station_coords


    def _execute_zoning_calculation(self, indicators_data: Dict[str, Any], station_coords: Dict[str, Any]) -> Dict[str, Any]:
        """ 支持跨区域算法调用"""
        
        # 根据 algoFrom 确定计算器名称
        if self.algo_from:
            calculator_name = self.algo_from
            print(f"使用跨区域计算器: {calculator_name}")
        else:
            calculator_name = f"zoning_{self.area_code}_{self.crop_code}_{self.zoning_type}"
            print(f"使用默认计算器: {calculator_name}")
        
        module_path = f"algorithms.zoning_calculator.{calculator_name}"
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            self.fjson.log(f"模块导入成功: {module_path}")
            
            # 获取计算器类名
            class_name = self._get_calculator_class_name()
            self.fjson.log(f"查找计算器类: {class_name}")
            
            # 获取计算器类
            if hasattr(module, class_name):
                calculator_class = getattr(module, class_name)
                self.fjson.log(f"找到计算器类: {calculator_class}")
                
                # 准备计算参数
                calc_params = {
                    'station_indicators': indicators_data,
                    'station_coords': station_coords,
                    'algorithmConfig': self.algorithm_config,
                    'algorithms': self._algorithms,
                    'config': self.config
                }
                
                self.fjson.log(f"调用计算器参数: {list(calc_params.keys())}")
                
                # 执行计算
                calculator_instance = calculator_class()
                result = calculator_instance.calculate(calc_params)
                self.fjson.log("专用计算器执行成功")
                
                return result
            else:
                self.fjson.log(f"模块 {module_path} 中没有找到类 {class_name}")
                raise AttributeError(f"类 {class_name} 不存在")
                
        except ImportError as e:
            self.fjson.log(f"专用计算器导入失败: {str(e)}")
            self.fjson.log("尝试使用通用计算器")
            return self._fallback_to_generic_calculator(indicators_data, station_coords)
            
        except Exception as e:
            self.fjson.log(f"计算器执行异常: {str(e)}")
            raise

    def _get_calculator_class_name(self) -> str:
        """获取计算器类名 - 支持跨区域计算"""
        crop_name = self.crop_code.capitalize()
        
        # 更详细的类名映射
        class_name_map = {
            "BH": f"{crop_name}PestZoning",           # 病虫害区划
            "ZZ": f"{crop_name}PlantingZoning",       # 种植区划  
            "PZ": f"{crop_name}QualityZoning",        # 品质区划
            "CL": f"{crop_name}YieldZoning",          # 产量区划
            "ZH": f"{crop_name}DisasterZoning",       # 气象灾害区划
            "QH": f"{crop_name}ComprehensiveZoning",  # 综合区划
        }

        class_name = class_name_map.get(self.zoning_type, f"{crop_name}Zoning")
        self.fjson.log(f"类名映射: {self.zoning_type} -> {class_name}")
        
        return class_name     

    def _fallback_to_generic_calculator(self, indicators_data: Dict[str, Any], station_coords: Dict[str, Any]) -> Dict[str, Any]:
        """备用通用计算器"""
        try:
            from algorithms.zoning_calculator.zoning_calculator import GenericZoningCalculator
            
            calc_params = {
                'station_indicators': indicators_data,
                'station_coords': station_coords,
                'grid_path': self.config['gridFilePath'],
                'dem_path': self.config.get('demFilePath'),
                'algorithmConfig': self.algorithm_config
            }
            
            calculator = GenericZoningCalculator()
            result = calculator.calculate(calc_params)
            self.fjson.log("通用计算器执行成功")
            return result
            
        except Exception as e:
            self.fjson.log(f"通用计算器也失败: {str(e)}")
            raise
           
    # def _execute_zoning_calculation(self, indicators_data: Dict[str, Any],station_coords: Dict[str, Any]) -> Dict[str, Any]:
    #     """修复的区划计算方法 - 确保正确调用算法"""
    #     calculator_name = f"zoning_{self.area_code}_{self.crop_code}_{self.zoning_type}"
    #     self.fjson.log(f"尝试调用计算器: {calculator_name}")

    #     module_path = f"zoning_calculator.{calculator_name}"
              
    #     try:
    #         # 构建完整的模块路径
    #         module_path = f"algorithms.zoning_calculator.{calculator_name}"
    #         # self.fjson.log(f"尝试导入模块: {module_path}")
            
    #         # 动态导入模块
    #         module = importlib.import_module(module_path)
    #         self.fjson.log(f"模块导入成功: {module_path}")
            
    #         # 获取计算器类名
    #         class_name = self._get_calculator_class_name()
    #         # self.fjson.log(f"查找计算器类: {class_name}")
            
    #         # 获取计算器类
    #         if hasattr(module, class_name):
    #             calculator_class = getattr(module, class_name)
    #             self.fjson.log(f"找到计算器类: {calculator_class}")
                
    #             # 准备计算参数
    #             calc_params = {
    #                 'station_indicators': indicators_data,
    #                 'station_coords': station_coords,
    #                 'algorithmConfig': self.algorithm_config,
    #                 'algorithms': self._algorithms,
    #                 'config':self.config
    #             }
                
    #             self.fjson.log(f"调用计算器参数: {list(calc_params.keys())}")
                
    #             # 执行计算
    #             # result = calculator_class.calculate(calc_params)
    #             q = calculator_class()
    #             result = q.calculate(calc_params)
    #             self.fjson.log("区划算法执行成功")
                
    #             return result
    #         else:
    #             self.fjson.log(f"模块 {module_path} 中没有找到类 {class_name}")
    #             raise AttributeError(f"类 {class_name} 不存在")
                
    #     except ImportError as e:
    #         self.fjson.log(f"专用计算器导入失败: {str(e)}")
    #         self.fjson.log("尝试使用通用计算器")
            
    #         # 方法2: 使用通用计算器
    #         try:
    #             from zoning_calculator.zoning_calculator import GenericZoningCalculator
                
    #             calc_params = {
    #                 'station_indicators': indicators_data['station_indicators'],
    #                 'station_coords': indicators_data['station_coords'],
    #                 'grid_path': self.config['gridFilePath'],
    #                 'dem_path': self.config.get('demFilePath'),
    #                 'algorithmConfig': self.algorithm_config
    #             }
                
    #             q = GenericZoningCalculator()
    #             result = q.calculate(calc_params)
    #             self.fjson.log("通用计算器执行成功")
    #             return result
                
    #         except Exception as e2:
    #             self.fjson.log(f"通用计算器也失败: {str(e2)}")
    #             raise
                
    #     except Exception as e:
    #         self.fjson.log(f"计算器执行异常: {str(e)}")
    #         raise
    
    # def _get_calculator_class_name(self) -> str:
    #     """获取计算器类名 - 修复版本"""
    #     crop_name = self.crop_code.capitalize()
        
    #     # 更详细的类名映射
    #     class_name_map = {
    #         "BH": f"{crop_name}PestZoning",           # 病虫害区划
    #         "ZZ": f"{crop_name}PlantingZoning",       # 种植区划  
    #         "PZ": f"{crop_name}QualityZoning",        # 品质区划
    #         "CL": f"{crop_name}YieldZoning",          # 产量区划
    #         "ZH": f"{crop_name}DisasterZoning",       # 气象灾害区划
    #         "ZH": f"{crop_name}ComprehensiveZoning",  # 综合区划
    #     }

    #     class_name = class_name_map.get(self.zoning_type, f"{crop_name}Zoning")
    #     self.fjson.log(f"类名映射: {self.zoning_type} -> {class_name}")
        
    #     return class_name
      
    def _save_results(self, zoning_result: Dict[str, Any]):
        """保存结果"""
        # 生成输出文件名
        tif_filename = self._generate_output_filename("tif")
        tif_path = os.path.join(self.result_path, tif_filename)
        
        # 保存TIFF文件
        self._save_geotiff(zoning_result['data'], zoning_result['meta'], tif_path)
        
        exit(0)
        # 生成PNG专题图
        png_filename = self._generate_output_filename("png")
        png_path = os.path.join(self.result_path, png_filename)
        
        # 调用QGIS生成专题图
        self._generate_qgis_map(tif_path, png_path)
        self.fjson.log(f"专题图已生成: {png_path}")
        
        # 保存结果信息
        result_info = {
            'task_id': self.config.get('taskId', ''),
            'area_code': self.area_code,
            'area_name': self.area_name,
            'crop_code': self.crop_code,
            'zoning_type': self.zoning_type,
            'element': self.element,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'output_files': {
                'tif': tif_path,
                'png': png_path
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存JSON文件
        json_filename = self._generate_output_filename("json")
        json_path = os.path.join(self.result_path, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        # 记录结果
        self.rjson.info('output_files', [tif_path, png_path, json_path])
        self.rjson.info('processing_time', datetime.now().isoformat())
        
        self.fjson.log(f"结果已保存: {tif_path}, {png_path}, {json_path}")
    
    def _generate_output_filename(self, file_type: str) -> str:
        """生成输出文件名"""
        product_type = "PR"
        crop_code = self.crop_code.upper()[:4]
        factor_code = self._get_factor_code(self.element)
        
        base_name = f"Q_{product_type}_{crop_code}-{self.zoning_type}_{self.area_code}_{factor_code}"
        
        if file_type == "tif":
            return f"{base_name}.tif"
        elif file_type == "png":
            return f"{base_name}.png"
        elif file_type == "json":
            return f"{base_name}.json"
        else:
            return f"{base_name}.{file_type}"
    
    def _get_factor_code(self, element: str) -> str:
        """获取因子代码"""
        factor_codes = {
            "protein": "001",
            "fat": "002",
            "sclerotinia": "101",
            "bean_moth": "102", 
            "high_temperature": "201"
        }
        return factor_codes.get(element, "000")
    
    def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str,nodata=0):
        """保存GeoTIFF文件"""
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
            json_params = {
                "resultfile": resultfile,
                "output": output_png,
                "areacode": self.area_code,
                "areaname": self.area_name,
                "cropname": self.crop_code,
                "zoningtype": self.zoning_type,
                "startdate": self.start_date,
                "enddate": self.end_date
            }
            
            # from qgis_src.main import main as main_plot
            # main_plot(json_params)
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
