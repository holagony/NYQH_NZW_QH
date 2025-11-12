from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import numpy as np

class ZoningType(Enum):
    """区划类型枚举"""
    PLANTING = "ZZ"  # 种植区划
    QUALITY = "PZ"   # 品质区划
    YIELD_ = "CL"    # 产量区划
    DISASTER = "ZH"  # 气象灾害区划
    PEST = "BH"      # 病虫害区划


@dataclass
class CalculationParams:
    """计算参数数据类"""
    task_id: str
    algo_from: str
    zoning_type: str
    input_file_path: str
    area_code: str
    area_name: str
    start_date: str
    end_date: str
    interpolation_method: str
    grid_file_path: str
    dem_file_path: str
    shp_file_path: str
    lulc_file_path: str
    other_files_path: str
    algo_output_file_path: str
    result_log_path: str
    result_json_path: str
    result_flow_path: str
    result_path: str
    
    # 可选参数放在后面
    element: str = None
    crop_code: str = None
    station_file_path: str = None
    classification_method: str = "equal_interval"
    multiprocess: bool = False
    num_processes: Optional[int] = None
    save_intermediate: bool = False
    # quality_indicator: str = None  # 新增品质指标字段

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建参数对象"""
        return cls(
            task_id=config_dict['taskId'],
            algo_from=config_dict['algoFrom'],
            zoning_type=config_dict['zoningType'],
            input_file_path=config_dict['inputFilePath'],
            area_code=config_dict['areaCode'],
            area_name=config_dict['areaName'],
            start_date=config_dict['startDate'],
            end_date=config_dict['endDate'],
            interpolation_method=config_dict['interpolationMethod'],
            grid_file_path=config_dict['gridFilePath'],
            dem_file_path=config_dict['demFilePath'],
            shp_file_path=config_dict['shpFilePath'],
            lulc_file_path=config_dict['lulcFilePath'],
            other_files_path=config_dict['otherFilesPath'],
            algo_output_file_path=config_dict['algoOuputFilePath'],
            result_log_path=config_dict['resultLogPath'],
            result_json_path=config_dict['resultJsonPath'],
            result_flow_path=config_dict['resultFlowPath'],
            result_path=config_dict['resultPath'],
            element=config_dict.get('element'),
            crop_code=config_dict.get('cropCode'),
            station_file_path=config_dict.get('stationFilePath'),
            classification_method=config_dict.get('classificationMethod', 'equal_interval'),
            multiprocess=config_dict.get('multiprocess', False),
            num_processes=config_dict.get('num_processes'),
            save_intermediate=config_dict.get('save_intermediate', False),
            # quality_indicator=config_dict.get('qualityIndicator')  # 新增
        )
        
        
# @dataclass
# class CalculationParams:
#     """计算参数数据类"""
#     task_id: str
#     algo_from: str
#     zoning_type: str
#     input_file_path: str
#     area_code: str
#     area_name: str
#     element: str
#     start_date: str
#     end_date: str
#     interpolation_method: str
#     grid_file_path: str
#     dem_file_path: str
#     shp_file_path: str
#     lulc_file_path: str
#     other_files_path: str
#     algo_output_file_path: str
#     result_log_path: str
#     result_json_path: str
#     result_flow_path: str
#     result_path: str
#     crop_code: str = None  # 新增作物代码字段
#     station_file_path: str = None  # 新增站点信息文件路径
#     # 新增分类方法参数
#     classification_method: str = "equal_interval"
#     # 其他可选参数
#     multiprocess: bool = False
#     num_processes: Optional[int] = None
#     save_intermediate: bool = False

#     @classmethod
#     def from_dict(cls, config_dict: Dict[str, Any]):
#         """从字典创建参数对象"""
#         return cls(
#             task_id=config_dict['taskId'],
#             algo_from=config_dict['algoFrom'],
#             zoning_type=config_dict['zoningType'],
#             input_file_path=config_dict['inputFilePath'],
#             area_code=config_dict['areaCode'],
#             area_name=config_dict['areaName'],
#             element=config_dict['element'],
#             start_date=config_dict['startDate'],
#             end_date=config_dict['endDate'],
#             interpolation_method=config_dict['interpolationMethod'],
#             grid_file_path=config_dict['gridFilePath'],
#             dem_file_path=config_dict['demFilePath'],
#             shp_file_path=config_dict['shpFilePath'],
#             lulc_file_path=config_dict['lulcFilePath'],
#             other_files_path=config_dict['otherFilesPath'],
#             algo_output_file_path=config_dict['algoOuputFilePath'],
#             result_log_path=config_dict['resultLogPath'],
#             result_json_path=config_dict['resultJsonPath'],
#             result_flow_path=config_dict['resultFlowPath'],
#             result_path=config_dict['resultPath'],
#             crop_code=config_dict.get('cropCode'),
#             station_file_path=config_dict.get('stationFilePath'),
#             classification_method=config_dict.get('classificationMethod', 'equal_interval'),
#             multiprocess=config_dict.get('multiprocess', False),
#             num_processes=config_dict.get('num_processes'),
#             save_intermediate=config_dict.get('save_intermediate', False)
#         )

class BaseCrop(ABC):
    """农作物基类"""
    
    def __init__(self, crop_code: str):
        self.crop_code = crop_code
        self.config = self.load_config()
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """加载农作物特定配置"""
        pass
    
    @abstractmethod
    def get_required_data(self, zoning_type: str, subtype: str = None) -> List[str]:
        """获取所需数据类型"""
        pass
    
    @abstractmethod
    def get_province_config(self, province_code: str, zoning_type: str, subtype: str = None) -> Dict[str, Any]:
        """获取省份特定配置"""
        pass

# core/base_classes.py 中 BaseZoningType 类的修改

# class BaseZoningType(ABC):
#     """区划类型基类 - 扩展版本"""
    
#     def __init__(self, zoning_code: str, calculator=None):
#         self.zoning_code = zoning_code
#         self.calculator = calculator
#         self.config = self.load_config()
    
#     @abstractmethod
#     def load_config(self) -> Dict[str, Any]:
#         """加载区划类型配置"""
#         pass
    
#     @abstractmethod
#     def calculate(self, data: Dict[str, Any], params: CalculationParams) -> Any:
#         """执行区划计算（站点级别）"""
#         pass
    
#     def calculate_grid(self, interpolated_indicators: Dict[str, Any], params: CalculationParams, 
#                       crop_config: Dict[str, Any]) -> Any:
#         """执行栅格级别的区划计算（用于before插值顺序）"""
#         # 默认实现，子类可以重写
#         raise NotImplementedError("该区划类型不支持栅格级别计算")
    
#     def _get_crop_config(self, crop_code: str, province_code: str, zoning_type: str, element: str = None) -> Dict[str, Any]:
#         """获取农作物配置（简化实现）"""
#         # 实际应该从calculator中获取
#         try:
#             from crops.soybean import Soybean
#             from crops.citrus import Citrus
#             from crops.corn import Corn
            
#             if crop_code == "soybean":
#                 crop = Soybean()
#             elif crop_code == "citrus":
#                 crop = Citrus() 
#             elif crop_code == "corn":
#                 crop = Corn()
#             else:
#                 return {}
                
#             return crop.get_province_config(province_code, zoning_type, element)
#         except:
#             return {}
    
#     def _get_algorithm(self, algorithm_name: str) -> Any:
#         """获取算法实例"""
#         if self.calculator is None:
#             raise ValueError("计算器实例未设置，无法获取算法")
#         return self.calculator._get_algorithm(algorithm_name)
    
#     # 在base_classes.py的_classify方法中改进
#     def _classify(self, data: Any, method: str, classification_params: Dict[str, Any] = None) -> Any:
#         """数据分级"""
#         if classification_params is None:
#             classification_params = {}
        
#         # 特殊处理风险表分类方法
#         if method.endswith('_risk_table'):
#             return self._risk_table_classify(data, method)
        
#         try:
#             classifier = self._get_algorithm(f"classification.{method}")
#             return classifier.execute(data, classification_params)
#         except Exception as e:
#             print(f"分类算法执行失败: {str(e)}")
#             #  fallback到默认分类
#             classifier = self._get_algorithm("classification.equal_interval")
#             return classifier.execute(data, classification_params)
    
#     def _interpolate_data(self, data: Any, method: str, params: Dict[str, Any] = None) -> Any:
#         """数据插值"""
#         if params is None:
#             params = {}
        
#         interpolator = self._get_algorithm(f"interpolation.{method}")
#         return interpolator.execute(data, params)
    
#     def _risk_table_classify(self, data: Any, risk_table_name: str) -> Any:
#         """风险表分类方法"""
#         # 这里实现具体的风险表分类逻辑
#         # 根据不同的风险表名称使用不同的分类标准
#         if risk_table_name == "sclerotinia_risk_table":
#             return self._sclerotinia_risk_classify(data)
#         elif risk_table_name == "bean_moth_risk_table":
#             return self._bean_moth_risk_classify(data)
#         elif risk_table_name == "high_temperature_risk_table":
#             return self._high_temperature_risk_classify(data)
#         else:
#             # 默认使用等间隔分类
#             classifier = self._get_algorithm("classification.equal_interval")
#             return classifier.execute(data)
    
#     def _sclerotinia_risk_classify(self, data: Any) -> Any:
#         """菌核病风险分级"""
#         if isinstance(data, dict):
#             result = {}
#             for station_id, value in data.items():
#                 if np.isnan(value):
#                     result[station_id] = 0
#                 elif value >= 9:
#                     result[station_id] = 1  # 极高风险
#                 elif value >= 6:
#                     result[station_id] = 2  # 高风险
#                 elif value >= 3:
#                     result[station_id] = 3  # 中风险
#                 else:
#                     result[station_id] = 4  # 低风险
#             return result
#         else:
#             # 处理数组数据
#             result = np.zeros_like(data, dtype=int)
#             result[(data >= 9) & ~np.isnan(data)] = 1
#             result[(data >= 6) & (data < 9) & ~np.isnan(data)] = 2
#             result[(data >= 3) & (data < 6) & ~np.isnan(data)] = 3
#             result[(data < 3) & ~np.isnan(data)] = 4
#             result[np.isnan(data)] = 0
#             return result
    
#     def _bean_moth_risk_classify(self, data: Any) -> Any:
#         """食心虫风险分级"""
#         if isinstance(data, dict):
#             result = {}
#             for station_id, value in data.items():
#                 if np.isnan(value):
#                     result[station_id] = 0
#                 elif value > 2.0:
#                     result[station_id] = 1  # 极高发生区
#                 elif value > 1.5:
#                     result[station_id] = 2  # 高发生区
#                 elif value > 1.0:
#                     result[station_id] = 3  # 中发生区
#                 else:
#                     result[station_id] = 4  # 低发生区
#             return result
#         else:
#             # 处理数组数据
#             result = np.zeros_like(data, dtype=int)
#             result[(data > 2.0) & ~np.isnan(data)] = 1
#             result[(data > 1.5) & (data <= 2.0) & ~np.isnan(data)] = 2
#             result[(data > 1.0) & (data <= 1.5) & ~np.isnan(data)] = 3
#             result[(data <= 1.0) & ~np.isnan(data)] = 4
#             result[np.isnan(data)] = 0
#             return result
    
#     def _high_temperature_risk_classify(self, data: Any) -> Any:
#         """高温灾害风险分级"""
#         if isinstance(data, dict):
#             result = {}
#             for station_id, value in data.items():
#                 if np.isnan(value):
#                     result[station_id] = 0
#                 elif value >= 20:
#                     result[station_id] = 1  # 极高风险
#                 elif value >= 10:
#                     result[station_id] = 2  # 高风险
#                 elif value >= 5:
#                     result[station_id] = 3  # 中风险
#                 else:
#                     result[station_id] = 4  # 低风险
#             return result
#         else:
#             # 处理数组数据
#             result = np.zeros_like(data, dtype=int)
#             result[(data >= 20) & ~np.isnan(data)] = 1
#             result[(data >= 10) & (data < 20) & ~np.isnan(data)] = 2
#             result[(data >= 5) & (data < 10) & ~np.isnan(data)] = 3
#             result[(data < 5) & ~np.isnan(data)] = 4
#             result[np.isnan(data)] = 0
#             return result

class BaseAlgorithm(ABC):
    """算法基类"""
    
    @abstractmethod
    def execute(self, data: Any, params: Dict[str, Any] = None) -> Any:
        """执行算法"""
        pass