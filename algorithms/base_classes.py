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

class BaseAlgorithm(ABC):
    """算法基类"""
    
    @abstractmethod
    def execute(self, data: Any, params: Dict[str, Any] = None) -> Any:
        """执行算法"""
        pass