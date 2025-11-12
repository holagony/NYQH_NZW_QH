import numpy as np
from typing import Dict, Any, List

class ClassificationTool:
    """分类工具类 - 包含所有分类方法的静态方法"""
    
    @staticmethod
    def equal_interval(data: Any, params: Dict[str, Any] = None) -> Any:
        """等间隔分类"""
        if params is None:
            params = {}
        
        num_classes = params.get('num_classes', 5)
        
        if isinstance(data, dict):
            return ClassificationTool._equal_interval_dict(data, num_classes)
        else:
            return ClassificationTool._equal_interval_array(data, num_classes)
    
    @staticmethod
    def natural_breaks(data: Any, params: Dict[str, Any] = None) -> Any:
        """自然断点分类"""
        if params is None:
            params = {}
        
        num_classes = params.get('num_classes', 5)
        
        if isinstance(data, dict):
            return ClassificationTool._natural_breaks_dict(data, num_classes)
        else:
            return ClassificationTool._natural_breaks_array(data, num_classes)
    
    @staticmethod
    def custom_thresholds(data: Any, params: Dict[str, Any]) -> Any:
        """自定义阈值分类"""
        thresholds = params.get('thresholds', [])
        
        if isinstance(data, dict):
            return ClassificationTool._custom_thresholds_dict(data, thresholds)
        else:
            return ClassificationTool._custom_thresholds_array(data, thresholds)
    
    # 私有辅助方法
    @staticmethod
    def _equal_interval_dict(data: Dict[str, float], num_classes: int) -> Dict[str, int]:
        """字典数据的等间隔分类"""
        values = np.array([v for v in data.values() if not np.isnan(v)])
        
        if len(values) == 0:
            return {k: 0 for k in data.keys()}
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if min_val == max_val:
            return {k: 1 for k in data.keys()}
        
        interval = (max_val - min_val) / num_classes
        
        result = {}
        for key, value in data.items():
            if np.isnan(value):
                result[key] = 0
            else:
                class_idx = min(int((value - min_val) / interval) + 1, num_classes)
                result[key] = class_idx
        
        return result
    
    @staticmethod
    def _equal_interval_array(data: np.ndarray, num_classes: int) -> np.ndarray:
        """数组数据的等间隔分类"""
        if data.size == 0:
            return data
        
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        
        if min_val == max_val:
            return np.ones_like(data, dtype=np.int32)
        
        interval = (max_val - min_val) / num_classes
        
        result = np.zeros_like(data, dtype=np.int32)
        
        for i in range(1, num_classes + 1):
            lower_bound = min_val + (i - 1) * interval
            upper_bound = min_val + i * interval if i < num_classes else max_val + 0.001
            
            mask = (data >= lower_bound) & (data < upper_bound) & ~np.isnan(data)
            result[mask] = i
        
        result[np.isnan(data)] = 0
        return result
    
    @staticmethod
    def _natural_breaks_dict(data: Dict[str, float], num_classes: int) -> Dict[str, int]:
        """字典数据的自然断点分类"""
        values = np.array([v for v in data.values() if not np.isnan(v)])
        
        if len(values) == 0:
            return {k: 0 for k in data.keys()}
        
        breaks = ClassificationTool._jenks_breaks(values, num_classes)
        
        result = {}
        for key, value in data.items():
            if np.isnan(value):
                result[key] = 0
            else:
                for i in range(1, len(breaks)):
                    if value <= breaks[i]:
                        result[key] = i
                        break
                else:
                    result[key] = len(breaks) - 1
        
        return result
    
    @staticmethod
    def _natural_breaks_array(data: np.ndarray, num_classes: int) -> np.ndarray:
        """数组数据的自然断点分类"""
        if data.size == 0:
            return data
        
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.zeros_like(data, dtype=np.int32)
        
        breaks = ClassificationTool._jenks_breaks(valid_data, num_classes)
        
        result = np.zeros_like(data, dtype=np.int32)
        
        for i in range(1, len(breaks)):
            if i < len(breaks) - 1:
                mask = (data > breaks[i-1]) & (data <= breaks[i]) & ~np.isnan(data)
            else:
                mask = (data > breaks[i-1]) & ~np.isnan(data)
            
            result[mask] = i
        
        return result
    
    @staticmethod
    def _custom_thresholds_dict(data: Dict[str, float], thresholds: List[Dict]) -> Dict[str, int]:
        """字典数据的自定义阈值分类"""
        result = {}
        
        for key, value in data.items():
            if np.isnan(value):
                result[key] = 0
                continue
            
            classified = False
            for threshold in thresholds:
                level = threshold["level"]
                min_val = threshold.get("min", -np.inf)
                max_val = threshold.get("max", np.inf)
                
                if min_val <= value < max_val:
                    result[key] = level
                    classified = True
                    break
            
            if not classified:
                # 如果没有匹配的阈值，分类到最低级别
                min_level = min([t["level"] for t in thresholds])
                result[key] = min_level
        
        return result
    
    @staticmethod
    def _custom_thresholds_array(data: np.ndarray, thresholds: List[Dict]) -> np.ndarray:
        """数组数据的自定义阈值分类"""
        result = np.zeros_like(data, dtype=np.int32)
        result[np.isnan(data)] = 0
        
        for threshold in thresholds:
            level = threshold["level"]
            condition_mask = np.ones_like(data, dtype=bool)
            
            if "min" in threshold:
                condition_mask = condition_mask & (data >= threshold["min"])
            if "max" in threshold:
                condition_mask = condition_mask & (data < threshold["max"])
            
            result[condition_mask & ~np.isnan(data)] = level
        
        return result
    
    @staticmethod
    def _jenks_breaks(data: np.ndarray, num_classes: int) -> list:
        """Jenks自然断点算法"""
        sorted_data = np.sort(data)
        breaks = []
        
        for i in range(num_classes + 1):
            quantile = i / num_classes
            if quantile == 1.0:
                idx = len(sorted_data) - 1
            else:
                idx = int(quantile * len(sorted_data))
            
            if idx < len(sorted_data):
                breaks.append(sorted_data[idx])
        
        unique_breaks = []
        for break_val in breaks:
            if break_val not in unique_breaks:
                unique_breaks.append(break_val)
        
        return unique_breaks