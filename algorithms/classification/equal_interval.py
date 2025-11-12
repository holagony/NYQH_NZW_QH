from typing import Dict, Any
import numpy as np

class EqualIntervalClassification():
    """等间隔分级算法"""
    
    def execute(self, data: Any, params: Dict[str, Any] = None) -> Any:
        """执行等间隔分级"""
        if isinstance(data, dict):
            # 处理站点数据字典
            return self._classify_station_data(data, params)
        else:
            # 处理数组数据
            return self._classify_array_data(data, params)
    
    def _classify_station_data(self, station_data: Dict[str, float], params: Dict[str, Any] = None) -> Dict[str, int]:
        """对站点数据进行等间隔分级"""
        if not station_data:
            return {}
        
        # 提取有效值
        values = np.array([v for v in station_data.values() if not np.isnan(v)])
        
        if len(values) == 0:
            return {k: 0 for k in station_data.keys()}
        
        # 获取分级参数
        num_classes = params.get('num_classes', 5) if params else 5
        
        # 计算等间隔的分级界限
        min_val = np.min(values)
        max_val = np.max(values)
        
        if min_val == max_val:
            # 所有值相同，都分为同一类
            return {k: 1 for k in station_data.keys()}
        
        # 计算间隔
        interval = (max_val - min_val) / num_classes
        
        # 对每个站点值进行分类
        result = {}
        for station_id, value in station_data.items():
            if np.isnan(value):
                result[station_id] = 0  # 无效值
            else:
                # 计算类别 (1 到 num_classes)
                class_idx = min(int((value - min_val) / interval) + 1, num_classes)
                result[station_id] = class_idx
        
        return result

    def _classify_array_data(self, data: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        """对数组数据进行等间隔分级"""
        if data.size == 0:
            return data
        
        # 获取分级参数
        num_classes = params.get('num_classes', 5) if params else 5
        
        # 计算等间隔的分级界限
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        
        if min_val == max_val:
            # 所有值相同，都分为同一类
            return np.ones_like(data, dtype=np.int32)
        
        # 计算间隔
        interval = (max_val - min_val) / num_classes
        
        # 创建结果数组 - 使用int32类型
        result = np.zeros_like(data, dtype=np.int32)
        
        # 对每个值进行分类
        for i in range(1, num_classes + 1):
            lower_bound = min_val + (i - 1) * interval
            upper_bound = min_val + i * interval if i < num_classes else max_val + 0.001  # 包含最大值
            
            mask = (data >= lower_bound) & (data < upper_bound) & ~np.isnan(data)
            result[mask] = i
        
        # 处理NaN值
        result[np.isnan(data)] = 0
        
        return result