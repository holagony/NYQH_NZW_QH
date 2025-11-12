from typing import Dict, Any
import numpy as np

class CustomClassification():
    """自定义分级算法"""
    
    def execute(self, data: Any, params: Dict[str, Any] = None) -> Any:
        """执行自定义分级"""
        if isinstance(data, dict):
            # 处理站点数据字典
            return self._classify_station_data(data, params)
        else:
            # 处理数组数据
            return self._classify_array_data(data, params)
    
    def _classify_station_data(self, station_data: Dict[str, float], params: Dict[str, Any] = None) -> Dict[str, int]:
        """对站点数据进行自定义分级"""
        if not station_data:
            return {}
        
        # 获取分级参数
        thresholds = params.get('thresholds', []) if params else []
        
        if not thresholds:
            # 如果没有提供阈值，返回默认级别
            return {k: 0 for k in station_data.keys()}
        
        # 对每个站点值进行分类
        result = {}
        for station_id, value in station_data.items():
            if np.isnan(value):
                result[station_id] = 0  # 无效值
            else:
                result[station_id] = self._classify_value(value, thresholds)
        
        return result
    
    def _classify_array_data(self, data: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        """对数组数据进行自定义分级 - 使用向量化操作提高性能"""
        if data.size == 0:
            return np.zeros_like(data, dtype=np.int32)
        
        # 获取分级参数
        thresholds = params.get('thresholds', []) if params else []
        
        if not thresholds:
            # 如果没有提供阈值，返回默认级别
            return np.zeros_like(data, dtype=np.int32)
        
        # 创建结果数组 - 使用int32类型
        result = np.zeros_like(data, dtype=np.int32)
        
        # 处理有效数据（非NaN）
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if valid_data.size == 0:
            return result
        
        # 使用向量化操作创建分类结果
        classified = np.zeros_like(valid_data, dtype=np.int32)
        
        # 对每个阈值应用条件
        for threshold in thresholds:
            min_val = self._parse_threshold_value(threshold.get('min', None))
            max_val = self._parse_threshold_value(threshold.get('max', None))
            level = threshold.get('level', 0)
            
            # 创建条件掩码
            condition = np.ones_like(valid_data, dtype=bool)
            
            # 应用最小值条件
            if min_val is not None:
                condition &= (valid_data >= min_val)
            
            # 应用最大值条件
            if max_val is not None:
                condition &= (valid_data <= max_val)
            
            # 将符合条件的值设置为对应级别
            classified[condition] = level
        
        # 将分类结果放回原数组
        result[valid_mask] = classified
        
        return result
    
    def _parse_threshold_value(self, value: Any) -> float:
        """解析阈值数值，处理空字符串等情况"""
        if value == "" or value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _classify_value(self, value: float, thresholds: list) -> int:
        """根据阈值对单个值进行分类"""
        for threshold in thresholds:
            min_val = self._parse_threshold_value(threshold.get('min', None))
            max_val = self._parse_threshold_value(threshold.get('max', None))
            level = threshold.get('level', 0)
            
            # 检查值是否在当前阈值范围内
            match = True
            
            # 检查最小值条件
            if min_val is not None:
                if value < min_val:
                    match = False
            
            # 检查最大值条件
            if max_val is not None:
                if value > max_val:
                    match = False
            
            # 如果匹配当前阈值范围，返回对应级别
            if match:
                return level
        
        # 如果没有匹配的阈值，返回0级
        return 0

# 性能测试和示例
if __name__ == "__main__":
    import time
    
    # 创建分类器实例
    classifier = CustomClassification()
    
    # 创建大量测试数据
    np.random.seed(42)
    large_array = np.random.normal(5, 3, 1000000)  # 100万个数据点
    large_array = np.where(large_array < 0, np.nan, large_array)  # 添加一些NaN值
    
    # 分级参数
    params = {
        "num_classes": 4,
        "thresholds": [
            {"min": "", "max": 3, "level": 4, "label": "低风险"},    
            {"min": 3, "max": 6, "level": 3, "label": "中风险"},  
            {"min": 6, "max": 9, "level": 2, "label": "高风险"},  
            {"min": 9, "max": "", "level": 1, "label": "极高风险"}   
        ]
    }
    
    # 测试性能
    start_time = time.time()
    result = classifier.execute(large_array, params)
    end_time = time.time()
    
    print(f"处理100万个数据点耗时: {end_time - start_time:.4f}秒")
    
    # 验证结果
    unique, counts = np.unique(result, return_counts=True)
    print("分级结果统计:")
    for level, count in zip(unique, counts):
        print(f"级别 {level}: {count} 个数据点")
    
    # 小数据测试
    print("\n=== 小数据测试 ===")
    small_array = np.array([2.5, 4.0, 7.2, 10.5, np.nan])
    small_result = classifier.execute(small_array, params)
    print(f"小数据: {small_array} -> {small_result}")
    
    # 站点数据测试
    station_data = {
        'station1': 2.5,
        'station2': 4.0,
        'station3': 7.2,
        'station4': 10.5,
        'station5': np.nan
    }
    station_result = classifier.execute(station_data, params)
    print(f"站点数据: {station_result}")