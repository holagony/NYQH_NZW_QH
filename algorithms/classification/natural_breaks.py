from typing import Dict, Any
import numpy as np

class NaturalBreaksClassification():
    """自然断点分级算法（Jenks自然断点法）- 反转等级"""
    
    def execute(self, data: Any, params: Dict[str, Any] = None) -> Any:
        """执行自然断点分级（等级反转）"""
        if isinstance(data, dict):
            # 处理站点数据字典
            return self._classify_station_data(data, params)
        else:
            # 处理数组数据
            return self._classify_array_data(data, params)
    
    def _classify_station_data(self, station_data: Dict[str, float], params: Dict[str, Any] = None) -> Dict[str, int]:
        """对站点数据进行自然断点分级（等级反转）"""
        if not station_data:
            return {}
        
        # 提取有效值
        values = np.array([v for v in station_data.values() if not np.isnan(v)])
        
        if len(values) == 0:
            return {k: 0 for k in station_data.keys()}
        
        # 获取分级参数
        num_classes = params.get('num_classes', 5) if params else 5
        
        # 计算自然断点
        breaks = self._jenks_breaks(values, num_classes)
        
        # 对每个站点值进行分类（等级反转）
        result = {}
        for station_id, value in station_data.items():
            if np.isnan(value):
                result[station_id] = 0  # 无效值
            else:
                # 查找值所在的区间
                for i in range(1, len(breaks)):
                    if value <= breaks[i]:
                        # 等级反转：原本的等级i变成 (num_classes - i + 1)
                        result[station_id] = num_classes - i + 1
                        break
                else:
                    result[station_id] = 1  # 最高值对应最低等级（反转后）
        
        return result
    
    def _classify_array_data(self, data: np.ndarray, params: Dict[str, Any] = None) -> np.ndarray:
        """对数组数据进行自然断点分级（等级反转）"""
        if data.size == 0:
            return data
        
        # 获取分级参数
        num_classes = params.get('num_classes', 4) if params else 4
        
        # 提取有效值
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return np.zeros_like(data, dtype=np.int32)
        
        # 计算自然断点
        breaks = self._jenks_breaks(valid_data, num_classes)
        
        # 创建结果数组 - 使用int32类型
        result = np.zeros_like(data, dtype=np.int32)
        
        # 对每个值进行分类（等级反转）
        for i in range(1, len(breaks)):
            # 计算反转后的等级
            reversed_class = num_classes - i + 1
            
            if i < len(breaks) - 1:
                mask = (data > breaks[i-1]) & (data <= breaks[i]) & ~np.isnan(data)
            else:
                mask = (data > breaks[i-1]) & ~np.isnan(data)
            
            result[mask] = reversed_class
        
        return result

    
    # def _jenks_breaks(self, data: np.ndarray, num_classes: int) -> list:
    #     """计算Jenks自然断点"""
    #     # 对数据进行排序
    #     sorted_data = np.sort(data)
    #
    #     # 使用分位数计算断点
    #     breaks = []
    #     for i in range(num_classes + 1):
    #         quantile = i / num_classes
    #         if quantile == 1.0:
    #             idx = len(sorted_data) - 1
    #         else:
    #             idx = int(quantile * len(sorted_data))
    #
    #         if idx < len(sorted_data):
    #             breaks.append(sorted_data[idx])
    #
    #     # 确保断点是唯一的
    #     unique_breaks = []
    #     for break_val in breaks:
    #         if break_val not in unique_breaks:
    #             unique_breaks.append(break_val)
    #
    #     # 如果断点数量不足，补充断点
    #     while len(unique_breaks) < num_classes + 1:
    #         if len(unique_breaks) == 0:
    #             unique_breaks.append(np.min(data))
    #         unique_breaks.append(np.max(data))
    #         # 去重
    #         unique_breaks = list(set(unique_breaks))
    #         unique_breaks.sort()
    #
    #     return unique_breaks
    def _jenks_breaks(self, data: np.ndarray, num_classes: int) -> list:
        """计算Jenks自然断点"""
        # 对数据进行排序
        sorted_data = np.sort(data)

        # 如果数据值太少或全部相同，使用等间距断点
        unique_values = np.unique(sorted_data)

        if len(unique_values) <= num_classes:
            # 数据不足，使用简单断点
            if len(unique_values) == 1:
                # 所有值相同，创建等间距断点
                value = unique_values[0]
                return [value - 0.1, value, value + 0.1][:num_classes + 1]
            else:
                # 使用现有唯一值作为断点，必要时补充
                breaks = list(unique_values)
                if len(breaks) < num_classes + 1:
                    # 在最小值和最大值之间插入等间距断点
                    min_val, max_val = np.min(data), np.max(data)
                    breaks = list(np.linspace(min_val, max_val, num_classes + 1))
                return breaks

        # 使用分位数计算断点
        breaks = []
        for i in range(num_classes + 1):
            quantile = i / num_classes
            if quantile == 1.0:
                idx = len(sorted_data) - 1
            else:
                idx = int(quantile * len(sorted_data))

            if idx < len(sorted_data):
                breaks.append(sorted_data[idx])

        # 确保断点是唯一的
        unique_breaks = []
        for break_val in breaks:
            if break_val not in unique_breaks:
                unique_breaks.append(break_val)

        # 如果断点数量不足，使用等间距断点
        if len(unique_breaks) < num_classes + 1:
            min_val, max_val = np.min(data), np.max(data)
            if min_val == max_val:
                # 所有值相同，创建微小差异的断点
                unique_breaks = list(np.linspace(min_val - 0.01, max_val + 0.01, num_classes + 1))
            else:
                unique_breaks = list(np.linspace(min_val, max_val, num_classes + 1))

        return unique_breaks