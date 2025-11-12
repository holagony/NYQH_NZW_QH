from typing import Dict, List, Any, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class IndicatorCalculator:
    """指标计算器，用于从基础气象数据中计算各种指标 - 增强版支持多种频率"""
    
    def __init__(self):
        self._functions = self._register_functions()

    def calculate_batch(self, data_dict: Dict[str, pd.DataFrame], 
                       indicator_configs: Dict[str, Any]) -> Dict[str, Dict[str, Union[float, Dict, pd.Series]]]:
        """批量计算多个站点的多个指标 - 支持多种频率输出"""
        results = {}
        
        for station_id, data in data_dict.items():
            station_results = {}
            for indicator_name, indicator_config in indicator_configs.items():
                try:
                    value = self.calculate(data, indicator_config)
                    station_results[indicator_name] = value
                except Exception as e:
                    print(f"站点 {station_id} 指标 {indicator_name} 计算失败: {str(e)}")
                    station_results[indicator_name] = np.nan
            
            results[station_id] = station_results
        
        return results
    
    def calculate(self, data: pd.DataFrame, indicator_config: Dict[str, Any]) -> Any:
        """根据配置计算指标 - 支持频率参数"""
        indicator_type = indicator_config.get("type")
        frequency = indicator_config.get("frequency", "lta")  # 默认为多年平均
        
        if indicator_type not in self._functions:
            raise ValueError(f"不支持的指标类型: {indicator_type}")
        
        return self._functions[indicator_type](data, indicator_config, frequency)

    def _register_functions(self) -> Dict[str, Callable]:
        """注册所有指标计算函数"""
        return {
            "daily_value": self._calculate_daily_value,
            "period_mean": self._calculate_period_mean,
            "period_sum": self._calculate_period_sum,
            "period_extreme": self._calculate_period_extreme,
            "conditional_count": self._calculate_conditional_count,
            "conditional_sum": self._calculate_conditional_sum,
            "growing_degree_days": self._calculate_growing_degree_days,
            "custom_formula": self._calculate_custom_formula,
            "standardize": self._calculate_standardize,
            "bean_moth_X1": self._calculate_bean_moth_X1,
            "bean_moth_X2": self._calculate_bean_moth_X2,
            "bean_moth_X3": self._calculate_bean_moth_X3,
            "bean_moth_X4": self._calculate_bean_moth_X4
        }
        
    def _calculate_daily_value(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """计算某一天的某个气象要素值 - 支持多种频率"""
        date_str = config["date"]
        variable = config["variable"]
        
        # 解析日期格式
        if len(date_str) == 5:  # MM-DD 格式
            # 对于逐年数据，需要处理每年的该日期
            if frequency == "yearly":
                yearly_values = {}
                years = data.index.year.unique()
                
                for year in years:
                    full_date = f"{year}-{date_str}"
                    target_date = pd.to_datetime(full_date)
                    if target_date in data.index:
                        yearly_values[year] = data.loc[target_date, variable]
                    else:
                        yearly_values[year] = np.nan
                
                return yearly_values
            else:
                # 对于多年平均，计算所有年份的平均值
                yearly_values = []
                years = data.index.year.unique()
                
                for year in years:
                    full_date = f"{year}-{date_str}"
                    target_date = pd.to_datetime(full_date)
                    if target_date in data.index:
                        yearly_values.append(data.loc[target_date, variable])
                
                if yearly_values:
                    return np.mean(yearly_values)
                else:
                    return np.nan
        else:
            # 完整日期格式 YYYY-MM-DD
            target_date = pd.to_datetime(date_str)
            if target_date in data.index:
                return data.loc[target_date, variable]
            else:
                return np.nan
        
    def _calculate_period_mean(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的时间段平均值计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        variable = config["variable"]
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码的向量化方法
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            """为整个时间序列创建年份掩码"""
            # 获取所有年份
            years = data_index.year.unique()
            
            # 为每个年份创建时间段掩码
            all_masks = []
            year_info = []
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            # 合并所有年份的掩码
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        # 创建时间段掩码
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码一次性计算所有年份的平均值
        period_data = data.loc[period_mask, variable]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 按年份分组计算每年平均值
        yearly_means = period_data.groupby(period_data.index.year).mean()
        
        if frequency == "yearly":
            # 返回逐年数据
            return yearly_means.to_dict()
        else:
            # 返回多年平均
            return float(yearly_means.mean()) if not yearly_means.empty else np.nan
    
    def _calculate_period_sum(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的时间段累计值计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        variable = config["variable"]
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            years = data_index.year.unique()
            all_masks = []
            year_info = []
            
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码一次性计算所有年份的累计值
        period_data = data.loc[period_mask, variable]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 按年份分组计算每年累计值
        yearly_sums = period_data.groupby(period_data.index.year).sum()
        
        if frequency == "yearly":
            # 返回逐年数据
            return yearly_sums.to_dict()
        else:
            # 返回多年平均
            return float(yearly_sums.mean()) if not yearly_sums.empty else np.nan
    
    def _calculate_period_extreme(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的时间段极值计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        variable = config["variable"]
        extreme_type = config.get("extreme_type", "max")
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            years = data_index.year.unique()
            all_masks = []
            year_info = []
            
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码一次性计算所有年份的极值
        period_data = data.loc[period_mask, variable]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 按年份分组计算每年极值
        if extreme_type == "max":
            yearly_extremes = period_data.groupby(period_data.index.year).max()
        else:
            yearly_extremes = period_data.groupby(period_data.index.year).min()
        
        if frequency == "yearly":
            # 返回逐年数据
            return yearly_extremes.to_dict()
        else:
            # 返回多年平均
            return float(yearly_extremes.mean()) if not yearly_extremes.empty else np.nan
    
    def _calculate_conditional_sum(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的条件累计计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        conditions = config["conditions"]
        variable = config["variable"]
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            years = data_index.year.unique()
            all_masks = []
            year_info = []
            
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码获取时间段数据
        period_data = data.loc[period_mask]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 构建条件表达式并应用
        condition_expr = self._build_condition_expression(conditions)
        
        if condition_expr:
            try:
                # 使用query方法进行条件筛选
                filtered_data = period_data.query(condition_expr)
                
                if filtered_data.empty:
                    return np.nan if frequency == "lta" else {}
                
                # 按年份分组计算每年条件累计值
                yearly_sums = filtered_data[variable].groupby(filtered_data.index.year).sum()
                
                if frequency == "yearly":
                    # 返回逐年数据
                    return yearly_sums.to_dict()
                else:
                    # 返回多年平均
                    return float(yearly_sums.mean()) if not yearly_sums.empty else np.nan
            except Exception as e:
                print(f"条件累计计算错误: {str(e)}")
                return np.nan if frequency == "lta" else {}
        else:
            return np.nan if frequency == "lta" else {}
    
    def _calculate_conditional_count(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的条件计数计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        conditions = config["conditions"]
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            years = data_index.year.unique()
            all_masks = []
            year_info = []
            
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码获取时间段数据
        period_data = data.loc[period_mask]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 构建条件表达式并应用
        condition_expr = self._build_condition_expression(conditions)
        
        if condition_expr:
            try:
                # 使用eval方法计算条件满足的天数
                condition_result = period_data.eval(condition_expr)
                
                # 按年份分组计算每年满足条件的天数
                yearly_counts = condition_result.groupby(period_data.index.year).sum()
                
                if frequency == "yearly":
                    # 返回逐年数据
                    return yearly_counts.to_dict()
                else:
                    # 返回多年平均
                    return float(yearly_counts.mean()) if not yearly_counts.empty else 0
            except Exception as e:
                print(f"条件计数计算错误: {str(e)}")
                return 0 if frequency == "lta" else {}
        else:
            return 0 if frequency == "lta" else {}
    
    def _calculate_growing_degree_days(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的活动积温计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        base_temp = config.get("base_temp", 10)
        method = config.get("method", "mean")
        year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 创建年份掩码
        def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
            years = data_index.year.unique()
            all_masks = []
            year_info = []
            
            for year in years:
                start_date = pd.to_datetime(f"{year}-{start_date_str}")
                end_year = year + year_offset
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)
                year_info.append(year)
            
            if all_masks:
                combined_mask = all_masks[0]
                for mask in all_masks[1:]:
                    combined_mask = combined_mask | mask
                return combined_mask, year_info
            else:
                return pd.Series(False, index=data_index), []
        
        period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        if not period_mask.any():
            return np.nan if frequency == "lta" else {}
        
        # 使用掩码获取时间段数据
        period_data = data.loc[period_mask]
        
        if period_data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 计算活动积温 - 使用向量化操作
        if method == "mean":
            # 使用日平均温度计算
            daily_gdd = np.maximum(period_data["tavg"] - base_temp, 0)
        elif method == "min_max":
            # 使用日最高最低温度计算
            daily_gdd = np.maximum((period_data["tmax"] + period_data["tmin"]) / 2 - base_temp, 0)
        else:
            raise ValueError(f"不支持的积温计算方法: {method}")
        
        # 按年份分组计算每年积温
        yearly_gdd = daily_gdd.groupby(period_data.index.year).sum()
        
        if frequency == "yearly":
            # 返回逐年数据
            return yearly_gdd.to_dict()
        else:
            # 返回多年平均
            return yearly_gdd.mean() if not yearly_gdd.empty else 0
    
    def _calculate_custom_formula(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Any:
        """使用自定义公式计算指标 - 支持多种频率输出"""
        formula = config["formula"]
        variables = config.get("variables", {})
        
        # 准备变量数据
        var_data = {}
        for var_name, var_config in variables.items():
            # 递归计算子指标，传递频率参数
            if isinstance(var_config, dict) and "type" in var_config:
                var_config_with_freq = var_config.copy()
                var_config_with_freq["frequency"] = frequency
                var_data[var_name] = self.calculate(data, var_config_with_freq)
            elif isinstance(var_config, dict) and "ref" in var_config:
                # 引用其他指标
                var_config_with_freq = var_config.copy()
                var_config_with_freq["frequency"] = frequency
                var_data[var_name] = self.calculate(data, var_config_with_freq)
            else:
                # 直接使用配置中的值
                var_data[var_name] = var_config.get("value", 0) if isinstance(var_config, dict) else var_config
        
        # 替换公式中的变量名
        for var_name, var_value in var_data.items():
            formula = formula.replace(var_name, str(var_value))
        
        # 计算表达式
        try:
            result = eval(formula)
            
            # 根据频率处理结果
            if frequency == "yearly" and isinstance(result, dict):
                return result
            elif frequency == "lta" and isinstance(result, (int, float)):
                return result
            else:
                return result
                
        except Exception as e:
            raise ValueError(f"公式计算错误: {formula}, 错误: {str(e)}")
    
    # 其他辅助函数保持不变
    def _build_condition_expression(self, conditions: List[Dict[str, Any]]) -> str:
        """构建条件表达式"""
        expr_parts = []
        
        for condition in conditions:
            variable = condition["variable"]
            operator = condition["operator"]
            value = condition["value"]
            
            if operator == "between":
                if isinstance(value, list) and len(value) == 2:
                    expr_parts.append(f"({variable} >= {value[0]}) & ({variable} <= {value[1]})")
                else:
                    raise ValueError("between操作符需要两个值的列表")
            elif operator == "in":
                if isinstance(value, list):
                    value_str = ", ".join(map(str, value))
                    expr_parts.append(f"{variable} in [{value_str}]")
                else:
                    raise ValueError("in操作符需要值的列表")
            else:
                expr_parts.append(f"{variable} {operator} {value}")
        
        return " & ".join(expr_parts) if expr_parts else ""
    
    # 大豆食心虫相关函数也需要支持频率参数
    def _calculate_bean_moth_X1(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """根据大豆食心虫赋值表计算X1 - 支持多种频率输出"""
        value = config["value"]
        if isinstance(value, dict) and "ref" in value:
            # 传递频率参数
            value_with_freq = value.copy()
            value_with_freq["frequency"] = frequency
            T1 = self.calculate(data, value_with_freq)
            
            # 根据赋值表计算X1
            if frequency == "yearly" and isinstance(T1, dict):
                # 处理逐年数据
                yearly_X1 = {}
                for year, t1_value in T1.items():
                    if np.isnan(t1_value):
                        yearly_X1[year] = np.nan
                    elif t1_value < 15:
                        yearly_X1[year] = 1
                    elif 15 <= t1_value < 18:
                        yearly_X1[year] = 2
                    else:
                        yearly_X1[year] = 3
                return yearly_X1
            else:
                # 处理标量数据
                if np.isnan(T1):
                    return np.nan
                elif T1 < 15:
                    return 1
                elif 15 <= T1 < 18:
                    return 2
                else:
                    return 3
        else:
            return value
    
    def _calculate_bean_moth_X2(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """根据大豆食心虫赋值表计算X2 - 支持多种频率输出"""
        value = config["value"]
        if isinstance(value, dict) and "ref" in value:
            # 传递频率参数
            value_with_freq = value.copy()
            value_with_freq["frequency"] = frequency
            R1 = self.calculate(data, value_with_freq)
            
            # 根据赋值表计算X2
            if frequency == "yearly" and isinstance(R1, dict):
                # 处理逐年数据
                yearly_X2 = {}
                for year, r1_value in R1.items():
                    if np.isnan(r1_value):
                        yearly_X2[year] = np.nan
                    elif r1_value < 10:
                        yearly_X2[year] = 1
                    elif 10 <= r1_value < 20:
                        yearly_X2[year] = 2
                    else:
                        yearly_X2[year] = 3
                return yearly_X2
            else:
                # 处理标量数据
                if np.isnan(R1):
                    return np.nan
                elif R1 < 10:
                    return 1
                elif 10 <= R1 < 20:
                    return 2
                else:
                    return 3
        else:
            return value
    
    def _calculate_bean_moth_X3(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """根据大豆食心虫赋值表计算X3 - 支持多种频率输出"""
        value = config["value"]
        if isinstance(value, dict) and "ref" in value:
            # 传递频率参数
            value_with_freq = value.copy()
            value_with_freq["frequency"] = frequency
            D1 = self.calculate(data, value_with_freq)
            
            # 根据赋值表计算X3
            if frequency == "yearly" and isinstance(D1, dict):
                # 处理逐年数据
                yearly_X3 = {}
                for year, d1_value in D1.items():
                    if np.isnan(d1_value):
                        yearly_X3[year] = np.nan
                    elif d1_value < 5:
                        yearly_X3[year] = 1
                    elif 5 <= d1_value < 10:
                        yearly_X3[year] = 2
                    else:
                        yearly_X3[year] = 3
                return yearly_X3
            else:
                # 处理标量数据
                if np.isnan(D1):
                    return np.nan
                elif D1 < 5:
                    return 1
                elif 5 <= D1 < 10:
                    return 2
                else:
                    return 3
        else:
            return value
    
    def _calculate_bean_moth_X4(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """根据大豆食心虫赋值表计算X4 - 支持多种频率输出"""
        value = config["value"]
        if isinstance(value, dict) and "type" in value:
            # 递归计算子指标，传递频率参数
            value_with_freq = value.copy()
            value_with_freq["frequency"] = frequency
            return self.calculate(data, value_with_freq)
        else:
            return value
    
    def _calculate_standardize(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """标准化处理 - 支持多种频率输出"""
        value = config["value"]
        if isinstance(value, dict) and "ref" in value:
            # 传递频率参数
            value_with_freq = value.copy()
            value_with_freq["frequency"] = frequency
            ref_value = self.calculate(data, value_with_freq)
            
            # 标准化处理
            if frequency == "yearly" and isinstance(ref_value, dict):
                # 对逐年数据进行标准化
                values = list(ref_value.values())
                valid_values = [v for v in values if not np.isnan(v)]
                
                if len(valid_values) > 0:
                    min_val = min(valid_values)
                    max_val = max(valid_values)
                    
                    if max_val == min_val:
                        # 所有值相同，标准化为0.5
                        standardized = {year: 0.5 for year in ref_value.keys()}
                    else:
                        standardized = {}
                        for year, val in ref_value.items():
                            if np.isnan(val):
                                standardized[year] = np.nan
                            else:
                                standardized[year] = (val - min_val) / (max_val - min_val)
                    return standardized
                else:
                    return {year: np.nan for year in ref_value.keys()}
            else:
                # 对标量数据进行标准化（需要所有站点的值，这里暂时返回原值）
                return ref_value
        else:
            # 直接使用给定的值
            return value