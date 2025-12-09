from typing import Dict, List, Any, Callable, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class IndicatorCalculator:
    """指标计算器，用于从基础气象数据中计算各种指标 - 增强版支持多种频率"""
    
    def __init__(self):
        self._functions = self._register_functions()

    def calculate_batch(self, data_dict: Dict[str, pd.DataFrame], 
                       indicator_configs: Dict[str, Any]) -> Dict[str, Dict[str, Union[float, Dict, pd.DataFrame]]]:
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
            "total_radiation": self._calculate_total_radiation,
            "stable_threshold_accumulation": self._calculate_stable_threshold_accumulation,
            "stable_threshold_extreme": self._calculate_stable_threshold_extreme,
            "fao_pet": self._calculate_fao_pet,
            "aridity_index": self._calculate_aridity_index,
            "daily_sequence": self._calculate_daily_sequence,  # 新增：逐日序列计算
            "daily_range_mean": self._calculate_daily_range_mean,  # 新增：TDI 平均气温日较差 （℃）
        }

    # def _create_period_mask(self, data_index: pd.DatetimeIndex, start_date_str: str, end_date_str: str) -> pd.Series:
    #     """创建时间段掩码 - 向量化版本"""
    #     years = data_index.year.unique()
    #     all_masks = []
    #
    #     for year in years:
    #         try:
    #             start_date = pd.to_datetime(f"{year}-{start_date_str}")
    #             end_date = pd.to_datetime(f"{year}-{end_date_str}")
    #
    #             # 检查跨年情况
    #             if end_date < start_date:
    #                 end_date = pd.to_datetime(f"{year+1}-{end_date_str}")
    #
    #             year_mask = (data_index >= start_date) & (data_index <= end_date)
    #             all_masks.append(year_mask)
    #         except Exception as e:
    #             print(f"警告: 无法解析 {year} 年的日期范围: {start_date_str} - {end_date_str}, 错误: {e}")
    #             continue
    #
    #     if all_masks:
    #         combined_mask = all_masks[0]
    #         for mask in all_masks[1:]:
    #             combined_mask = combined_mask | mask
    #         return combined_mask
    #     else:
    #         return pd.Series(False, index=data_index)

    def _create_period_mask(self, data_index: pd.DatetimeIndex, start_date_str: str, end_date_str: str,
                            year_offset: int = 0) -> pd.Series:
        """
        创建时间段掩码 - 支持年份偏移的向量化版本

        Parameters:
        -----------
        data_index : pd.DatetimeIndex
            时间索引
        start_date_str : str
            开始日期字符串，格式 "MM-DD"
        end_date_str : str
            结束日期字符串，格式 "MM-DD"
        year_offset : int, default=0
            年份偏移量
            0: 当年 (默认)
            -1: 去年
            1: 明年
            -2: 前年
            2: 后年

        Returns:
        --------
        pd.Series
            布尔掩码Series
        """
        years = data_index.year.unique()
        all_masks = []

        for base_year in years:
            try:
                # 应用年份偏移
                start_year = base_year + year_offset
                end_year = base_year + year_offset

                # 构建完整的日期
                start_date = pd.to_datetime(f"{start_year}-{start_date_str}")
                end_date = pd.to_datetime(f"{end_year}-{end_date_str}")

                # 检查跨年情况（保持原有逻辑）
                if end_date < start_date:
                    end_date = pd.to_datetime(f"{end_year + 1}-{end_date_str}")

                # 创建当前基础年份的掩码
                year_mask = (data_index >= start_date) & (data_index <= end_date)
                all_masks.append(year_mask)

            except Exception as e:
                print(
                    f"警告: 无法解析基础年份 {base_year} 的日期范围: {start_year}-{start_date_str} - {end_year}-{end_date_str}, 错误: {e}")
                continue

        if all_masks:
            combined_mask = all_masks[0]
            for mask in all_masks[1:]:
                combined_mask = combined_mask | mask
            return combined_mask
        else:
            return pd.Series(False, index=data_index)

    def _calculate_daily_sequence(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[pd.DataFrame, float, Dict]:
        """计算逐日序列指标 - 返回包含每日数据的DataFrame"""
        variable = config["variable"]
        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        
        if data.empty:
            if frequency == "daily":
                return pd.DataFrame()
            else:
                return np.nan if frequency == "lta" else {}
        
        # # 提取指定时间段的数据
        # if start_date and end_date:
        #     # 创建年份掩码
        #     def create_year_mask(data_index, start_date_str, end_date_str):
        #         years = data_index.year.unique()
        #         all_masks = []
                
        #         for year in years:
        #             start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #             end_date = pd.to_datetime(f"{year}-{end_date_str}")
                    
        #             year_mask = (data_index >= start_date) & (data_index <= end_date)
        #             all_masks.append(year_mask)
                
        #         if all_masks:
        #             combined_mask = all_masks[0]
        #             for mask in all_masks[1:]:
        #                 combined_mask = combined_mask | mask
        #             return combined_mask
        #         else:
        #             return pd.Series(False, index=data_index)
            
        #     period_mask = create_year_mask(data.index, start_date, end_date)
        #     period_data = data.loc[period_mask, variable]
        # else:
        #     period_data = data[variable]

        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            period_data = data.loc[period_mask,variable]
            if data.empty:
                if frequency == "daily":
                    return pd.DataFrame()
                else:
                    return np.nan if frequency == "lta" else {}
        
        if frequency == "daily":
            # 返回逐日数据的DataFrame
            daily_df = period_data.reset_index()
            daily_df.columns = ['datetime', variable]
            daily_df['datetime'] = daily_df['datetime'].dt.strftime('%Y%m%d')
            return daily_df
        elif frequency == "yearly":
            # 按年份分组计算统计值
            yearly_stats = period_data.groupby(period_data.index.year).apply(
                lambda x: {'mean': x.mean(), 'sum': x.sum(), 'max': x.max(), 'min': x.min()}
            )
            return yearly_stats.to_dict()
        else:
            # 返回多年平均值
            return float(period_data.mean()) if not period_data.empty else np.nan

    def _calculate_stable_threshold_accumulation(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """计算稳定通过温度阈值的有效积温或降水累计 - 支持多种频率输出"""
        variable = config["variable"]
        base_temp = config.get("base_temp", None)  # 温度阈值，None表示不进行温度筛选
        # frequency = config.get("frequency", "stn")  # stn: 多年平均, yearly: 逐年
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 获取温度数据
        if base_temp is not None:
            tem_data = data["tavg"]  # 使用日平均温度
        else:
            tem_data = None
        
        # 获取要累计的变量数据
        target_data = data[variable]
        
        # 计算稳定通过阈值的开始和结束时间
        first_days, last_days, day_lengths = self._stable_through_temp_threshold(tem_data, base_temp)
        
        # 按年份分组
        data_group_y = target_data.groupby(target_data.index.year)
        
        # 处理有效年份
        years = list(data_group_y.groups.keys())
        valid_indices = [i for i, (first, last) in enumerate(zip(first_days, last_days)) 
                        if not np.isnan(first) and not np.isnan(last)]
        
        if not valid_indices:
            return np.nan if frequency == "lta" else {}
        
        # 提取有效年份的数据
        valid_years = [years[i] for i in valid_indices]
        valid_first_days = [first_days[i] for i in valid_indices]
        valid_last_days = [last_days[i] for i in valid_indices]
        
        # 计算每年的累计值
        yearly_accumulations = {}
        for i, year in enumerate(valid_years):
            year_data = data_group_y.get_group(year).values
            first_day = int(valid_first_days[i])
            last_day = int(valid_last_days[i])
            
            # 确保索引在有效范围内
            if first_day < len(year_data) and last_day < len(year_data) and first_day <= last_day:
                period_data = year_data[first_day:last_day+1]
                # 对于温度数据，减去阈值；对于降水等数据，直接累计
                # if variable in ["tavg", "tmax", "tmin"] and base_temp is not None:
                #     period_accumulation = np.nansum(np.maximum(period_data - base_temp, 0))
                # else:
                period_accumulation = np.nansum(period_data)
                yearly_accumulations[year] = period_accumulation
        
        if frequency == "yearly":
            return yearly_accumulations
        else:
            # 返回多年平均
            values = list(yearly_accumulations.values())
            return np.nanmean(values) if values else np.nan
    
    def _calculate_stable_threshold_extreme(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """计算稳定通过温度阈值期间的最大日降水量 - 支持多种频率输出"""
        variable = config["variable"]
        base_temp = config.get("base_temp", None)  # 温度阈值
        ref_value = config.get("ref_value", 0)  # 数值阈值（如最小降水量）
        extreme_type = config.get("extreme_type", "max")  # max或min
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # 获取温度数据
        if base_temp is not None:
            tem_data = data["tavg"]
        else:
            tem_data = None
        
        # 获取目标变量数据
        target_data = data[variable]
        
        # 计算稳定通过阈值的开始和结束时间
        first_days, last_days, day_lengths = self._stable_through_temp_threshold(tem_data, base_temp)
        
        # 按年份分组
        data_group_y = target_data.groupby(target_data.index.year)
        
        # 处理有效年份
        years = list(data_group_y.groups.keys())
        valid_indices = [i for i, (first, last) in enumerate(zip(first_days, last_days)) 
                        if not np.isnan(first) and not np.isnan(last)]
        
        if not valid_indices:
            return np.nan if frequency == "lta" else {}
        
        # 提取有效年份的数据
        valid_years = [years[i] for i in valid_indices]
        valid_first_days = [first_days[i] for i in valid_indices]
        valid_last_days = [last_days[i] for i in valid_indices]
        
        # 计算每年的极值
        yearly_extremes = {}
        for i, year in enumerate(valid_years):
            year_data = data_group_y.get_group(year).values
            first_day = int(valid_first_days[i])
            last_day = int(valid_last_days[i])
            
            # 确保索引在有效范围内
            if first_day < len(year_data) and last_day < len(year_data) and first_day <= last_day:
                period_data = year_data[first_day:last_day+1]
                
                # 应用数值阈值筛选
                valid_period_data = period_data[period_data > ref_value]
                
                if len(valid_period_data) > 0:
                    if extreme_type == "max":
                        yearly_extremes[year] = np.nanmax(valid_period_data)
                    else:
                        yearly_extremes[year] = np.nanmin(valid_period_data)
                else:
                    yearly_extremes[year] = np.nan
        
        if frequency == "yearly":
            return yearly_extremes
        else:
            # 返回多年平均
            values = [v for v in yearly_extremes.values() if not np.isnan(v)]
            return np.nanmean(values) if values else np.nan
    
    def _stable_through_temp_threshold(self, tem_data: pd.Series, ref_tem: float) -> tuple:
        """计算温度稳定通过某个温度阈值的开始时间、结束时间、持续时间"""
        first_days = []
        last_days = []
        day_lengths = []
        
        if ref_tem is not None:
            # 五日滑动平均
            tem_5day = tem_data.rolling(window=5, min_periods=3).mean()
            data_cut = tem_5day
        else:
            # 不进行五日滑动平均
            data_cut = tem_data
        
        # 按年份分组
        tem_group_y = data_cut.groupby(data_cut.index.year)
        
        for year, year_data in tem_group_y:
            if year_data.isna().all():
                first_days.append(np.nan)
                last_days.append(np.nan)
                day_lengths.append(np.nan)
                continue
            
            if ref_tem is not None:
                # 分为前后半年处理
                year_start = pd.Timestamp(f"{year}-01-01")
                mid_year = pd.Timestamp(f"{year}-07-01")
                year_end = pd.Timestamp(f"{year}-12-31")
                
                # 前半年低于阈值的数据
                valid_temps1 = year_data[(year_data.index < mid_year) & (year_data < ref_tem)]
                # 后半年低于阈值的数据
                valid_temps2 = year_data[(year_data.index >= mid_year) & (year_data < ref_tem)]
                
                if valid_temps1.isna().all():
                    first_valid_index = year_start
                    first_day = 0
                else:
                    first_valid_index = valid_temps1.last_valid_index() + pd.Timedelta(days=1)
                    first_day = (first_valid_index - year_start).days
                
                if valid_temps2.isna().all():
                    last_valid_index = year_end
                    last_day = (year_end - year_start).days
                else:
                    last_valid_index = valid_temps2.first_valid_index() - pd.Timedelta(days=1)
                    last_day = (last_valid_index - year_start).days
                
                day_length = (last_valid_index - first_valid_index).days + 1
                
            else:
                # 不使用温度阈值，直接使用有效数据范围
                year_start = pd.Timestamp(f"{year}-01-01")
                first_valid_index = year_data.first_valid_index()
                last_valid_index = year_data.last_valid_index()
                
                if pd.isna(first_valid_index) or pd.isna(last_valid_index):
                    first_days.append(np.nan)
                    last_days.append(np.nan)
                    day_lengths.append(np.nan)
                    continue
                
                first_day = (first_valid_index - year_start).days
                last_day = (last_valid_index - year_start).days
                day_length = (last_valid_index - first_valid_index).days + 1
            
            first_days.append(first_day)
            last_days.append(last_day)
            day_lengths.append(day_length)
        
        return first_days, last_days, day_lengths

    def _calculate_fao_pet(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[pd.DataFrame, float, Dict[int, float]]:
        """FAO-Penman-Monteith方法计算潜在蒸散量"""
        # 获取配置参数

        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        
        if data.empty:
            if frequency == "daily":
                return pd.DataFrame()
            else:
                return np.nan if frequency == "lta" else {}
        
        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            data = data.loc[period_mask]
            if data.empty:
                if frequency == "daily":
                    return pd.DataFrame()
                else:
                    return np.nan if frequency == "lta" else {}
        
        # # 确保必要的列存在
        # for col, default_val in [('rhum', 70), ('ssh', 6), ('wind', 2)]:
        #     if col not in data.columns:
        #         data[col] = default_val
        
        # 向量化计算PET
        dates = data.index
        lat = data["lat"].values
        alti = data["altitude"].values
        Tmean = data["tavg"].values
        Tmax = data["tmax"].values
        Tmin = data["tmin"].values
        rh = data["rhum"].values
        ssh = data["sunshine"].values
        u2 = data["wind"].values
        
        # 直接使用向量化计算，避免循环
        pet_values = self._fao_pet_vectorized(dates, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2)
        
        # 创建PET序列
        pet_series = pd.Series(pet_values, index=data.index)
        
        # 根据频率返回结果
        if frequency == "daily":
            daily_df = pd.DataFrame({
                'datetime': data.index.strftime('%Y%m%d'),
                'ET0': pet_series
            })
            return daily_df
        else:
            yearly_pet = pet_series.groupby(pet_series.index.year).sum()
            
            if frequency == "yearly":
                return yearly_pet.to_dict()
            else:
                return float(yearly_pet.mean()) if not yearly_pet.empty else np.nan

    def _fao_pet_vectorized(self, dates, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2):
        """FAO-PET计算"""
        # 风速转换
        u2m = u2 * 4.87 / np.log(67.8 * 10 - 5.42)
        
        # 计算饱和水汽压es，kPa
        def e0(T):
            T = np.asarray(T, dtype=np.float64) 
            e = 0.6108 * (np.exp(17.27 * T / (T + 237.3)))
            return e
        
        es = (e0(Tmax) + e0(Tmin)) / 2

        # 计算实际水汽压ea，kPa
        ea = es * rh / 100

        # 水汽压-温度曲线的斜率
        delta = 4098 * e0(Tmean) / (Tmean + 237.3) ** 2

        # 土壤热通量
        G = 0
        
        # 大气压，kPa
        P = 101.3 * ((293 - 0.0065 * alti) / 293) ** 5.26
        gamma = 0.665e-3 * P
    
        # 计算地表净辐射
        Rs, par, Ra = self._calculate_radiation_vectorized(dates, lat, ssh)
        
        # 计算太阳（短波）净辐射
        alpha = 0.23
        Rns = (1 - alpha) * Rs
        
        # 计算晴空地表太阳辐射
        Rs0 = (0.75 + 0.00002 * alti) * Ra
        
        # 计算地表净长波净辐射
        Tmax_k = 273.15 + Tmax
        Tmin_k = 273.15 + Tmin
        ea_ = np.asarray(ea, dtype=np.float64) 
        Rnl = 4.903e-9 * 0.5 * (Tmax_k ** 4 + Tmin_k ** 4) * (0.34 - 0.14 * np.sqrt(ea_)) * (1.35 * Rs / Rs0 - 0.35)
        
        # 计算地表净辐射
        Rn = Rns - Rnl
        
        # 计算潜在蒸散量，mm/day
        PET_1 = 0.408 * delta * (Rn - G)
        PET_2 = gamma * 900 * u2m * (es - ea) / (Tmean + 273)
        PET_3 = delta + gamma * (1 + 0.34 * u2m)
        PET = (PET_1 + PET_2) / PET_3
        
        return np.maximum(PET, 0)  # 确保PET不为负
   
    def _calculate_aridity_index(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[pd.DataFrame, float, Dict[int, float]]:
        """计算干燥度指数"""
        lat = config.get("lat")
        alti = config.get("alti")
        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        
        if data.empty:
            if frequency == "daily":
                return pd.DataFrame()
            else:
                return np.nan if frequency == "lta" else {}
        
        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            data = data.loc[period_mask]
            if data.empty:
                if frequency == "daily":
                    return pd.DataFrame()
                else:
                    return np.nan if frequency == "lta" else {}
        
        # # 确保必要的列存在
        # for col, default_val in [('rhum', 70), ('ssh', 6), ('wind', 2)]:
        #     if col not in data.columns:
        #         data[col] = default_val
        
        # 向量化计算PET
        dates = data.index
        Tmean = data["tavg"].values
        Tmax = data["tmax"].values
        Tmin = data["tmin"].values
        rh = data["rhum"].values
        ssh = data["sunshine"].values
        u2 = data["wind"].values
        
        pet_values = self._fao_pet_vectorized(dates, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2)
        pet_series = pd.Series(pet_values, index=data.index)
        pre_series = data["precip"]
        
        # 向量化计算干燥度指数
        # 避免除零错误
        adi_values = np.where(pre_series > 0, pet_series / pre_series, np.nan)
        adi_series = pd.Series(adi_values, index=data.index)
        
        # 根据频率返回结果
        if frequency == "daily":
            daily_df = pd.DataFrame({
                'datetime': data.index.strftime('%Y%m%d'),
                'ADI': adi_series
            })
            return daily_df
        else:
            # 多年平均
            valid_adi = adi_series[~np.isnan(adi_series)]
            return float(valid_adi.mean()) if len(valid_adi) > 0 else np.nan   

    def _calculate_total_radiation(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[pd.DataFrame, float, Dict[int, float]]:
        """计算太阳总辐射"""
        start_date_str = config.get("start_date")
        end_date_str = config.get("end_date")
        
        if data.empty:
            if frequency == "daily":
                return pd.DataFrame()
            else:
                return np.nan if frequency == "lta" else {}
        
        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            data = data.loc[period_mask]
            if data.empty:
                if frequency == "daily":
                    return pd.DataFrame()
                else:
                    return np.nan if frequency == "lta" else {}
        
        # # 确保必要的列存在
        # if 'ssh' not in data.columns:
        #     data['ssh'] = 6
        
        # 获取纬度信息
        # if 'lat' in data.columns:
        lat_values = data['lat'].values
        # else:
        # fixed_latitude = 40.0
        # lat_values = np.full(len(data), fixed_latitude)
        
        # 向量化计算辐射
        dates = data.index
        ssh_values = data["sunshine"].values
        
        Rs_values, par_values, Ra_values = self._calculate_radiation_vectorized(dates, lat_values, ssh_values)
        Rs_series = pd.Series(Rs_values, index=data.index)
        
        # 根据频率返回结果
        if frequency == "daily":
            daily_df = pd.DataFrame({
                'datetime': data.index.strftime('%Y%m%d'),
                'Rs': Rs_series
            })
            return daily_df
        else:
            # 多年平均
            yearly_radiation = Rs_series.groupby(Rs_series.index.year).sum()
            return float(yearly_radiation.mean()) if not yearly_radiation.empty else 0

    def _calculate_radiation_vectorized(self, dates, lat, ssh):
        """向量化的辐射计算"""
        # 转换为numpy数组确保向量化操作
        lat = np.asarray(lat)
        ssh = np.asarray(ssh)
        
        
        a = np.nanmax(ssh)
        b = np.nanmin(ssh)
        if (a>24) | (b<0) :
            print(np.nanmax(ssh))
            print(np.nanmin(ssh))
        # 如果dates是DatetimeIndex，直接获取dayofyear
        if hasattr(dates, 'dayofyear'):
            day_of_year = dates.dayofyear.values
        else:
            # 如果是字符串格式的日期，需要转换
            day_of_year = pd.to_datetime(dates).dayofyear.values
        
        lat_rad = np.radians(lat)
        
        # 地球轨道偏心率订正系数
        dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
        # 太阳赤纬
        sigma = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
        # 日落时角
        cos_ws = -np.tan(lat_rad) * np.tan(sigma)
        cos_ws = np.clip(cos_ws, -1.0, 1.0)
        ws = np.arccos(cos_ws)
        
        # 晴天状态下日太阳总辐射
        Ra = (24 * 60 / np.pi) * 0.082 * dr * (
            ws * np.sin(lat_rad) * np.sin(sigma) + np.cos(lat_rad) * np.cos(sigma) * np.sin(ws)
        )
        Ra = np.maximum(Ra, 0)
        
        # 可能日照时数
        N = (24 * ws / np.pi)
        
        # 地表日太阳总辐射
        a = 0.25
        b = 0.5
        # 避免除零错误
        N_safe = np.where(N > 0, N, 1)
        Rs = (a + b * ssh / N_safe) * Ra
        
        # 光合有效辐射
        par = 0.5 * Rs
        
        return Rs, par, Ra
    
    # def _calculate_fao_pet(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
    #     """FAO-Penman-Monteith方法计算潜在蒸散量"""
    #     # 获取配置参数
    #     lat = config.get("lat")  # 纬度
    #     alti = config.get("alti", 0)  # 海拔，默认为0
        
    #     if data.empty:
    #         return np.nan if frequency == "lta" else {}
        
    #     # 计算每日的PET
    #     pet_values = []
    #     dates = data.index
        
    #     for i, date in enumerate(dates):
    #         Tmean = data.iloc[i]["tavg"]
    #         Tmax = data.iloc[i]["tmax"]
    #         Tmin = data.iloc[i]["tmin"]
    #         rh = data.iloc[i].get("rhum", 70)  # 相对湿度，默认70%
    #         ssh = data.iloc[i].get("ssh", 6)   # 日照时数，默认6小时
    #         u2 = data.iloc[i].get("wind", 2)   # 风速，默认2m/s
            
    #         pet = self._fao_penman_monteith(date, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2)
    #         pet_values.append(pet)
        
    #     # 创建PET序列
    #     pet_series = pd.Series(pet_values, index=data.index)
        
    #     # 按年份分组计算
    #     yearly_pet = pet_series.groupby(pet_series.index.year).sum()
        
    #     if frequency == "yearly":
    #         return yearly_pet.to_dict()
    #     else:
    #         return float(yearly_pet.mean()) if not yearly_pet.empty else np.nan
    
    # def _fao_penman_monteith(self, datetime_val, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2):
    #     """FAO-Penman-Monteith方法计算日潜在蒸散量"""
    #     # 风速转换
    #     u2m = u2 * 4.87 / np.log(67.8 * 10 - 5.42) if u2 > 0 else 0.5
        
    #     # 计算饱和水汽压es，kPa
    #     def e0(T):
    #         T = np.asarray(T, dtype=np.float64)
    #         e = 0.6108 * (np.exp(17.27 * T / (T + 237.3)))
    #         return e
        
    #     es = (e0(Tmax) + e0(Tmin)) / 2
        
    #     # 计算实际水汽压ea，kPa
    #     ea = es * rh / 100
        
    #     # 水汽压-温度曲线的斜率
    #     delta = 4098 * e0(Tmean) / (Tmean + 237.3) ** 2
        
    #     # 土壤热通量
    #     G = 0
        
    #     # 大气压，kPa
    #     P = 101.3 * ((293 - 0.0065 * alti) / 293) ** 5.26
    #     gamma = 0.665e-3 * P
        
    #     # 计算地表净辐射
    #     Rs, par, Ra = self._calculate_radiation(datetime_val, lat, ssh)
        
    #     # 计算太阳（短波）净辐射
    #     alpha = 0.23
    #     Rns = (1 - alpha) * Rs
        
    #     # 计算晴空地表太阳辐射
    #     Rs0 = (0.75 + 0.00002 * alti) * Ra
        
    #     # 计算地表净长波净辐射
    #     Tmax_k = 273.15 + Tmax
    #     Tmin_k = 273.15 + Tmin
    #     ea_ = np.asarray(ea, dtype=np.float64)
    #     Rnl = 4.903e-9 * 0.5 * (Tmax_k ** 4 + Tmin_k ** 4) * (0.34 - 0.14 * np.sqrt(ea_)) * (1.35 * Rs / Rs0 - 0.35)
        
    #     # 计算地表净辐射
    #     Rn = Rns - Rnl
        
    #     # 计算潜在蒸散量，mm/day
    #     PET_1 = 0.408 * delta * (Rn - G)
    #     PET_2 = gamma * 900 * u2m * (es - ea) / (Tmean + 273)
    #     PET_3 = delta + gamma * (1 + 0.34 * u2m)
    #     PET = (PET_1 + PET_2) / PET_3
        
    #     return max(PET, 0)  # 确保PET不为负
    
    # def _calculate_radiation(self, datetime_val, lat, ssh):
    #     """计算太阳辐射相关参数"""
    #     if isinstance(datetime_val, pd.Timestamp):
    #         day_of_year = datetime_val.dayofyear
    #     else:
    #         day_of_year = pd.to_datetime(datetime_val).dayofyear
        
    #     lat_rad = np.radians(lat)
        
    #     # 地球轨道偏心率订正系数
    #     dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
        
    #     # 太阳赤纬
    #     sigma = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
        
    #     # 日落时角
    #     ws = np.arccos(-np.tan(lat_rad) * np.tan(sigma))
        
    #     # 晴天状态下日太阳总辐射
    #     Ra = (24 * 60 / np.pi) * 0.082 * dr * (
    #         ws * np.sin(lat_rad) * np.sin(sigma) + np.cos(lat_rad) * np.cos(sigma) * np.sin(ws))
    #     Ra = max(Ra, 0)
        
    #     # 可能日照时数
    #     N = (24 * ws / np.pi)
        
    #     # 地表日太阳总辐射
    #     a = 0.25
    #     b = 0.5
    #     Rs = (a + b * ssh / N) * Ra
        
    #     # 光合有效辐射
    #     par = 0.5 * Rs
        
    #     return Rs, par, Ra
    
    # def _calculate_aridity_index(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
    #     """计算干燥度指数 - 潜在蒸散与降水的比值"""
    #     lat = config.get("lat")
    #     alti = config.get("alti", 0)
    #     period = config.get("period", "year")
    #     months = config.get("months", None)
        
    #     if data.empty:
    #         return np.nan if frequency == "lta" else {}
        
    #     # 计算每日PET
    #     pet_daily = []
    #     for i, date in enumerate(data.index):
    #         Tmean = data.iloc[i]["tavg"]
    #         Tmax = data.iloc[i]["tmax"]
    #         Tmin = data.iloc[i]["tmin"]
    #         rh = data.iloc[i].get("rhum", 70)
    #         ssh = data.iloc[i].get("ssh", 6)
    #         u2 = data.iloc[i].get("wind", 2)
            
    #         pet = self._fao_penman_monteith(date, Tmean, Tmax, Tmin, rh, lat, ssh, alti, u2)
    #         pet_daily.append(pet)
        
    #     pet_series = pd.Series(pet_daily, index=data.index)
    #     pre_series = data["prec"]
        
    #     # 按指定时间段汇总
    #     if period == "year":
    #         yearly_pet = pet_series.groupby(pet_series.index.year).sum()
    #         yearly_pre = pre_series.groupby(pre_series.index.year).sum()
    #     elif period == "months" and months:
    #         start_month, end_month = months
    #         pet_period = pet_series[pet_series.index.month.between(start_month, end_month)]
    #         pre_period = pre_series[pre_series.index.month.between(start_month, end_month)]
    #         yearly_pet = pet_period.groupby(pet_period.index.year).sum()
    #         yearly_pre = pre_period.groupby(pre_period.index.year).sum()
    #     else:
    #         raise ValueError("不支持的period参数")
        
    #     # 计算干燥度指数
    #     yearly_adi = {}
    #     for year in set(yearly_pet.index) & set(yearly_pre.index):
    #         if yearly_pre[year] > 0:
    #             yearly_adi[year] = yearly_pet[year] / yearly_pre[year]
    #         else:
    #             yearly_adi[year] = np.nan
        
    #     if frequency == "yearly":
    #         return yearly_adi
    #     else:
    #         values = [v for v in yearly_adi.values() if not np.isnan(v)]
    #         return np.nanmean(values) if values else np.nan


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

        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str, year_offset)
            period_data = data.loc[period_mask,variable]
            if period_data.empty:
                return np.nan if frequency == "lta" else {}
        
        # # 创建年份掩码的向量化方法
        # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
        #     """为整个时间序列创建年份掩码"""
        #     # 获取所有年份
        #     years = data_index.year.unique()
            
        #     # 为每个年份创建时间段掩码
        #     all_masks = []
        #     year_info = []
        #     for year in years:
        #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #         end_year = year + year_offset
        #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
        #         year_mask = (data_index >= start_date) & (data_index <= end_date)
        #         all_masks.append(year_mask)
        #         year_info.append(year)
            
        #     # 合并所有年份的掩码
        #     if all_masks:
        #         combined_mask = all_masks[0]
        #         for mask in all_masks[1:]:
        #             combined_mask = combined_mask | mask
        #         return combined_mask, year_info
        #     else:
        #         return pd.Series(False, index=data_index), []
        
        # # 创建时间段掩码
        # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        # if not period_mask.any():
        #     return np.nan if frequency == "lta" else {}
        
        # # 使用掩码一次性计算所有年份的平均值
        # period_data = data.loc[period_mask, variable]
        
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
        # year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # # 创建年份掩码
        # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
        #     years = data_index.year.unique()
        #     all_masks = []
        #     year_info = []
            
        #     for year in years:
        #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #         end_year = year + year_offset
        #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
        #         year_mask = (data_index >= start_date) & (data_index <= end_date)
        #         all_masks.append(year_mask)
        #         year_info.append(year)
            
        #     if all_masks:
        #         combined_mask = all_masks[0]
        #         for mask in all_masks[1:]:
        #             combined_mask = combined_mask | mask
        #         return combined_mask, year_info
        #     else:
        #         return pd.Series(False, index=data_index), []
        
        # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        # if not period_mask.any():
        #     return np.nan if frequency == "lta" else {}
        
        # # 使用掩码一次性计算所有年份的累计值
        # period_data = data.loc[period_mask, variable]
        
        # if period_data.empty:
        #     return np.nan if frequency == "lta" else {}

        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            period_data = data.loc[period_mask,variable]
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
    
    def _calculate_period_extreme(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "yearly") -> Union[float, Dict[int, float]]:
        """优化的时间段极值计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        variable = config["variable"]
        extreme_type = config.get("extreme_type")
        # year_offset = config.get("year_offset", 0)

        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # # 创建年份掩码
        # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
        #     years = data_index.year.unique()
        #     all_masks = []
        #     year_info = []
            
        #     for year in years:
        #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #         end_year = year + year_offset
        #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
        #         year_mask = (data_index >= start_date) & (data_index <= end_date)
        #         all_masks.append(year_mask)
        #         year_info.append(year)
            
        #     if all_masks:
        #         combined_mask = all_masks[0]
        #         for mask in all_masks[1:]:
        #             combined_mask = combined_mask | mask
        #         return combined_mask, year_info
        #     else:
        #         return pd.Series(False, index=data_index), []
        
        # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        # if not period_mask.any():
        #     return np.nan if frequency == "lta" else {}
        
        # # 使用掩码一次性计算所有年份的极值
        # period_data = data.loc[period_mask, variable]
        
        # if period_data.empty:
        #     return np.nan if frequency == "lta" else {}
        
        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
            period_data = data.loc[period_mask,variable]
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

    def _calculate_daily_range_mean(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[
        float, Dict[int, float]]:
        """
        计算日较差（日最高温-日最低温）的平均值 - 支持多种频率输出

        参数：
            data: 包含tmax和tmin字段的DataFrame
            config: 配置字典
            frequency: 输出频率，"lta"或"yearly"

        返回：
            多年平均或逐年字典
        """
        try:
            # 参数验证
            required_fields = ['tmax', 'tmin']
            for field in required_fields:
                if field not in data.columns:
                    print(f"错误：数据中缺少必要字段 '{field}'")
                    return np.nan if frequency == "lta" else {}

            # 获取配置参数
            start_date_str = config.get("start_date")
            end_date_str = config.get("end_date")

            if not start_date_str or not end_date_str:
                print("错误：配置中缺少start_date或end_date")
                return np.nan if frequency == "lta" else {}

            year_offset = config.get("year_offset", 0)

            if data.empty:
                return np.nan if frequency == "lta" else {}

            # 创建时间段掩码
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str, year_offset)

            if not period_mask.any():
                print(f"警告：时间段 {start_date_str} 到 {end_date_str} 内没有数据")
                return np.nan if frequency == "lta" else {}

            # 筛选时间段内的数据
            period_data = data.loc[period_mask]

            if period_data.empty:
                return np.nan if frequency == "lta" else {}

            # 计算日较差（最高温-最低温）
            daily_range = period_data['tmax'] - period_data['tmin']

            # 检查是否有有效数据
            if daily_range.isna().all():
                print("警告：所有日较差计算结果为NaN")
                return np.nan if frequency == "lta" else {}

            # 按年份分组计算每年平均值
            yearly_means = daily_range.groupby(daily_range.index.year).mean()

            # 清理无效值
            yearly_means = yearly_means.dropna()

            if frequency == "yearly":
                # 返回逐年数据
                result = yearly_means.to_dict()
                # print(f"逐年日较差平均值: {result}")
                return result
            else:
                # 返回多年平均
                if yearly_means.empty:
                    return np.nan

                lta_mean = float(yearly_means.mean())
                # print(f"多年平均日较差: {lta_mean:.2f}°C")
                return lta_mean

        except Exception as e:
            print(f"计算日较差时发生错误: {str(e)}")
            return np.nan if frequency == "lta" else {}

    def _calculate_conditional_sum(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> Union[float, Dict[int, float]]:
        """优化的条件累计计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        conditions = config["conditions"]
        variable = config["variable"]
        # year_offset = config.get("year_offset", 0)
        
        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # # 创建年份掩码
        # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
        #     years = data_index.year.unique()
        #     all_masks = []
        #     year_info = []
            
        #     for year in years:
        #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #         end_year = year + year_offset
        #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
        #         year_mask = (data_index >= start_date) & (data_index <= end_date)
        #         all_masks.append(year_mask)
        #         year_info.append(year)
            
        #     if all_masks:
        #         combined_mask = all_masks[0]
        #         for mask in all_masks[1:]:
        #             combined_mask = combined_mask | mask
        #         return combined_mask, year_info
        #     else:
        #         return pd.Series(False, index=data_index), []
        
        # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        # if not period_mask.any():
        #     return np.nan if frequency == "lta" else {}
        
        # # 使用掩码获取时间段数据
        # period_data = data.loc[period_mask]
        
        # if period_data.empty:
        #     return np.nan if frequency == "lta" else {}

        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
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
    
    def _calculate_conditional_count(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "yearly") -> Union[float, Dict[int, float]]:
        """优化的条件计数计算 - 支持多种频率输出"""
        start_date_str = config["start_date"]
        end_date_str = config["end_date"]
        conditions = config["conditions"]
        # year_offset = config.get("year_offset", 0)

        if data.empty:
            return np.nan if frequency == "lta" else {}
        
        # # 创建年份掩码
        # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
        #     years = data_index.year.unique()
        #     all_masks = []
        #     year_info = []
            
        #     for year in years:
        #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
        #         end_year = year + year_offset
        #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")
                
        #         year_mask = (data_index >= start_date) & (data_index <= end_date)
        #         all_masks.append(year_mask)
        #         year_info.append(year)
            
        #     if all_masks:
        #         combined_mask = all_masks[0]
        #         for mask in all_masks[1:]:
        #             combined_mask = combined_mask | mask
        #         return combined_mask, year_info
        #     else:
        #         return pd.Series(False, index=data_index), []
        
        # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)
        
        # if not period_mask.any():
        #     return np.nan if frequency == "lta" else {}
              
        # # 使用掩码获取时间段数据
        # period_data = data.loc[period_mask]
        
        # if period_data.empty:
        #     return np.nan if frequency == "lta" else {}

        # 时间筛选
        if start_date_str and end_date_str:
            period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
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

        # 计算活动积温 - 温度小于等于base_temp时为0，大于时取原温度值
        if method == "mean":
            # 使用日平均温度计算
            daily_gdd = np.where(period_data["tavg"] > base_temp, period_data["tavg"], 0)
        elif method == "min_max":
            # 使用日最高最低温度计算
            daily_mean_temp = (period_data["tmax"] + period_data["tmin"]) / 2
            daily_gdd = np.where(daily_mean_temp > base_temp, daily_mean_temp, 0)
        else:
            raise ValueError(f"不支持的积温计算方法: {method}")

        # 将numpy数组转换为pandas Series以便使用groupby
        daily_gdd_series = pd.Series(daily_gdd, index=period_data.index)

        # 按年份分组计算每年积温
        yearly_gdd = daily_gdd_series.groupby(daily_gdd_series.index.year).sum()
        

        if frequency == "yearly":
            # 返回逐年数据
            return yearly_gdd.to_dict()
        else:
            # 返回多年平均
            return yearly_gdd.mean() if not yearly_gdd.empty else 0

    # def _calculate_total_radiation(self, data: pd.DataFrame, config: Dict[str, Any], frequency: str = "lta") -> \
    # Union[float, Dict[int, float]]:
    #     """优化的活动积温计算 - 支持多种频率输出"""
    #     start_date_str = config["start_date"]
    #     end_date_str = config["end_date"]
    #     # year_offset = config.get("year_offset", 0)

    #     if data.empty:
    #         if frequency == "daily":
    #             return pd.DataFrame()
    #         else:
    #             return np.nan if frequency == "lta" else {}
        
    #     # 时间筛选
    #     if start_date_str and end_date_str:
    #         period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
    #         data = data.loc[period_mask]
    #         if data.empty:
    #             if frequency == "daily":
    #                 return pd.DataFrame()
    #             else:
    #                 return np.nan if frequency == "lta" else {}

    #     # # 创建年份掩码
    #     # def create_year_mask(data_index, start_date_str, end_date_str, year_offset):
    #     #     years = data_index.year.unique()
    #     #     all_masks = []
    #     #     year_info = []

    #     #     for year in years:
    #     #         start_date = pd.to_datetime(f"{year}-{start_date_str}")
    #     #         end_year = year + year_offset
    #     #         end_date = pd.to_datetime(f"{end_year}-{end_date_str}")

    #     #         year_mask = (data_index >= start_date) & (data_index <= end_date)
    #     #         all_masks.append(year_mask)
    #     #         year_info.append(year)

    #     #     if all_masks:
    #     #         combined_mask = all_masks[0]
    #     #         for mask in all_masks[1:]:
    #     #             combined_mask = combined_mask | mask
    #     #         return combined_mask, year_info
    #     #     else:
    #     #         return pd.Series(False, index=data_index), []

    #     def sunset_hour_angle(latitude, solar_declination):
    #         """
    #         计算日落时角（弧度）

    #         参数:
    #         - latitude: 纬度（度数）
    #         - solar_declination: 太阳赤纬（弧度）

    #         返回:
    #         - 日落时角（弧度）
    #         """
    #         lat_rad = np.radians(latitude)

    #         # 日落时角公式: cos(ωs) = -tan(φ) * tan(δ)
    #         cos_omega = -np.tan(lat_rad) * np.tan(solar_declination)

    #         # 限制在有效范围内 [-1, 1]
    #         cos_omega = np.clip(cos_omega, -1.0, 1.0)

    #         # 计算日落时角
    #         omega_s = np.arccos(cos_omega)

    #         return omega_s

    #     def solar_declination(day_of_year):
    #         """
    #         计算太阳赤纬

    #         参数:
    #         - day_of_year: 年内的日序数（1-365/366）

    #         返回:
    #         - 太阳赤纬（弧度）
    #         """
    #         return 0.409 * np.sin((2*np.pi/365)*day_of_year-1.39)

    #     def d_m_2(day_of_year):
    #         '''地球轨道偏心率订正系数'''
    #         return 1+0.033*np.cos((2*np.pi/365)*day_of_year)

    #     def calculate_solar_radiation(latitude, day_of_year):
    #         """
    #         计算总辐射Ra (MJ/m²/day)

    #         公式: Ra = I_0 × d_m_2 × T/π × (ω_s × sin(纬度) × sin(太阳赤纬) + cos(纬度) × cos(太阳赤纬) × sin(ω_s))
    #         """
    #         # 常数
    #         I_0 = 0.0820  # 太阳常数 (MJ/m²/min)
    #         T = 1440  # 每天的分钟数

    #         # 计算中间变量
    #         delta = solar_declination(day_of_year)  # 太阳赤纬
    #         d_m2 = d_m_2(day_of_year)  # 地球轨道偏心率订正系数
    #         omega_s = sunset_hour_angle(latitude, delta)  # 日落时角

    #         lat_rad = np.radians(latitude)  # 纬度转换为弧度

    #         # 计算总辐射
    #         term1 = omega_s * np.sin(lat_rad) * np.sin(delta)
    #         term2 = np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)

    #         Ra = I_0 * d_m2 * (T / np.pi) * (term1 + term2)

    #         return Ra

    #     # period_mask, years = create_year_mask(data.index, start_date_str, end_date_str, year_offset)

    #     # if not period_mask.any():
    #     #     return np.nan if frequency == "lta" else {}

    #     # # 使用掩码获取时间段数据
    #     # period_data = data.loc[period_mask]

    #     # if period_data.empty:
    #     #     return np.nan if frequency == "lta" else {}

    #     # 时间筛选
    #     if start_date_str and end_date_str:
    #         period_mask = self._create_period_mask(data.index, start_date_str, end_date_str)
    #         period_data = data.loc[period_mask]
    #         if period_data.empty:
    #             return np.nan if frequency == "lta" else {}

    #     # 计算总辐射公式
    #     period_data['day_of_year'] = period_data.index.dayofyear  # pandas内置的日序数计算
    #     # 假设数据中包含纬度信息，这里需要根据实际情况获取纬度
    #     # 如果数据中有纬度列，使用数据中的纬度；否则使用固定纬度
    #     if 'lat' in period_data.columns:
    #         # 使用每个站点的实际纬度
    #         period_data['solar_radiation'] = period_data.apply(
    #             lambda row: calculate_solar_radiation(row['lat'], row['day_of_year']), axis=1
    #         )
    #     else:
    #         # 使用固定纬度（需要你提供实际纬度值）
    #         fixed_latitude = 40.0  # 示例纬度，请替换为实际值
    #         period_data['solar_radiation'] = period_data['day_of_year'].apply(
    #             lambda doy: calculate_solar_radiation(fixed_latitude, doy)
    #         )

    #     # 按年份分组计算每年总辐射
    #     yearly_solar_radiation = period_data.groupby(period_data.index.year)['solar_radiation'].sum()

    #     if frequency == "yearly":
    #         # 返回逐年数据
    #         return yearly_solar_radiation.to_dict()
    #     else:
    #         # 返回多年平均
    #         return yearly_solar_radiation.mean() if not yearly_solar_radiation.empty else 0

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