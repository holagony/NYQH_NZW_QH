#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内蒙古大豆产量区划计算器
"""

import os
import pandas as pd
import numpy as np
from osgeo import gdal
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import re

class CL_QXCL:
    """大豆产量区划计算器"""

    def __init__(self):
        self.config = None
        self._algorithms = None
        self.kc_params = None
        self.y1_params = None

    def calculate(self, params):
        """执行区划计算"""
        config = params['config']
        self._algorithms = params['algorithms']
        self.algorithm_config = params['algorithmConfig']
        self.config = config  # 保存配置供其他方法使用
        
        # # 默认路径
        # self.kc_file = params['growthPeriodPath']
        # self.y1_params_file = params['photosyntheticParamsPath']

        return self._calculate_element(params)

    def _calculate_element(self, params):
        """计算大豆产量区划要素"""
        try:
            # 1. 获取基础指标数据
            station_indicators = params['station_indicators']
            station_coords = params['station_coords']

            # 保存RY中间结果
            config = params['config']
            file_name = "intermediate_RY.tif"
            intermediate_dir = Path(config["resultPath"]) / "intermediate"
            output_path = intermediate_dir / file_name 
            if not  os.path.exists(output_path):           
                # 2. 加载参数文件
                self._load_parameters(params)
                
                # 3. 计算作物实际蒸散发ETc
                etc_results = self._calculate_etc(station_indicators)
                
                # 4. 计算气候生产潜力
                climate_potential_results = self._calculate_climate_potential(etc_results)
                
                # 5. 保存中间结果
                self._save_intermediate_results(climate_potential_results, params['config'])
                
                # 6. 分别对Y1、Y2、Y3进行插值
                y1_interpolated = self._perform_interpolation_for_indicator(
                    climate_potential_results, station_coords, params, "Y1")
                y2_interpolated = self._perform_interpolation_for_indicator(
                    climate_potential_results, station_coords, params, "Y2")
                y3_interpolated = self._perform_interpolation_for_indicator(
                    climate_potential_results, station_coords, params, "Y3")
                
                # 7. 计算相对生产力RY
                ry_interpolated = self._calculate_relative_yield_from_raster(y3_interpolated, params)
            
            else:
                ry_interpolated = self._load_intermediate_result(output_path)    
            # 8. 对相对生产力RY进行分级
            final_result = self._perform_classification(ry_interpolated, params)
            
            print(f'计算{params["config"].get("cropCode","")}-{params["config"].get("zoningType","")}-{params["config"].get("element","")}-区划完成')
            return final_result
            
        except Exception as e:
            print(f"大豆产量区划计算失败: {str(e)}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            raise

    def _load_parameters(self, params):
        """加载参数文件 - 处理CSV格式"""
        config = params['config']

        # 获取新的参数文件路径
        growth_period_path = config.get('growthPeriodPath')
        photo_params_path = config.get('photosyntheticParamsPath')

        # 检查文件是否存在
        if not growth_period_path or not Path(growth_period_path).exists():
            raise ValueError(f"生长周期参数文件不存在: {growth_period_path}")

        if not photo_params_path or not Path(photo_params_path).exists():
            raise ValueError(f"光合生产潜力参数文件不存在: {photo_params_path}")

        # 根据文件扩展名决定处理方式
        file_extension = Path(growth_period_path).suffix.lower()

        if file_extension == '.xlsx' or file_extension == '.xls':
            # Excel格式处理（保持原有逻辑）
            try:
                # 直接读取Excel，不指定任何转换
                growth_period_df = pd.read_excel(growth_period_path, sheet_name='三基点温度')

                # 将数字日期转换为字符串
                growth_period_df = self._convert_excel_dates_to_string(growth_period_df)

                # 暂时只提取大兴安岭东南麓的信息
                self.kc_params = growth_period_df[growth_period_df['地区'] == '大兴安岭东南麓'].copy()

                if self.kc_params.empty:
                    raise ValueError("在大兴安岭东南麓地区未找到对应的生长周期参数")

                # 加载作物系数表
                kc_df = pd.read_excel(growth_period_path, sheet_name='作物系数')

                # 将数字日期转换为字符串
                kc_df = self._convert_excel_dates_to_string(kc_df)

                # 合并作物系数到主参数表
                self.kc_params = self.kc_params.merge(kc_df[['生育期', 'kc']], on='生育期', how='left')

                # 打印调试信息
                print("加载的生长周期参数(Excel格式):")
                print(self.kc_params[['生育期', '开始日期', '结束日期', 'T0', 'T1', 'T2', 'kc']])

            except Exception as e:
                print(f"读取Excel生长周期参数文件失败: {str(e)}")
                raise

        elif file_extension == '.csv':
            # CSV格式处理（新的逻辑）
            try:
                # 读取CSV文件，尝试不同的编码
                try:
                    growth_period_df = pd.read_csv(growth_period_path, encoding='utf-8')
                except UnicodeDecodeError:
                    growth_period_df = pd.read_csv(growth_period_path, encoding='gbk')

                # 直接使用所有数据，不进行筛选
                self.kc_params = growth_period_df.copy()

                if self.kc_params.empty:
                    raise ValueError("生长周期参数文件为空")

                # 重置索引
                self.kc_params = self.kc_params.reset_index(drop=True)

                # 打印调试信息
                print("加载的生长周期参数(CSV格式):")
                print(f"列名: {self.kc_params.columns.tolist()}")
                print(f"记录数量: {len(self.kc_params)}")
                print(self.kc_params)

            except Exception as e:
                print(f"读取CSV生长周期参数文件失败: {str(e)}")
                raise

        else:
            raise ValueError(f"不支持的文件格式: {file_extension}。请使用.xlsx、.xls或.csv格式")

        # 加载光合生产潜力参数（原CL_Y1_parameters.csv）
        try:
            y1_params_df = pd.read_csv(photo_params_path, encoding='utf-8')
        except:
            try:
                y1_params_df = pd.read_csv(photo_params_path, encoding='gbk')
            except Exception as e:
                raise ValueError(f"无法读取光合生产潜力参数文件: {str(e)}")

        # 筛选对应作物的参数
        crop_code = config.get('cropCode', 'SPSO')
        crop_params = y1_params_df[y1_params_df['crop'] == crop_code]

        if crop_params.empty:
            raise ValueError(f"在参数文件中未找到作物 {crop_code} 的参数")

        self.y1_params = crop_params.iloc[0].to_dict()

    def _convert_excel_dates_to_string(self, df):
        """将Excel中的数字日期转换为中文日期字符串"""
        # 检查并转换开始日期列
        if '开始日期' in df.columns:
            df['开始日期'] = df['开始日期'].apply(self._excel_number_to_chinese_date)
        
        # 检查并转换结束日期列
        if '结束日期' in df.columns:
            df['结束日期'] = df['结束日期'].apply(self._excel_number_to_chinese_date)
        
        return df

    def _excel_number_to_chinese_date(self, excel_number):
        """将Excel日期数字转换为中文日期字符串"""
        try:
            # 如果已经是字符串，直接返回
            if isinstance(excel_number, str):
                return excel_number
            
            # 如果是数字，转换为日期
            if isinstance(excel_number, (int, float)):
                # Excel日期从1900-01-01开始
                base_date = pd.Timestamp('1899-12-30')
                date = base_date + pd.Timedelta(days=excel_number)
                
                # 转换为中文格式 "M月D日"
                month = date.month
                day = date.day
                return f"{month}月{day}日"
            
            # 其他情况返回原值
            return excel_number
        except Exception as e:
            print(f"转换Excel日期失败: {excel_number}, 错误: {e}")
            return excel_number

    def _calculate_etc(self, station_indicators):
        """计算作物实际蒸散发ETc"""
        print("计算作物实际蒸散发ETc...")
        
        # 如果station_indicators是DataFrame（逐日数据）
        if isinstance(station_indicators, pd.DataFrame):
            return self._calculate_etc_daily(station_indicators)
        else:
            # 如果是字典格式（站点数据），需要先转换为DataFrame
            # 这里假设需要重新计算逐日数据
            raise ValueError("需要逐日数据格式进行计算")

    def _calculate_etc_daily(self, daily_df):
        """使用逐日数据计算ETc"""
        print("计算作物实际蒸散发ETc...")
        
        # 转换kc参数表为逐日格式
        kc_daily_df = self._convert_kc_to_daily_format(daily_df)
        daily_df['datetime'] = daily_df['datetime'].astype(str)
        # 合并kc参数到daily_df
        daily_df_with_kc = daily_df.merge(
            kc_daily_df[['datetime', 'period', 'kc', 'T0', 'T1', 'T2']],
            on='datetime',
            how='left'
        )
        
        # 计算ETc = ET0 * Kc
        daily_df_with_kc['ETc'] = daily_df_with_kc['ET0'] * daily_df_with_kc['kc']
        
        return daily_df_with_kc

    def _convert_kc_to_daily_format(self, daily_df):
        """将kc参数表转换为逐日格式 - 日期保持字符串格式"""
        print("转换kc参数表为逐日格式...")
        
        # 正确解析daily_df中的日期以获取年份范围
        daily_dates = pd.to_datetime(daily_df['datetime'], format='%Y%m%d')
        years = daily_dates.dt.year.unique()
        min_year, max_year = years.min(), years.max()
        
        # 创建空的DataFrame存储结果
        kc_daily_list = []
        
        # 处理每个生长期
        for _, period_row in self.kc_params.iterrows():
            period_name = period_row['生育期']
            kc = period_row['kc']
            T0 = period_row['T0']
            T1 = period_row['T1']
            T2 = period_row['T2']
            
            # 获取开始和结束日期字符串（直接使用，不转换格式）
            start_date_str = period_row['开始日期']  # 如 "5月12日"
            end_date_str = period_row['结束日期']    # 如 "5月25日"
            
            # 为每个年份生成该生长期的逐日数据
            for year in range(min_year, max_year + 1):
                # 构建完整的日期字符串（保持原格式）
                start_date = f"{year}年{start_date_str}"  # 如 "1991年5月12日"
                end_date = f"{year}年{end_date_str}"      # 如 "1991年5月25日"
                
                # 转换为datetime对象用于日期范围生成（仅内部使用）
                try:
                    start_dt = pd.to_datetime(start_date, format='%Y年%m月%d日')
                    end_dt = pd.to_datetime(end_date, format='%Y年%m月%d日')
                    
                    # 处理跨年情况
                    if end_dt < start_dt:
                        end_dt = pd.to_datetime(f"{year+1}年{end_date_str}", format='%Y年%m月%d日')
                    
                    # 生成日期序列
                    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
                    
                    # 创建该生长期的逐日数据，保持输出日期格式为YYYYMMDD字符串
                    period_daily = pd.DataFrame({
                        'datetime': date_range.strftime('%Y%m%d'),  # 保持字符串格式
                        'period': period_name,
                        'kc': kc,
                        'T0': T0,
                        'T1': T1,
                        'T2': T2
                    })
                    
                    kc_daily_list.append(period_daily)
                except Exception as e:
                    print(f"处理生长期 {period_name} 的日期转换失败: {e}")
                    continue
            
        # 合并所有生长期的逐日数据
        if kc_daily_list:
            kc_daily_df = pd.concat(kc_daily_list, ignore_index=True)
            
            # 去除重复的日期（如果有重叠）
            kc_daily_df = kc_daily_df.drop_duplicates(subset=['datetime'], keep='first')
            
            return kc_daily_df
        else:
            return pd.DataFrame()

    def _convert_kc_to_daily_format(self, daily_df):
        """将kc参数表转换为逐日格式 - 处理中文日期字符串"""
        print("转换kc参数表为逐日格式...")
        
        # 正确解析daily_df中的日期以获取年份范围
        daily_dates = pd.to_datetime(daily_df['datetime'], format='%Y%m%d')
        years = daily_dates.dt.year.unique()
        min_year, max_year = years.min(), years.max()
        
        # 创建空的DataFrame存储结果
        kc_daily_list = []
        
        # 处理每个生长期
        for _, period_row in self.kc_params.iterrows():
            period_name = period_row['生育期']  # 注意列名可能是中文
            kc = period_row['kc']
            T0 = period_row['T0']
            T1 = period_row['T1']
            T2 = period_row['T2']
            
            # 获取开始和结束日期字符串（中文格式）
            start_date_str = str(period_row['开始日期'])  # 确保是字符串，如 "5月12日"
            end_date_str = str(period_row['结束日期'])    # 确保是字符串，如 "5月25日"
            
            # print(f"处理生长期: {period_name}, 开始: {start_date_str}, 结束: {end_date_str}")
            
            # 为每个年份生成该生长期的逐日数据
            for year in range(min_year, max_year + 1):
                try:
                    # 将中文日期转换为标准日期格式
                    start_date = self._parse_chinese_date(year, start_date_str)
                    end_date = self._parse_chinese_date(year, end_date_str)
                    
                    # 处理跨年情况
                    if end_date < start_date:
                        end_date = self._parse_chinese_date(year + 1, end_date_str)
                    
                    # 生成日期序列
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    # 创建该生长期的逐日数据，保持输出日期格式为YYYYMMDD字符串
                    period_daily = pd.DataFrame({
                        'datetime': date_range.strftime('%Y%m%d'),  # 保持字符串格式
                        'period': period_name,
                        'kc': kc,
                        'T0': T0,
                        'T1': T1,
                        'T2': T2
                    })
                    
                    kc_daily_list.append(period_daily)
                except Exception as e:
                    print(f"处理生长期 {period_name} 的日期转换失败: {e}")
                    continue
            
        # 合并所有生长期的逐日数据
        if kc_daily_list:
            kc_daily_df = pd.concat(kc_daily_list, ignore_index=True)
            
            # 去除重复的日期（如果有重叠）
            kc_daily_df = kc_daily_df.drop_duplicates(subset=['datetime'], keep='first')
            
            print(f"生成KC逐日数据: {len(kc_daily_df)} 条记录")
            return kc_daily_df
        else:
            print("警告: 未能生成任何KC逐日数据")
            return pd.DataFrame()

    def _parse_chinese_date(self, year, chinese_date_str):
        """解析中文日期字符串为datetime对象"""
        # 移除可能存在的空格和特殊字符
        chinese_date_str = str(chinese_date_str).strip()
        
        # 处理不同的中文日期格式
        if '月' in chinese_date_str and '日' in chinese_date_str:
            # 格式: "5月12日"
            month_day = chinese_date_str.replace('月', '-').replace('日', '')
            date_str = f"{year}-{month_day}"
            return pd.to_datetime(date_str, format='%Y-%m-%d')
        elif '/' in chinese_date_str:
            # 格式: "5/12"
            month_day = chinese_date_str.split('/')
            if len(month_day) == 2:
                date_str = f"{year}-{month_day[0]}-{month_day[1]}"
                return pd.to_datetime(date_str, format='%Y-%m-%d')
        elif '-' in chinese_date_str:
            # 格式: "5-12"
            month_day = chinese_date_str.split('-')
            if len(month_day) == 2:
                date_str = f"{year}-{month_day[0]}-{month_day[1]}"
                return pd.to_datetime(date_str, format='%Y-%m-%d')
        
        # 如果无法解析，抛出异常
        raise ValueError(f"无法解析中文日期格式: {chinese_date_str}")


    # def _convert_kc_to_daily_format(self, daily_df):
    #     """将kc参数表转换为逐日格式"""
    #     print("转换kc参数表为逐日格式...")
        
    #     # 正确解析daily_df中的日期以获取年份范围
    #     daily_dates = pd.to_datetime(daily_df['datetime'], format='%Y%m%d')
    #     years = daily_dates.dt.year.unique()
    #     min_year, max_year = years.min(), years.max()
        
    #     # 创建空的DataFrame存储结果
    #     kc_daily_list = []
        
    #     # 处理每个生长期
    #     for _, period_row in self.kc_params.iterrows():
    #         period_name = period_row['生育期']
    #         kc = period_row['kc']
    #         T0 = period_row['T0']
    #         T1 = period_row['T1']
    #         T2 = period_row['T2']
            
    #         # 正确转换日期格式（处理"7月3日"这样的格式）
    #         start_date_mmdd = self._convert_chinese_date(period_row['开始日期'])
    #         end_date_mmdd = self._convert_chinese_date(period_row['结束日期'])
            
    #         # 为每个年份生成该生长期的逐日数据
    #         for year in range(min_year, max_year + 1):
    #             # 构建完整的日期
    #             start_date = f"{year}-{start_date_mmdd}"
    #             end_date = f"{year}-{end_date_mmdd}"
                
    #             # 处理跨年情况
    #             start_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    #             end_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
                
    #             if end_dt < start_dt:
    #                 end_dt = pd.to_datetime(f"{year+1}-{end_date_mmdd}", format='%Y-%m-%d')
                
    #             # 生成日期序列
    #             date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
                
    #             # 创建该生长期的逐日数据
    #             period_daily = pd.DataFrame({
    #                 'datetime': date_range.strftime('%Y%m%d'),
    #                 'period': period_name,
    #                 'kc': kc,
    #                 'T0': T0,
    #                 'T1': T1,
    #                 'T2': T2
    #             })
                
    #             kc_daily_list.append(period_daily)
        
    #     # 合并所有生长期的逐日数据
    #     kc_daily_df = pd.concat(kc_daily_list, ignore_index=True)
        
    #     # 去除重复的日期（如果有重叠）
    #     kc_daily_df = kc_daily_df.drop_duplicates(subset=['datetime'], keep='first')
        
    #     return kc_daily_df

    # def _convert_chinese_date(self, chinese_date_str):
    #     """将中文日期格式转换为MM-DD格式"""
    #     # 移除"月"和"日"，然后分割
    #     parts = chinese_date_str.replace('月', '-').replace('日', '').split('-')
        
    #     if len(parts) != 2:
    #         raise ValueError(f"无法解析中文日期格式: {chinese_date_str}")
        
    #     month = parts[0].zfill(2)  # 月份补零
    #     day = parts[1].zfill(2)    # 日期补零
        
    #     return f"{month}-{day}"

    def _calculate_climate_potential(self, daily_df_with_kc):
        """计算气候生产潜力 - 向量化版本"""
        print("计算气候生产潜力...")
        
        # 正确解析日期并添加年份列
        daily_df_with_kc['year'] = pd.to_datetime(daily_df_with_kc['datetime'], format='%Y%m%d').dt.year
        
        # 按站点、年份、生长期分组计算各生长期的统计量
        period_stats = daily_df_with_kc.groupby(['station_id', 'year', 'period']).agg({
            'Rs': 'sum',           # 总辐射量
            'tavg': 'mean',        # 平均温度
            'precip': 'sum',         # 总降水量
            'ETc': 'sum',          # 总需水量
            'T0': 'first',         # 下限温度
            'T1': 'first',         # 最适温度
            'T2': 'first',         # 上限温度
            'lat': 'first',
            'lon': 'first',
            'altitude': 'first',
            'province': 'first',
            'city': 'first',
            'county': 'first'
        }).reset_index()
        
        # 计算各生长期的Y1, Y2, Y3
        period_stats['Y1'] = period_stats.apply(
            lambda row: self._calculate_Y1(row['Rs']), axis=1
        )
        
        period_stats['Y2'] = period_stats.apply(
            lambda row: self._calculate_Y2(row['Y1'], row['tavg'], row['T0'], row['T1'], row['T2']), axis=1
        )
        
        period_stats['Y3'] = period_stats.apply(
            lambda row: self._calculate_Y3(row['Y2'], row['precip'], row['ETc']), axis=1
        )
        
        # 按站点和年份对各生长期结果进行累加，得到逐年结果
        yearly_results = period_stats.groupby(['station_id', 'year']).agg({
            'Y1': 'sum',
            'Y2': 'sum',
            'Y3': 'sum',
            'lat': 'first',
            'lon': 'first',
            'altitude': 'first',
            'province': 'first',
            'city': 'first',
            'county': 'first'
        }).reset_index()
        
        # 按站点对多年结果进行平均
        station_y3_mean = yearly_results.groupby('station_id').agg({
            'Y1': 'mean',
            'Y2': 'mean',
            'Y3': 'mean',
            'lat': 'first',
            'lon': 'first',
            'altitude': 'first',
            'province': 'first',
            'city': 'first',
            'county': 'first'
        }).reset_index()
        
        return {
            'period_results': period_stats,
            'yearly_results': yearly_results,
            'station_y3_mean': station_y3_mean
        }

    def _calculate_Y1(self, total_radiation):
        """计算光合生产潜力Y1"""
        C = 10000  # 单位换算系数
        S = self.y1_params['S']
        epsilon = self.y1_params['epsilon']
        phi = self.y1_params['phi']
        alpha = self.y1_params['alpha']
        beta = self.y1_params['beta']
        rho = self.y1_params['rho']
        gamma = self.y1_params['gamma']
        omega = self.y1_params['omega']
        fL = self.y1_params['fL']
        E = self.y1_params['E']
        q = self.y1_params['q']
        eta = self.y1_params['eta']
        ksi = self.y1_params['ksi']
        
        Y1 = C * S * epsilon * phi * (1 - alpha) * (1 - beta) * (1 - rho) * \
             (1 - gamma) * (1 - omega) * fL * E * total_radiation / \
             (q * (1 - eta) * (1 - ksi))
        
        return Y1

    def _calculate_Y2(self, Y1, mean_temperature, T0, T1, T2):
        """计算光温生产潜力Y2"""
        if mean_temperature < T1 or mean_temperature > T2:
            F_T = 0
        elif T1 <= mean_temperature <= T2:
            if mean_temperature == T0:
                F_T = 1
            else:
                B = (T2 - T0) / (T0 - T1)
                numerator = (mean_temperature - T1) * (T2 - mean_temperature) ** B
                denominator = (T0 - T1) * (T2 - T0) ** B
                F_T = numerator / denominator
        else:
            F_T = 0
            
        Y2 = Y1 * F_T
        return Y2

    def _calculate_Y3(self, Y2, total_precipipitation, total_etc):
        """计算气候生产潜力Y3"""
        if total_etc == 0:
            return 0
            
        P = total_precipipitation
        
        if P < 0.9 * total_etc:
            F_P = P / (0.9 * total_etc)
        elif 0.9 * total_etc <= P <= 1.2 * total_etc:
            F_P = 1
        else:
            F_P = (1.2 * total_etc) / P
            
        Y3 = Y2 * F_P
        return Y3

    def _save_intermediate_results(self, climate_results, config):
        """保存中间结果"""
        result_path = config.get("resultPath", "./results")
        intermediate_dir = Path(result_path) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存逐年逐生育期结果
        period_file = intermediate_dir / "intermediate_SPSO_CL_period_results.csv"
        climate_results['period_results'].to_csv(period_file, index=False, encoding='gbk')
        print(f"保存逐年逐生育期结果到: {period_file}")
        
        # 保存逐年累积结果
        yearly_file = intermediate_dir / "intermediate_SPSO_CL_yearly_results.csv"
        climate_results['yearly_results'].to_csv(yearly_file, index=False, encoding='gbk')
        print(f"保存逐年累积结果到: {yearly_file}")
        
        # 保存站点多年平均Y3结果
        y3_file = intermediate_dir / "intermediate_SPSO_CL_Y3.csv"
        climate_results['station_y3_mean'].to_csv(y3_file, index=False, encoding='gbk')
        print(f"保存站点气候生产潜力结果到: {y3_file}")

    def _perform_interpolation_for_indicator(self, climate_results, station_coords, params, indicator_name,min_value=np.nan,max_value=np.nan):
        """对指定指标进行插值计算"""
        print(f"执行{indicator_name}插值计算...")
        
        config = params['config']
        station_y3_mean = climate_results['station_y3_mean']
        
        if station_y3_mean.empty:
            raise ValueError(f"没有有效的站点数据进行{indicator_name}插值")
        
        # 准备插值数据
        station_values = {}
        for _, row in station_y3_mean.iterrows():
            station_id = row['station_id']
            station_values[station_id] = row[indicator_name]
        
        # 获取插值算法
        algorithmConfig = params['algorithmConfig']
        interpolation_config = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation_config.get("method", "lsm_idw")
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        
        # 准备插值参数
        interpolation_data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", ""),
            'min_value':min_value,
            'max_value':max_value,
        }
        
        file_name = f"intermediate_{indicator_name}.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name
        
        if not os.path.exists(output_path):
            interpolated_result = interpolator.execute(interpolation_data, interpolation_config.get("params", {}))
            print(f"{indicator_name}插值完成")
            
            # 保存中间结果
            self._save_intermediate_raster(interpolated_result, output_path)
        else:
            interpolated_result = self._load_intermediate_raster(output_path)
        
        return interpolated_result

    def _calculate_relative_yield_from_raster(self, y3_interpolated, params):
        """基于Y3栅格数据计算相对生产力RY = Y3/Y3max * 100%"""
        print("计算相对生产力RY...")
        
        # 获取Y3插值数据
        y3_data = y3_interpolated['data']
        
        # 找出研究区域内的Y3最大值（排除无效值）
        valid_y3 = y3_data[~np.isnan(y3_data)]
        
        if len(valid_y3) == 0:
            print("警告: 没有有效的Y3数据用于计算相对生产力")
            # 返回原始Y3数据，不进行计算
            return y3_interpolated
        
        Y3max = np.max(valid_y3)
        print(f"研究区域Y3最大值: {Y3max}")
        
        # 计算相对生产力RY = Y3/Y3max * 100%
        # 使用向量化操作，避免除零错误
        with np.errstate(divide='ignore', invalid='ignore'):
            RY = np.where(
                (~np.isnan(y3_data)),
                (y3_data / Y3max) * 100,  # 转换为百分比
                np.nan
            )
        
        # 创建新的插值结果对象
        ry_interpolated = {
            'data': RY,
            'meta': y3_interpolated['meta'].copy()
        }
        
        # 保存RY中间结果
        config = params['config']
        file_name = "intermediate_RY.tif"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        output_path = intermediate_dir / file_name
        self._save_intermediate_raster(ry_interpolated, output_path)
        
        # 打印统计信息
        valid_RY = RY[~np.isnan(RY)]
        if len(valid_RY) > 0:
            print(f"相对生产力RY统计: 最小值={np.min(valid_RY):.2f}%, 最大值={np.max(valid_RY):.2f}%, 平均值={np.mean(valid_RY):.2f}%")
        
        return ry_interpolated

    def _perform_classification(self, ry_interpolated, params):
        """对相对生产力RY进行分级计算"""
        print("执行相对生产力RY分级计算...")
        
        algorithmConfig = params['algorithmConfig']
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(ry_interpolated['data'], classification)
        
        ry_interpolated['data'] = data
        return ry_interpolated

    def _get_algorithm(self, algorithm_name):
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_intermediate_raster(self, result, output_path):
        """保存中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        data = result['data']
        meta = result['meta']
        
        # 根据数据类型确定GDAL数据类型
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32
        
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
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
        band.SetNoDataValue(0)
        
        band.FlushCache()
        dataset = None
        
        print(f"中间栅格结果已保存: {output_path}")

    def _load_intermediate_raster(self, input_path):
        """加载中间栅格结果"""
        from osgeo import gdal
        import numpy as np
        
        dataset = gdal.Open(str(input_path))
        if dataset is None:
            raise FileNotFoundError(f"无法打开文件: {input_path}")
        
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        
        transform = dataset.GetGeoTransform()
        crs = dataset.GetProjection()
        
        meta = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'transform': transform,
            'crs': crs
        }
        
        dataset = None
        
        return {
            'data': data,
            'meta': meta
        }
        
    def _load_intermediate_result(self, file_path: Path) -> Dict[str, Any]:
        """加载中间结果"""
        ds = gdal.Open(str(file_path))
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()
        data = ds.ReadAsArray()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        
        return {
            'data': data,
            'meta': {
                'transform': geo,
                'crs': proj,
                'height': data.shape[0],
                'width': data.shape[1],
                'dtype': data.dtype,
                'nodata': nodata
            }
        }