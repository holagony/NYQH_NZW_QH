import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os
import re
import time
import multiprocessing
from datetime import datetime, timedelta
from tqdm import tqdm
from .indicators import IndicatorCalculator

class DataManager:
    """数据管理类 - 支持逐日数据输出"""
    
    def __init__(self, data_dir: str, station_file: str = None, multiprocess: bool = True, num_processes: int = 8):
        self.data_dir = Path(data_dir)
        self.station_file = Path(station_file) if station_file else None
        # self.logger = logging.getLogger(__name__)  # 新增日志
        self._cache = {}    

        self._station_info_cache = {}
        self.indicator_calculator = IndicatorCalculator()
        self.multiprocess = multiprocess
        self.num_processes = num_processes if num_processes else multiprocessing.cpu_count()
        # 字段映射：原始字段名 -> 标准字段名
        self.field_mapping = {
            "Datetime": "date",
            "TEM_Avg": "tavg",
            "TEM_Max": "tmax",
            "TEM_Min": "tmin",
            "SSH": "sunshine",
            "RHU_Avg": "rhum",
            "PRE_Time_2020": "precip",
            "WIN_S_2mi_Avg": "wind",
            "Station_Id_C": "station_id",
            "Lat": "lat",
            "Lon": "lon",
            "Alti": "altitude"
        }
        
        # 加载站点信息
        if self.station_file and self.station_file.exists():
            self._load_station_info()
    
    def calculate_indicators_for_stations(self, station_ids: List[str], indicator_configs: Dict[str, Any],
                                        start_date: str = None, end_date: str = None,
                                        output_csv: str = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """计算多个站点的多个指标 - 返回常规结果和逐日结果"""
        print(f"开始计算 {len(station_ids)} 个站点的指标")
        print(f"指标配置: {list(indicator_configs.keys())}")
        
        # 检查是否有daily频率的指标
        has_daily_indicators = any(
            config.get("frequency") == "daily" 
            for config in indicator_configs.values()
        )
        
        if self.multiprocess:
            results_df, daily_results_df = self._calculate_parallel_optimized(
                station_ids, indicator_configs, start_date, end_date, output_csv, has_daily_indicators
            )
        else:
            results_df, daily_results_df = self._calculate_sequential_optimized(
                station_ids, indicator_configs, start_date, end_date, output_csv, has_daily_indicators
            )
        
        # 保存常规结果
        if output_csv and not results_df.empty:
            self._save_intermediate_csv(results_df, output_csv, indicator_configs)
        
        # 保存逐日结果
        if has_daily_indicators and daily_results_df is not None and not daily_results_df.empty:
            daily_output_csv = str(Path(output_csv).with_name(f"daily_{Path(output_csv).name}"))
            self._save_daily_results(daily_results_df, daily_output_csv)
        
        # 检查结果
        if not results_df.empty:
            self._print_indicator_stats(results_df, indicator_configs)
        
        if daily_results_df is not None and not daily_results_df.empty:
            print(f"逐日数据包含 {len(daily_results_df)} 条记录，时间范围: {daily_results_df['datetime'].min()} 到 {daily_results_df['datetime'].max()}")
    
        return results_df, daily_results_df

    def _save_daily_results(self, daily_df: pd.DataFrame, output_csv: str) -> None:
        """保存逐日结果到CSV"""
        try:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保列的顺序
            base_columns = ['station_id', 'lat', 'lon', 'altitude', 'province', 'city', 'county', 'datetime']
            other_columns = [col for col in daily_df.columns if col not in base_columns]
            ordered_columns = base_columns + other_columns
            
            # 只保存存在的列
            existing_columns = [col for col in ordered_columns if col in daily_df.columns]
            daily_df_to_save = daily_df[existing_columns]
            
            # 保存为CSV
            daily_df_to_save.to_csv(output_path, encoding='gbk', index=False)
            print(f"逐日结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存逐日结果失败: {str(e)}")
            # 尝试UTF-8编码
            try:
                daily_df_to_save.to_csv(output_path, encoding='utf-8', index=False)
                print(f"逐日结果已保存到: {output_path} (使用UTF-8编码)")
            except Exception as e2:
                print(f"保存逐日结果最终失败: {str(e2)}")

    def _calculate_parallel_optimized(self, station_ids, indicator_configs, start_date, end_date, output_csv, has_daily_indicators):
        """优化的并行计算 - 返回常规结果和逐日结果"""
        print(f"使用多进程计算，进程数: {self.num_processes}")
        start_time = time.time()
        
        results_df = pd.DataFrame()
        daily_results_list = []
        
        import multiprocessing as mp
        from multiprocessing import Pool
        
        try:
            with Pool(processes=self.num_processes) as pool:
                # 准备任务
                tasks = [(station_id, indicator_configs, start_date, end_date, has_daily_indicators) 
                        for station_id in station_ids]
                
                # 使用imap_unordered提高性能
                results = list(tqdm(
                    pool.imap_unordered(self._station_worker_optimized, tasks),
                    total=len(station_ids),
                    desc="计算站点指标"
                ))
            
            # 分离常规结果和逐日结果
            valid_results = []
            for r in results:
                if r and 'error' not in r or r.get('error') is None:
                    # 分离逐日数据
                    if '_daily_data' in r:
                        daily_results_list.append(r['_daily_data'])
                        del r['_daily_data']
                    valid_results.append(r)
            
            error_results = [r for r in results if r and 'error' in r and r['error'] is not None]
            
            if valid_results:
                results_df = pd.DataFrame(valid_results)
            
            # 合并逐日数据
            if daily_results_list:
                daily_results_df = pd.concat(daily_results_list, ignore_index=True)
            else:
                daily_results_df = pd.DataFrame()
                
            if error_results:
                error_df = pd.DataFrame(error_results)
                print(f"有 {len(error_results)} 个站点计算失败")
            
        except Exception as e:
            print(f"多进程计算失败: {str(e)}，回退到单进程")
            return self._calculate_sequential_optimized(station_ids, indicator_configs, start_date, end_date, output_csv, has_daily_indicators)
        
        end_time = time.time()
        print(f'多进程计算完成，耗时 {end_time - start_time:.2f} 秒，成功: {len(valid_results)}/{len(station_ids)}')
        
        # 保存结果
        if output_csv and not results_df.empty:
            self._save_results_optimized(results_df, output_csv)
        
        return results_df, daily_results_df

    def _calculate_sequential_optimized(self, station_ids, indicator_configs, start_date, end_date, output_csv, has_daily_indicators):
        """优化的顺序计算 - 返回常规结果和逐日结果"""
        print("使用优化的单进程计算")
        start_time = time.time()
        
        results = []
        daily_results_list = []
        
        for station_id in tqdm(station_ids, desc="计算站点指标"):
            try:
                result = self._station_worker_optimized((station_id, indicator_configs, start_date, end_date, has_daily_indicators))
                if result:
                    # 分离逐日数据
                    if '_daily_data' in result:
                        daily_results_list.append(result['_daily_data'])
                        del result['_daily_data']
                    results.append(result)
            except Exception as e:
                print(f"站点 {station_id} 计算异常: {str(e)}")
                results.append({"station_id": station_id, "error": str(e)})
        
        # 批量创建DataFrame
        if results:
            results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame()
        
        # 合并逐日数据
        if daily_results_list:
            daily_results_df = pd.concat(daily_results_list, ignore_index=True)
        else:
            daily_results_df = pd.DataFrame()
        
        # 统计成功和失败
        success_count = len([r for r in results if 'error' not in r or not r.get('error')])
        error_count = len(results) - success_count
        
        end_time = time.time()
        print(f'单进程计算完成，耗时 {end_time - start_time:.2f} 秒，成功: {success_count}/{len(station_ids)}')
        
        # 保存结果
        if output_csv and not results_df.empty:
            self._save_results_optimized(results_df, output_csv)
        
        return results_df, daily_results_df

    def _station_worker_optimized(self, args):
        """优化的工作函数 - 支持逐日数据"""
        station_id, indicator_configs, start_date, end_date, has_daily_indicators = args
        try:
            # 加载站点数据
            data = self.load_station_data(station_id, start_date, end_date)
            if data.empty:
                return {"station_id": station_id, "error": "无数据"}
            
            station_results = {"station_id": station_id}
            daily_data_list = []
            
            # 获取站点信息
            station_info = self.get_station_info(station_id)
            station_results.update(station_info)
            
            # 计算每个指标
            for indicator_name, indicator_config in indicator_configs.items():
                try:
                    value = self.indicator_calculator.calculate(data, indicator_config)
                    
                    # 处理逐日数据
                    if (has_daily_indicators and 
                        indicator_config.get("frequency") == "daily" and 
                        isinstance(value, pd.DataFrame) and 
                        not value.empty):
                        
                        # 为逐日数据添加站点信息
                        daily_df = value.copy()
                        daily_df['station_id'] = station_id
                        for info_key in ['lat', 'lon', 'altitude', 'province', 'city', 'county']:
                            if info_key in station_info:
                                daily_df[info_key] = station_info[info_key]
                        
                        daily_data_list.append(daily_df)
                        
                        # 对于daily指标，也计算一个汇总值用于常规结果
                        if len(daily_df) > 0:
                            # 取第一个数值列的平均值作为汇总
                            value_columns = [col for col in daily_df.columns if col not in 
                                           ['station_id', 'lat', 'lon', 'altitude', 'province', 'city', 'county', 'datetime']]
                            if value_columns:
                                summary_value = daily_df[value_columns[0]].mean()
                                station_results[indicator_name] = summary_value
                            else:
                                station_results[indicator_name] = np.nan
                        else:
                            station_results[indicator_name] = np.nan
                    else:
                        # 常规指标
                        station_results[indicator_name] = value
                        
                except Exception as e:
                    print(f"站点 {station_id} 指标 {indicator_name} 计算失败: {str(e)}")
                    station_results[indicator_name] = np.nan
            
            # 合并所有逐日数据
            if daily_data_list:
                # 按datetime合并所有逐日指标
                from functools import reduce
                merged_daily = reduce(lambda left, right: pd.merge(left, right, 
                                                                  on=['station_id', 'lat', 'lon', 'altitude', 'province', 'city', 'county', 'datetime'], 
                                                                  how='outer'), daily_data_list)
                station_results['_daily_data'] = merged_daily
            
            return station_results
        except Exception as e:
            print(f"计算站点 {station_id} 失败: {str(e)}")
            return {"station_id": station_id, "error": str(e)}
    
    # 其他方法保持不变...


    # def calculate_indicators_for_stations(self, station_ids: List[str], indicator_configs: Dict[str, Any],
    #                                     start_date: str = None, end_date: str = None,
    #                                     output_csv: str = None) -> pd.DataFrame:
    #     """计算多个站点的多个指标 - 支持中间CSV输出"""
    #     # print(f"开始计算 {len(station_ids)} 个站点的指标")
    #     print(f"指标配置: {list(indicator_configs.keys())}")
        
    #     results_df = self.calculate_indicators_for_stations_optimized(
    #         station_ids, indicator_configs, start_date, end_date, output_csv
    #     )
        
    #     # 如果指定了输出CSV路径，保存中间结果
    #     if output_csv and not results_df.empty:
    #         self._save_intermediate_csv(results_df, output_csv, indicator_configs)
        
    #     # 检查结果
    #     if not results_df.empty:
    #         # 打印统计信息
    #         self._print_indicator_stats(results_df, indicator_configs)
    
    #     return results_df
    
    def _save_intermediate_csv(self, results_df: pd.DataFrame, output_csv: str, 
                             indicator_configs: Dict[str, Any]) -> None:
        """保存中间CSV结果"""
        try:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 选择需要保存的列
            columns_to_save = ['station_id', 'lat', 'lon', 'altitude']
            indicator_names = list(indicator_configs.keys())
            columns_to_save.extend(indicator_names)
            
            # 只保存存在的列
            existing_columns = [col for col in columns_to_save if col in results_df.columns]
            results_to_save = results_df[existing_columns]
            
            # 保存为CSV
            results_to_save.to_csv(output_path, encoding='gbk', index=False)
            print(f"中间CSV结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存中间CSV结果失败: {str(e)}")
    
    def _print_indicator_stats(self, results_df: pd.DataFrame, indicator_configs: Dict[str, Any]) -> None:
        """打印指标统计信息 - 适配 yearly 和 lta 数据类型"""
        indicator_names = list(indicator_configs.keys())
        
        print("指标计算结果统计:")
        for indicator in indicator_names:
            if indicator in results_df.columns:
                values = results_df[indicator]
                
                # 处理不同类型的数据
                valid_values = []
                for val in values:
                    if isinstance(val, dict):
                        # yearly 数据：字典类型，提取所有年份的值
                        yearly_values = [v for v in val.values() if not np.isnan(v)]
                        valid_values.extend(yearly_values)
                    elif isinstance(val, (int, float)) and not np.isnan(val):
                        # lta 数据：数值类型
                        valid_values.append(val)
                    # 其他情况（如字符串、None等）跳过
                
                if len(valid_values) > 0:
                    min_val = min(valid_values)
                    max_val = max(valid_values)
                    mean_val = sum(valid_values) / len(valid_values)
                    valid_count = len(valid_values)
                    
                    # 获取数据类型信息
                    data_type = "混合"
                    sample_value = values.iloc[0] if len(values) > 0 else None
                    if isinstance(sample_value, dict):
                        data_type = f"逐年数据({len(sample_value)}年)"
                    elif isinstance(sample_value, (int, float)):
                        data_type = "多年平均"
                    
                    print(f"  {indicator}[{data_type}]: 有效值{valid_count}个, 范围[{min_val:.2f}, {max_val:.2f}], 均值{mean_val:.2f}")
                else:
                    print(f"  {indicator}: 无有效值")
                    
    def calculate_indicators_for_stations_optimized(self, station_ids: List[str], indicator_configs: Dict[str, Any],
                                                  start_date: str = None, end_date: str = None,
                                                  output_csv: str = None) -> pd.DataFrame:
        """优化的批量指标计算方法"""
        if self.multiprocess:
            return self._calculate_parallel_optimized(station_ids, indicator_configs, start_date, end_date, output_csv)
        else:
            return self._calculate_sequential_optimized(station_ids, indicator_configs, start_date, end_date, output_csv)
    
    # def _calculate_parallel_optimized(self, station_ids, indicator_configs, start_date, end_date, output_csv):
    #     """优化的并行计算"""
    #     print(f"使用多进程计算，进程数: {self.num_processes}")
    #     start_time = time.time()
        
    #     results_df = pd.DataFrame()
        
    #     # 使用multiprocessing.Pool（更稳定）
    #     import multiprocessing as mp
    #     from multiprocessing import Pool
        
    #     try:
    #         with Pool(processes=self.num_processes) as pool:
    #             # 准备任务
    #             tasks = [(station_id, indicator_configs, start_date, end_date) 
    #                     for station_id in station_ids]
                
    #             # 使用imap_unordered提高性能
    #             results = list(tqdm(
    #                 pool.imap_unordered(self._station_worker_optimized, tasks),
    #                 total=len(station_ids),
    #                 desc="计算站点指标"
    #             ))
            
    #         # 合并结果
    #         valid_results = [r for r in results if r and 'error' not in r or r.get('error') is None]
    #         error_results = [r for r in results if r and 'error' in r and r['error'] is not None]
            
    #         if valid_results:
    #             results_df = pd.DataFrame(valid_results)
    #         if error_results:
    #             error_df = pd.DataFrame(error_results)
    #             print(f"有 {len(error_results)} 个站点计算失败")
    #             # 可以选择保存错误信息
            
    #     except Exception as e:
    #         print(f"多进程计算失败: {str(e)}，回退到单进程")
    #         return self._calculate_sequential_optimized(station_ids, indicator_configs, start_date, end_date, output_csv)
        
    #     end_time = time.time()
    #     print(f'多进程计算完成，耗时 {end_time - start_time:.2f} 秒，成功: {len(valid_results)}/{len(station_ids)}')
        
    #     # 保存结果
    #     if output_csv and not results_df.empty:
    #         self._save_results_optimized(results_df, output_csv)
        
    #     return results_df
    
    # def _calculate_sequential_optimized(self, station_ids, indicator_configs, start_date, end_date, output_csv):
    #     """优化的顺序计算"""
    #     print("使用优化的单进程计算")
    #     start_time = time.time()
        
    #     results = []
        
    #     for station_id in tqdm(station_ids, desc="计算站点指标"):
    #         try:
    #             result = self._station_worker_optimized((station_id, indicator_configs, start_date, end_date))
    #             if result:
    #                 results.append(result)
    #         except Exception as e:
    #             print(f"站点 {station_id} 计算异常: {str(e)}")
    #             results.append({"station_id": station_id, "error": str(e)})
        
    #     # 批量创建DataFrame
    #     if results:
    #         results_df = pd.DataFrame(results)
    #     else:
    #         results_df = pd.DataFrame()
        
    #     # 统计成功和失败
    #     success_count = len([r for r in results if 'error' not in r or not r.get('error')])
    #     error_count = len(results) - success_count
        
    #     end_time = time.time()
    #     print(f'单进程计算完成，耗时 {end_time - start_time:.2f} 秒，成功: {success_count}/{len(station_ids)}')
        
    #     # 保存结果
    #     if output_csv and not results_df.empty:
    #         self._save_results_optimized(results_df, output_csv)
        
    #     return results_df
    
    # def _station_worker_optimized(self, args):
    #     """优化的工作函数"""
    #     station_id, indicator_configs, start_date, end_date = args
    #     try:
    #         # 加载站点数据
    #         data = self.load_station_data(station_id, start_date, end_date)
    #         if data.empty:
    #             return {"station_id": station_id, "error": "无数据"}
            
    #         station_results = {"station_id": station_id}
            
    #         # 获取站点信息
    #         station_info = self.get_station_info(station_id)
    #         station_results.update(station_info)
            
    #         # 计算每个指标
    #         for indicator_name, indicator_config in indicator_configs.items():
    #             try:
    #                 value = self.indicator_calculator.calculate(data, indicator_config)
    #                 station_results[indicator_name] = value
    #             except Exception as e:
    #                 print(f"站点 {station_id} 指标 {indicator_name} 计算失败: {str(e)}")
    #                 station_results[indicator_name] = np.nan
            
    #         return station_results
    #     except Exception as e:
    #         print(f"计算站点 {station_id} 失败: {str(e)}")
    #         return {"station_id": station_id, "error": str(e)}
    
    def _save_results_optimized(self, results_df, output_csv):
        """优化的结果保存"""
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 尝试GBK编码
            results_df.to_csv(output_path, encoding='gbk', index=False)
            # print(f"中间结果已保存到: {output_path}")
        except UnicodeEncodeError:
            # 如果GBK失败，使用UTF-8
            try:
                results_df.to_csv(output_path, encoding='utf-8', index=False)
                print(f"中间结果已保存到: {output_path} (使用UTF-8编码)")
            except Exception as e:
                print(f"保存结果失败: {str(e)}")
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
    
    # 其他必要的方法保持不变
    def _load_station_info(self):
        """加载站点信息文件"""
        try:
            # 读取站点信息文件（GBK编码）
            station_df = pd.read_csv(self.station_file, encoding='gbk')
            
            # 处理列名可能的空格问题
            station_df.columns = station_df.columns.str.strip()
            
            # 将站点信息缓存到字典中
            for _, row in station_df.iterrows():
                station_id = str(row['站号']).strip()
                self._station_info_cache[station_id] = {
                    'station_name': row['站名'] if '站名' in row else '',
                    'station_id': station_id,
                    'lon': float(row['经度']) if '经度' in row else np.nan,
                    'lat': float(row['纬度']) if '纬度' in row else np.nan,
                    'altitude': float(row['海拔']) if '海拔' in row else np.nan,  # 新增海拔字段
                    'county_code': str(row['县编号']) if '县编号' in row else '',
                    'PAC': str(row['PAC']) if 'PAC' in row else '',
                    'county': row['县'] if '县' in row else '',
                    'province': row['省'] if '省' in row else '',
                    'city': row['市'] if '市' in row else '',
                    'PAC_prov': str(row['PAC_prov']) if 'PAC_prov' in row else '',
                    'PAC_city': str(row['PAC_city']) if 'PAC_city' in row else ''
                }
            
            print(f"成功加载 {len(self._station_info_cache)} 个站点的信息")
        except Exception as e:
            print(f"加载站点信息文件失败: {str(e)}")
    
    def get_stations_by_province(self, province_code: str) -> List[str]:
        """根据省份代码获取站点列表"""
        if not self._station_info_cache:
            # 如果没有站点信息文件，则从数据目录中获取所有站点
            station_files = self.data_dir.glob("*.csv")
            return [file.stem for file in station_files]
        
        # 根据省份代码筛选站点
        matched_stations = []
        for station_id, info in self._station_info_cache.items():
            # 使用PAC_prov进行匹配
            if (province_code == "000000") | (info.get('PAC_prov', '') == province_code):
                matched_stations.append(station_id)
        
        # print(f"找到 {len(matched_stations)} 个属于省份代码 {province_code} 的站点")
        return matched_stations
    
    def get_station_info(self, station_id: str) -> Dict[str, Any]:
        """获取站点基本信息"""
        # 首先从站点信息文件中获取
        if station_id in self._station_info_cache:
            return self._station_info_cache[station_id]
        
        # 如果没有站点信息文件，则从数据文件中获取
        try:
            data = self.load_station_data(station_id)
            if not data.empty:
                first_record = data.iloc[0]
                station_info = {
                    'station_id': station_id,
                    'lat': first_record.get('lat', np.nan),
                    'lon': first_record.get('lon', np.nan),
                    'altitude': first_record.get('altitude', np.nan),
                    'province': first_record.get('province', np.nan),
                    'city':first_record.get('city', np.nan),
                    'county': first_record.get('county', np.nan),
                }
                return station_info
        except:
            pass
        
        return {
            'station_id': station_id,
            'lat': np.nan,
            'lon': np.nan,
            'altitude': np.nan,
            'province': '',
            'city': '',
            'county': ''
        }

    def preprocessData(self,data):
        """
        数据清洗
        """
        # targetdata=data.iloc[:,1:]
        # data.iloc[:,1:]=targetdata.replace(["999999.0","999990.0", "999.0", "999999","-999999",-999999, 999999, 999, "999", np.nan, None], -999)
        # data.iloc[:,1:]=data.iloc[:,1:].astype(float)
        # data.iloc[:,1:]=targetdata.applymap(lambda x:x-999600 if (x>999600)&(x<999700) else x)
        # data.iloc[:,1:]=targetdata.applymap(lambda x:x-999700 if (x>999700)&(x<999800) else x)
        # data.iloc[:,1:]=targetdata.applymap(lambda x:x-999800 if (x>999800)&(x<999900) else x)
        # data.iloc[:,1:]=targetdata.applymap(lambda x:-999 if (x>1000)|(x<-1000) else x)
        
        data = data.replace(["999999.0","999990.0", "999.0", "999999","-999999",-999999, 999999,999990,999998, 999, "999", np.nan, None],np.nan)
        data[(data.values>999600) & (data.values<999700)] = data - 999600
        data[(data.values>999700) & (data.values<999800)] = data - 999700
        data[(data.values>999800) & (data.values<999900)] = data - 999800
        # data[(data.values > 1000) | (data.values < -1000) ] = -999 
        return data
    
    def load_station_data(self, station_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """加载站点的基础气象数据"""
        cache_key = f"{station_id}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从文件加载数据（假设每个站点一个CSV文件）
        file_path = self.data_dir / f"{station_id}.csv"

        if not file_path.exists():
            # 尝试其他可能的文件扩展名
            file_path = self.data_dir / f"{station_id}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {station_id}")
        import shutil
        # copypath = self.data_dir/ "150000" / f"{station_id}.csv"
        # os.makedirs(self.data_dir/ "150000",exist_ok=True)
        # shutil.copy(file_path,copypath)

        try:
            # 读取CSV文件
            data = pd.read_csv(file_path, dtype=str, encoding="gbk")
        except:
            try:
                data = pd.read_csv(file_path, dtype=str, encoding="utf-8")
            except Exception as e:
                raise ValueError(f"无法读取站点 {station_id} 的数据文件: {str(e)}")
        
        # 重命名字段
        data = data.rename(columns=self.field_mapping)

        # 选择数据
        stn_info = data.iloc[0:1,:]
        data = data.iloc[1:,:]  # 获取时间和四列变量
        data['lat']=stn_info['lat'][0]
        data['lon']=stn_info['lon'][0]
        data['altitude'] = stn_info['altitude'][0]
        # data['City'] = stn_info['City'][0]
        # data['Cnty'] = stn_info['Cnty'][0]
        
        # 转换日期格式
        try:
            data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
        except:
            try:
                data['date'] = pd.to_datetime(data['date'])
            except:
                raise ValueError(f"无法解析站点 {station_id} 的日期数据")
        
        data = data.set_index('date')
        
        # 转换数值字段
        numeric_columns = ['tavg', 'tmax', 'tmin', 'sunshine', 'rhum', 'precip','wind', 'lat', 'lon', 'altitude']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                # 数据预处理
                data[col] = self.preprocessData(data[col])   
                     
        # 过滤指定日期范围的数据
        if start_date is not None:
            start_dt = pd.to_datetime(start_date, format='%Y%m%d')
            data = data[data.index >= start_dt]
        if end_date is not None:
            end_dt = pd.to_datetime(end_date, format='%Y%m%d')
            data = data[data.index <= end_dt]
        
        # 缓存数据
        self._cache[cache_key] = data
        return data
    
    def calculate_indicator(self, station_id: str, indicator_config: Dict[str, Any], 
                           start_date: str = None, end_date: str = None) -> float:
        """计算单个站点的指标"""
        # 加载基础数据
        data = self.load_station_data(station_id, start_date, end_date)
        
        # 计算指标
        return self.indicator_calculator.calculate(data, indicator_config)
    
    def get_all_stations(self) -> List[str]:
        """获取所有站点ID"""
        station_files = self.data_dir.glob("*.csv")
        stations = [file.stem for file in station_files]
        # 如果没有CSV文件，尝试TXT文件
        if not stations:
            station_files = self.data_dir.glob("*.txt")
            stations = [file.stem for file in station_files]
        return stations
    
