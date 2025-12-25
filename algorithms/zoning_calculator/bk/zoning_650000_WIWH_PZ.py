import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from osgeo import gdal


class WIWH_PZ:
    """通用区划计算器 - 支持完全参数化配置"""

    def __init__(self):
        pass

    def calculate(self, params):
        """执行区划计算"""
        config = params['config']
        self._algorithms = params['algorithms']

        type = config['element']
        return self._calculate_element(params)

    def _calculate_element(self, params):
        """
        计算气候区划
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print(f'开始计算{config.get("cropCode", "")}-{config.get("zoningType", "")}-{config.get("element", "")}-流程')

        # 统一的计算流程
        result = self._calculate_with_interpolation(station_indicators, station_coords, config, algorithmConfig)

        # 保存分级前的综合指标
        composite_index_result = {
            'data': result['data'].copy(),
            'meta': result['meta'].copy()
        }
        self._save_composite_index(composite_index_result, config, "composite_index")

        # 分级
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(result['data'], classification)
        result['data'] = data

        print(f'计算{config.get("cropCode", "")}-{config.get("zoningType", "")}-{config.get("element", "")}-区划完成')
        return result

    def _load_parameters(self, params):
        """加载参数文件 - 直接处理数字日期"""
        config = params['config']

        # 获取新的参数文件路径
        growth_period_path = config.get('growthPeriodPath')

        # 检查文件是否存在
        if not growth_period_path or not Path(growth_period_path).exists():
            raise ValueError(f"生长周期参数文件不存在: {growth_period_path}")

        # 加载生长周期参数（xlsx格式）- 直接处理数字日期
        try:
            # 直接读取Excel，不指定任何转换
            growth_period_df = pd.read_excel(growth_period_path, sheet_name='三基点温度')

            # 将数字日期转换为字符串
            self.growth_period_df = self._convert_excel_dates_to_string(growth_period_df)

            if self.growth_period_df.empty:
                raise ValueError("在南疆和北疆未找到对应的生长周期参数")

            # 打印调试信息
            print("加载的生长周期参数:")
            print(self.growth_period_df[['地区', '生育期', '开始日期', '结束日期', ]])


        except Exception as e:
            print(f"读取生长周期参数文件失败: {str(e)}")
            raise

    def _save_composite_index(self, result: Dict[str, Any], config: Dict[str, Any], name: str):
        """保存分级前的综合指标"""
        try:
            print(f"保存分级前综合指标: {name}")

            # 创建输出目录
            output_dir = Path(config["resultPath"]) / "composite_index"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{name}.tif"

            # 保存为GeoTIFF
            self._save_geotiff_gdal(result['data'], result['meta'], output_path)
            print(f"综合指标保存成功: {output_path}")

            # 同时保存统计信息
            self._save_composite_stats(result['data'], output_dir, name)

        except Exception as e:
            print(f"保存综合指标失败: {str(e)}")

    def _save_composite_stats(self, data: np.ndarray, output_dir: Path, name: str):
        """保存综合指标统计信息"""
        try:
            valid_mask = ~np.isnan(data)
            valid_count = np.sum(valid_mask)
            total_count = data.size

            if valid_count > 0:
                valid_data = data[valid_mask]
                stats = {
                    'min': float(np.nanmin(valid_data)),
                    'max': float(np.nanmax(valid_data)),
                    'mean': float(np.nanmean(valid_data)),
                    'std': float(np.nanstd(valid_data)),
                    'valid_count': int(valid_count),
                    'total_count': int(total_count),
                    'valid_ratio': float(valid_count / total_count)
                }

                # 保存为JSON文件
                import json
                stats_path = output_dir / f"{name}_stats.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                print(f"综合指标统计: 范围[{stats['min']:.4f}, {stats['max']:.4f}], "
                      f"均值{stats['mean']:.4f}, 有效{valid_count}/{total_count}")

        except Exception as e:
            print(f"保存统计信息失败: {str(e)}")

    def _calculate_with_interpolation(self, station_indicators, station_coords,
                                      config, crop_config):
        """支持两种插值顺序的统一计算流程"""

        # 获取插值配置和顺序控制
        interpolation_config = crop_config.get("interpolation", {})
        interpolation_order = interpolation_config.get('order', 'before')
        interpolation_method = interpolation_config.get('method', 'idw')
        interpolation_params = interpolation_config.get('params', {})

        print(f"使用插值顺序: {interpolation_order}")

        # 获取插值算法
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        if interpolation_order == 'before':
            return self._interpolate_before_calculate(station_indicators, station_coords,
                                                      config, crop_config, interpolator,
                                                      interpolation_params)
        elif interpolation_order == 'after':
            return self._calculate_before_interpolate(station_indicators, station_coords,
                                                      config, crop_config, interpolator,
                                                      interpolation_params)
        else:
            raise ValueError(f"不支持的插值顺序: {interpolation_order}")

    def _interpolate_before_calculate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先插值各指标，再计算区划指标"""
        print("执行 before 插值顺序: 先插值各指标，再计算区划指标")

        indicator_configs = crop_config.get("indicators", {})
        interpolated_indicators = {}

        # 将站点数据转换为DataFrame
        station_df = self._convert_station_data_to_dataframe(station_indicators, indicator_configs)
        print(f"站点数据DataFrame形状: {station_df.shape}")

        for indicator_name in indicator_configs.keys():
            print(f"正在插值指标: {indicator_name}")

            # 从DataFrame中提取该指标的站点数值
            indicator_values = {}
            valid_count = 0

            for station_id in station_df.index:
                value = station_df.loc[station_id, indicator_name]
                if pd.notna(value):
                    indicator_values[station_id] = value
                    valid_count += 1
                else:
                    indicator_values[station_id] = np.nan

            if valid_count == 0:
                print(f"警告: 指标 {indicator_name} 没有有效数据，跳过插值")
                continue

            # 插值该指标
            try:
                interpolation_data = {
                    'station_values': indicator_values,
                    'station_coords': station_coords,
                    'dem_path': config.get("demFilePath", ""),
                    'shp_path': config.get("shpFilePath", ""),
                    'grid_path': config.get("gridFilePath", ""),
                    'area_code': config.get("areaCode", "")
                }

                # 检查中间结果是否存在
                file_name = indicator_name + ".tif"
                intermediate_dir = Path(config["resultPath"]) / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                output_path = intermediate_dir / file_name

                if not os.path.exists(output_path):
                    interpolated_dict = interpolator.execute(interpolation_data, interpolation_params)
                    self._save_intermediate_result(interpolated_dict, config, indicator_name)
                    print(f"指标 {indicator_name} 插值完成")
                else:
                    print(f"读取指标 {indicator_name} ")
                    interpolated_dict = self._load_intermediate_result(output_path)

                interpolated_indicators[indicator_name] = interpolated_dict

            except Exception as e:
                print(f"指标 {indicator_name} 插值失败: {str(e)}")
                interpolated_indicators[indicator_name] = None

        # 检查是否所有必要指标都成功插值
        required_indicators = list(indicator_configs.keys())
        successful_indicators = [name for name in required_indicators if
                                 name in interpolated_indicators and interpolated_indicators[name] is not None]

        if len(successful_indicators) == 0:
            raise ValueError("所有指标插值都失败了")

        print(f"成功读取指标: {successful_indicators}")

        # 使用插值后的栅格数据计算区划指标
        composite_result = self.calculate_composite_indicator(interpolated_indicators, crop_config)
        return composite_result

    def _calculate_before_interpolate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先根据站点计算数值，再插值"""
        print("执行 after 插值顺序: 先根据站点计算数值，再插值")

        # 检查是否需要归一化处理
        formula_config = crop_config.get("formula", {})
        variables_config = formula_config.get("variables", {})

        # 判断是否有需要归一化的变量
        has_normalization = any(
            var_config.get("type") == "standardize"
            for var_config in variables_config.values()
        )

        if has_normalization:
            print("检测到需要归一化的变量，执行站点级别归一化后再插值")
            return self._calculate_with_station_normalization(
                station_indicators, station_coords, config, crop_config,
                interpolator, interpolation_params
            )
        else:
            print("无需归一化处理，执行标准站点计算流程")
            return self._calculate_without_normalization(
                station_indicators, station_coords, config, crop_config,
                interpolator, interpolation_params
            )

    def _convert_station_data_to_dataframe(self, station_indicators: Dict, indicator_configs: Dict) -> pd.DataFrame:
        """将站点数据转换为DataFrame"""
        data_dict = {}
        print(station_indicators.head(5))

        for station_id, station_data in station_indicators.items():
            row_data = {}

            for indicator_name in indicator_configs.keys():
                if isinstance(station_data, dict) and indicator_name in station_data:
                    value = station_data[indicator_name]
                else:
                    value = station_data

                # 确保值是数值类型
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row_data[indicator_name] = value
                else:
                    row_data[indicator_name] = np.nan

            data_dict[station_id] = row_data

        # 创建DataFrame
        df = pd.DataFrame.from_dict(data_dict, orient='index')

        # 打印数据统计
        print("站点数据统计:")
        for col in df.columns:
            valid_count = df[col].notna().sum()
            total_count = len(df)
            if valid_count > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                print(f"  {col}: {valid_count}/{total_count} 有效, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
            else:
                print(f"  {col}: 0/{total_count} 有效")

        return df

    def _calculate_with_station_normalization(self, station_indicators, station_coords,
                                              config, crop_config, interpolator, interpolation_params):
        """在站点级别进行归一化处理后再插值"""
        print("执行站点级别归一化流程")

        formula_config = crop_config.get("formula", {})
        variables_config = formula_config.get("variables", {})
        indicator_configs = crop_config.get("indicators", {})

        # 第一步：将站点数据转换为DataFrame
        station_df = self._convert_station_data_to_dataframe(station_indicators, indicator_configs)

        # 第二步：计算全局归一化统计信息
        print("计算全局归一化统计信息...")
        global_stats = self._compute_global_normalization_stats_df(station_df, variables_config)

        # 第三步：使用全局统计信息对DataFrame进行归一化
        print("对站点数据进行归一化处理...")
        normalized_df = self._normalize_station_dataframe(station_df, variables_config, global_stats)

        # 第四步：计算每个站点的综合指标
        print("计算站点综合指标...")
        station_composite_values = self._calculate_composite_from_dataframe(normalized_df, formula_config)

        valid_station_count = station_composite_values.notna().sum()
        print(f"成功计算归一化综合指标的站点数: {valid_station_count}/{len(station_composite_values)}")

        if valid_station_count == 0:
            raise ValueError("所有站点计算归一化综合指标都失败了")

        # 对归一化后的综合指标结果进行插值
        try:
            interpolation_data = {
                'station_values': station_composite_values.to_dict(),
                'station_coords': station_coords,
                'dem_path': config.get("demFilePath", ""),
                'shp_path': config.get("shpFilePath", ""),
                'grid_path': config.get("gridFilePath", ""),
                'area_code': config.get("areaCode", "")
            }

            # 检查中间结果是否存在
            file_name = "normalized_composite.tif"
            intermediate_dir = Path(config["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            if not os.path.exists(output_path):
                composite_result = interpolator.execute(interpolation_data, interpolation_params)
                print("归一化综合指标插值完成")

                # 保存中间结果
                self._save_intermediate_result(composite_result, config, "normalized_composite")
            else:
                print(f"读取指标normalized_composite")
                composite_result = self._load_intermediate_result(output_path)

            return composite_result

        except Exception as e:
            print(f"归一化综合指标插值失败: {str(e)}")
            raise

    def _compute_global_normalization_stats_df(self, station_df: pd.DataFrame,
                                               variables_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """使用DataFrame计算所有站点的全局归一化统计信息"""
        global_stats = {}

        # 收集所有需要归一化的指标数据
        for var_name, var_config in variables_config.items():
            if var_config.get("type") == "standardize":
                value_config = var_config.get("value", {})
                if "ref" in value_config:
                    ref_name = value_config["ref"]
                    if ref_name in station_df.columns:
                        # 使用DataFrame的列直接计算统计信息
                        valid_values = station_df[ref_name].dropna()
                        if len(valid_values) > 0:
                            global_stats[ref_name] = {
                                'min': valid_values.min(),
                                'max': valid_values.max(),
                                'mean': valid_values.mean(),
                                'count': len(valid_values)
                            }
                            print(
                                f"指标 {ref_name}: 范围[{global_stats[ref_name]['min']:.4f}, {global_stats[ref_name]['max']:.4f}], "
                                f"均值{global_stats[ref_name]['mean']:.4f}, 站点数{global_stats[ref_name]['count']}")
                        else:
                            global_stats[ref_name] = {
                                'min': 0,
                                'max': 1,
                                'mean': 0.5,
                                'count': 0
                            }
                            print(f"警告: 指标 {ref_name} 没有有效数据，使用默认范围[0, 1]")

        return global_stats

    def _normalize_station_dataframe(self, station_df: pd.DataFrame,
                                     variables_config: Dict[str, Any],
                                     global_stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """使用全局统计信息对DataFrame进行归一化"""
        normalized_data = {}

        for var_name, var_config in variables_config.items():
            var_type = var_config.get("type", "")

            if var_type == "standardize":
                value_config = var_config.get("value", {})
                if "ref" in value_config:
                    ref_name = value_config["ref"]
                    if ref_name in station_df.columns and ref_name in global_stats:
                        raw_values = station_df[ref_name]
                        stats = global_stats[ref_name]

                        if stats['max'] != stats['min']:
                            normalized_values = (raw_values - stats['min']) / (stats['max'] - stats['min'])
                        else:
                            normalized_values = pd.Series(1.0, index=raw_values.index)  # 所有值相同的情况

                        normalized_data[var_name] = normalized_values
                    else:
                        normalized_data[var_name] = pd.Series(np.nan, index=station_df.index)

            elif "ref" in var_config:
                ref_name = var_config["ref"]
                if ref_name in station_df.columns:
                    normalized_data[var_name] = station_df[ref_name]
                else:
                    normalized_data[var_name] = pd.Series(np.nan, index=station_df.index)

            else:
                # 常量值
                value = var_config.get("value", 0)
                normalized_data[var_name] = pd.Series(value, index=station_df.index)

        # 创建归一化后的DataFrame
        normalized_df = pd.DataFrame(normalized_data, index=station_df.index)

        # 打印归一化后的统计信息
        print("归一化后变量统计:")
        for col in normalized_df.columns:
            valid_count = normalized_df[col].notna().sum()
            total_count = len(normalized_df)
            if valid_count > 0:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                mean_val = normalized_df[col].mean()
                print(f"  {col}: {valid_count}/{total_count} 有效, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")

        return normalized_df

    def _calculate_composite_from_dataframe(self, normalized_df: pd.DataFrame,
                                            formula_config: Dict[str, Any]) -> pd.Series:
        """从归一化DataFrame计算综合指标"""
        formula_str = formula_config.get("formula", "")
        weights = formula_config.get("weight", {})

        # 准备计算环境
        local_env = {'np': np, 'pd': pd}
        local_env.update(weights)

        # 为每个变量添加DataFrame列到计算环境
        for col in normalized_df.columns:
            local_env[col] = normalized_df[col]

        try:
            # 执行公式计算
            result = eval(formula_str, {"__builtins__": {}}, local_env)

            # 确保结果是Series
            if isinstance(result, pd.Series):
                return result
            else:
                # 如果是标量，创建相同值的Series
                return pd.Series(result, index=normalized_df.index)

        except Exception as e:
            print(f"DataFrame公式计算失败: {str(e)}")
            return pd.Series(np.nan, index=normalized_df.index)

    def _calculate_without_normalization(self, station_indicators, station_coords,
                                         config, crop_config, interpolator, interpolation_params):
        """无需归一化的标准站点计算流程"""
        indicator_configs = crop_config.get("indicators", {})
        nan_bei_jiang = crop_config.get("nanjiang_beijiang", {})


        # 根据站点所属区域计算相应日期范围的多个指标（多年平均）
        station_df = self.calculate_regional_indicators(config,station_indicators,nan_bei_jiang)

        # 计算每个站点的综合指标
        station_composite_values = {}
        valid_station_count = 0

        for station_id in station_df.index:
            try:
                # 获取站点的所有指标值
                station_values = station_df.loc[station_id].to_dict()

                # 在站点级别计算综合指标
                composite_value = self._calculate_composite_value(station_values, crop_config)

                if composite_value is not None and not np.isnan(composite_value):
                    station_composite_values[station_id] = composite_value
                    valid_station_count += 1
                else:
                    station_composite_values[station_id] = np.nan
                    print(f"站点 {station_id} 计算失败，结果为NaN")

            except Exception as e:
                print(f"站点 {station_id} 计算异常: {str(e)}")
                station_composite_values[station_id] = np.nan

        print(f"成功计算综合指标的站点数: {valid_station_count}/{len(station_df)}")

        if valid_station_count == 0:
            raise ValueError("所有站点计算综合指标都失败了")

        # 对综合指标结果进行插值
        try:
            interpolation_data = {
                'station_values': station_composite_values,
                'station_coords': station_coords,
                'dem_path': config.get("demFilePath", ""),
                'shp_path': config.get("shpFilePath", ""),
                'grid_path': config.get("gridFilePath", ""),
                'area_code': config.get("areaCode", "")
            }

            file_name = "composite_indicator.tif"
            intermediate_dir = Path(config["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            composite_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合指标插值完成")

            # 保存中间结果
            self._save_intermediate_result(composite_result, config, "composite_indicator")

            # if not os.path.exists(output_path):
            #     composite_result = interpolator.execute(interpolation_data, interpolation_params)
            #     print("综合指标插值完成")
            #
            #     # 保存中间结果
            #     self._save_intermediate_result(composite_result, config, "composite_indicator")
            # else:
            #     composite_result = self._load_intermediate_result(output_path)

            return composite_result

        except Exception as e:
            print(f"综合指标插值失败: {str(e)}")
            raise

    def calculate_regional_indicators(self, config: Dict[str, Any], station_indicators: pd.DataFrame,
                                      regional_config: Dict) -> pd.DataFrame:
        """根据站点所属区域计算相应日期范围的多个指标（多年平均）"""

        print("=== 计算区域多指标（多年平均）===")

        # 确保datetime列是日期格式
        if 'datetime' in station_indicators.columns:
            station_indicators['date'] = pd.to_datetime(station_indicators['datetime'], format='%Y%m%d')
        else:
            raise ValueError("数据中缺少datetime列")

        # 提取年份和月日
        station_indicators['year'] = station_indicators['date'].dt.year
        station_indicators['month_day'] = station_indicators['date'].dt.strftime('%m-%d')

        # 构建地区映射关系
        region_mapping = {}
        for region_name, region_config in regional_config.items():
            for area in region_config['area']:
                region_mapping[area] = region_name

        # 计算每个站点的多个指标
        results = {}

        for station_id, station_group in station_indicators.groupby('station_id'):
            # 获取站点所属地区（取第一个记录的地区信息）
            first_record = station_group.iloc[0]
            city = first_record['city']

            # 确定站点属于哪个区域
            station_region = None
            for area, region_name in region_mapping.items():
                if area in city:
                    station_region = region_name
                    break

            if station_region is None:
                print(f"警告: 站点 {station_id} 所在地区 {city} 未在区域配置中找到")
                continue

            # 获取该区域的日期范围
            region_config = regional_config[station_region]
            start_date = region_config['start_date']  # "05-11"
            end_date = region_config['end_date']  # "07-10"

            # print(f"站点 {station_id} 属于 {station_region}, 日期范围: {start_date} 到 {end_date}")

            # 初始化各指标的年度数据列表
            yearly_sunshine_totals = []
            yearly_high_temp_counts_30 = []  # ≥30℃日数
            yearly_high_temp_counts_32 = []  # ≥32℃日数
            yearly_avg_temp_means = []  # 日平均气温
            yearly_max_temp_means = []  # 平均日最高气温

            # 计算每年的各项指标
            for year, year_group in station_group.groupby('year'):
                # 筛选该年份内日期范围内的数据
                mask = (year_group['month_day'] >= start_date) & (year_group['month_day'] <= end_date)
                period_data = year_group[mask]

                if len(period_data) > 0:
                    # 1. 计算累计日照
                    total_sunshine = period_data['sunshine'].sum()
                    yearly_sunshine_totals.append(total_sunshine)

                    # 2. 计算高温日数（日最高气温≥30℃）
                    high_temp_days_30 = (period_data['tmax'] >= 30).sum()
                    yearly_high_temp_counts_30.append(high_temp_days_30)

                    # 3. 计算高温日数（日最高气温≥32℃）
                    high_temp_days_32 = (period_data['tmax'] >= 32).sum()
                    yearly_high_temp_counts_32.append(high_temp_days_32)

                    # 4. 计算日平均气温均值
                    avg_temp_mean = period_data['tavg'].mean()
                    yearly_avg_temp_means.append(avg_temp_mean)

                    # 5. 计算平均日最高气温
                    max_temp_mean = period_data['tmax'].mean()
                    yearly_max_temp_means.append(max_temp_mean)

            if yearly_sunshine_totals:  # 如果有有效数据
                # 计算多年平均值
                mean_sunshine = np.mean(yearly_sunshine_totals)
                mean_high_temp_days_30 = np.mean(yearly_high_temp_counts_30)
                mean_high_temp_days_32 = np.mean(yearly_high_temp_counts_32)
                mean_avg_temp = np.mean(yearly_avg_temp_means)
                mean_max_temp = np.mean(yearly_max_temp_means)
                years_count = len(yearly_sunshine_totals)

                # 以station_id为键存储结果
                results[station_id] = {
                    'region': station_region,
                    'SUN': mean_sunshine,
                    'Tmax30D': mean_high_temp_days_30,
                    'Tmax32D': mean_high_temp_days_32,
                    'Tave': mean_avg_temp,
                    'Tmax': mean_max_temp,
                    'years_count': years_count,
                    'lat': first_record['lat'],
                    'lon': first_record['lon'],
                    'city': city
                }

        # 创建结果DataFrame，设置station_id为索引
        result_df = pd.DataFrame.from_dict(results, orient='index')
        result_df.index.name = 'station_id'

        print(f"计算完成，共 {len(result_df)} 个站点")
        return result_df

    def calculate_composite_indicator(self, indicators: Dict[str, Any],
                                      crop_config: Dict[str, Any]) -> Any:
        """统一的综合指标计算入口 - 合并站点和栅格计算逻辑"""
        print("执行综合指标计算")

        # 获取公式配置
        formula_config = crop_config.get("formula", {})
        if not formula_config:
            raise ValueError("未找到公式配置")

        # 处理简单引用配置
        if "ref" in formula_config:
            ref_name = formula_config["ref"]
            if ref_name in indicators:
                result = indicators[ref_name]
                print(f"直接返回指标 {ref_name} 的结果")
                return result
            else:
                raise ValueError(f"引用的指标 {ref_name} 不存在")

        # 统一计算逻辑，不再区分站点和栅格
        return self._calculate_unified(indicators, crop_config)

    def _calculate_unified(self, indicators: Dict[str, Any], crop_config: Dict[str, Any]) -> Any:
        """统一的指标计算逻辑，同时支持站点数据和栅格数据"""
        formula_config = crop_config.get("formula", {})
        formula_str = formula_config.get("formula", "")
        variables_config = formula_config.get("variables", {})

        # 检测数据类型并统一处理
        data_type = self._detect_data_type(indicators)
        print(f"检测到数据类型: {data_type}")

        # 准备变量数据
        variables_data = {}
        for var_name, var_config in variables_config.items():
            print(f"处理变量: {var_name}")
            var_value = self._compute_variable_unified(var_config, indicators, data_type)
            variables_data[var_name] = var_value

        # # 计算公式（包含权重处理）
        # try:
        #     result_data = self._evaluate_formula_with_weights(formula_str, variables_data, formula_config)
        #     valid_count = np.sum(~np.isnan(result_data))
        #     total_count = result_data.size
        #     if valid_count > 0:
        #         min_val = np.nanmin(result_data)
        #         max_val = np.nanmax(result_data)
        #         mean_val = np.nanmean(result_data)
        #         print(f"综合指标结果: {valid_count}/{total_count} 有效像素, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
        #
        # except Exception as e:
        #     print(f"公式计算失败: {str(e)}")
        #     raise
        # 计算公式（包含权重处理）
        try:
            result_data = self._evaluate_formula_with_weights(formula_str, variables_data, formula_config)

            # 统一转换为numpy数组处理
            if isinstance(result_data, (int, float)):
                result_array = np.array([result_data])
            else:
                result_array = np.array(result_data)

            valid_count = np.sum(~np.isnan(result_array))
            total_count = result_array.size

            if valid_count > 0:
                min_val = np.nanmin(result_array)
                max_val = np.nanmax(result_array)
                mean_val = np.nanmean(result_array)

                if total_count == 1:
                    print(f"综合指标结果: 单个数值 {mean_val:.4f}")
                else:
                    print(
                        f"综合指标结果: {valid_count}/{total_count} 有效值, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
            else:
                print(f"综合指标结果: 无有效数据")

        except Exception as e:
            print(f"公式计算失败: {str(e)}")
            raise
        # 根据数据类型包装结果
        if data_type == "station":
            # 站点数据结果包装
            result = {
                'data': np.array([result_data]),
                'meta': {
                    'calculation_level': 'station',
                    'station_count': 1,
                    'description': '站点级别计算结果'
                }
            }
            print(f"站点级别计算结果: {result_data:.4f}")
        else:
            # 栅格数据结果包装
            first_indicator = next(iter(indicators.values()))
            result = {
                'data': result_data,
                'meta': first_indicator['meta']
            }

        return result

    def _calculate_composite_value(self, indicators: Dict[str, Any], crop_config: Dict[str, Any]) -> float:
        """计算单个站点的综合指标值"""
        # 复用统一计算逻辑
        result = self._calculate_unified(indicators, crop_config)
        return result['data'][0] if 'data' in result else np.nan

    def _detect_data_type(self, indicators: Dict[str, Any]) -> str:
        """检测数据类型"""
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict) and 'data' in indicator_data:
                data_content = indicator_data['data']
                if isinstance(data_content, np.ndarray):
                    if data_content.ndim > 1 or data_content.size > 100:
                        return "grid"
            # 如果是单值或者字典格式的站点数据
            if isinstance(indicator_data, (int, float, dict)) or (
                    isinstance(indicator_data, np.ndarray) and indicator_data.size == 1):
                return "station"
        return "grid"

    def _compute_variable_unified(self, var_config: Dict[str, Any],
                                  indicators: Dict[str, Any],
                                  data_type: str) -> Union[float, np.ndarray]:
        """统一的变量计算逻辑"""
        var_type = var_config.get("type", "")

        if var_type == "standardize":
            value_config = var_config.get("value", {})
            if "ref" in value_config:
                ref_name = value_config["ref"]
                if ref_name in indicators:
                    data = self._extract_data(indicators[ref_name], data_type)
                    return self.dataNormal_array(data)
                else:
                    raise ValueError(f"引用的指标不存在: {ref_name}")
            else:
                raise ValueError("标准化配置缺少引用")

        elif "value" in var_config:
            value_config = var_config.get("value", {})
            ref_name = value_config["ref"]
            if ref_name in indicators:
                return self._extract_data(indicators[ref_name], data_type)
            else:
                raise ValueError(f"引用的指标不存在: {ref_name}")

        elif var_type == "custom_formula":
            # 递归处理自定义公式
            formula_str = var_config.get("formula", "")
            sub_variables_config = var_config.get("variables", {})

            sub_variables_data = {}
            for sub_var_name, sub_var_config in sub_variables_config.items():
                sub_variables_data[sub_var_name] = self._compute_variable_unified(
                    sub_var_config, indicators, data_type)

            return self._evaluate_formula(formula_str, sub_variables_data)

        else:
            # 常量值
            value = var_config.get("value", 0)
            if data_type == "station":
                return value
            else:
                # 创建与输入栅格相同大小的常量栅格
                first_indicator = next(iter(indicators.values()))
                first_data = self._extract_data(first_indicator, data_type)
                return np.full_like(first_data, value)

    def _extract_data(self, indicator_data: Any, data_type: str) -> Union[float, np.ndarray]:
        """从指标数据中提取数值或数组"""
        if data_type == "station":
            # 站点数据提取
            if isinstance(indicator_data, (int, float)):
                return indicator_data
            elif isinstance(indicator_data, dict):
                # 如果是字典格式的站点数据，返回第一个值或特定处理
                if 'data' in indicator_data:
                    data_content = indicator_data['data']
                    if isinstance(data_content, np.ndarray) and data_content.size == 1:
                        return data_content[0]
                    else:
                        return data_content
                else:
                    # 如果是普通的站点指标字典，返回第一个数值
                    for key, value in indicator_data.items():
                        if isinstance(value, (int, float)):
                            return value
                    return np.nan
            elif isinstance(indicator_data, np.ndarray) and indicator_data.size == 1:
                return indicator_data[0]
            else:
                return np.nan
        else:
            # 栅格数据提取
            if isinstance(indicator_data, dict) and 'data' in indicator_data:
                return indicator_data['data']
            elif isinstance(indicator_data, np.ndarray):
                return indicator_data
            else:
                raise ValueError(f"不支持的栅格数据格式: {type(indicator_data)}")

    def _evaluate_formula_with_weights(self, formula_str: str, variables_data: Dict[str, Any],
                                       formula_config: Dict[str, Any]) -> Any:
        """评估公式，支持权重系数"""
        # 获取权重配置
        weights = formula_config.get("weight", {})

        # 准备变量映射，包含权重
        local_env = {'np': np}
        local_env.update(variables_data)
        local_env.update(weights)  # 添加权重到计算环境

        try:
            # 执行公式计算
            result = eval(formula_str, {"__builtins__": {}}, local_env)
            return result

        except Exception as e:
            print(f"公式评估失败: {formula_str}")
            print(f"可用变量: {list(variables_data.keys())}")
            print(f"可用权重: {list(weights.keys())}")
            raise ValueError(f"公式计算错误: {str(e)}")

    def _evaluate_formula(self, formula_str: str, variables_data: Dict[str, Any]) -> Any:
        """评估公式（无权重版本）"""
        local_env = {'np': np}
        local_env.update(variables_data)

        try:
            result = eval(formula_str, {"__builtins__": {}}, local_env)
            return result
        except Exception as e:
            print(f"公式评估失败: {formula_str}")
            raise ValueError(f"公式计算错误: {str(e)}")

    def dataNormal_array(self, data: Union[float, np.ndarray], nodata: float = np.nan) -> Union[float, np.ndarray]:
        """统一的归一化处理，同时支持单值和数组"""
        if isinstance(data, (int, float)):
            # 单值归一化
            if np.isnan(data) or data == nodata:
                return nodata
            return data  # 单值归一化逻辑可根据需要调整

        # 数组归一化
        outdata = np.full_like(data, nodata, dtype=float)
        valid_mask = (data != nodata) & (~np.isnan(data))

        if np.any(valid_mask):
            valid_data = data[valid_mask]
            max_value = np.nanmax(valid_data)
            min_value = np.nanmin(valid_data)

            if max_value != min_value:
                outdata[valid_mask] = (valid_data - min_value) / (max_value - min_value)
            else:
                outdata[valid_mask] = 1.0  # 所有值相同的情况

        outdata[np.isnan(outdata)] = np.nan
        return outdata

    # 以下工具方法保持不变
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

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any],
                                  indicator_name: str, nodata: float = -32768) -> None:
        """保存中间结果"""
        try:
            print(f"保存中间结果: {indicator_name}")

            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                print(f"警告: 中间结果 {indicator_name} 格式不支持，跳过保存")
                return

            meta["nodata"] = nodata
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
        """使用GDAL保存为GeoTIFF文件"""
        try:
            from osgeo import gdal, osr

            if len(data.shape) == 1:
                if meta.get('width') and meta.get('height'):
                    data = data.reshape((meta['height'], meta['width']))
                else:
                    data = data.reshape((1, -1))
            elif len(data.shape) > 2:
                data = data.squeeze()

            height, width = data.shape

            # 数据类型映射
            dtype_map = {
                np.uint8: gdal.GDT_Byte,
                np.uint16: gdal.GDT_UInt16,
                np.int16: gdal.GDT_Int16,
                np.uint32: gdal.GDT_UInt32,
                np.int32: gdal.GDT_Int32,
                np.float32: gdal.GDT_Float32,
                np.float64: gdal.GDT_Float64,
            }
            datatype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                str(output_path),
                width,
                height,
                1,
                datatype,
                ['COMPRESS=LZW']
            )

            if dataset is None:
                raise ValueError(f"无法创建文件: {output_path}")

            transform = meta.get('transform')
            dataset.SetGeoTransform(transform)

            crs = meta.get('crs')
            if crs is not None:
                dataset.SetProjection(crs)

            band = dataset.GetRasterBand(1)
            band.WriteArray(data)

            nodata = meta.get('nodata')
            if nodata is not None:
                band.SetNoDataValue(float(nodata))

            dataset = None
            print(f"GeoTIFF文件保存成功: {output_path}")

        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _convert_excel_dates_to_string(self, df):
        """将Excel中的数字日期转换为中文日期字符串"""
        # 检查并转换开始日期列
        if '开始日期' in df.columns:
            df['开始日期'] = df['开始日期'].apply(self._excel_number_to_chinese_date)

        # 检查并转换结束日期列
        if '结束日期' in df.columns:
            df['结束日期'] = df['结束日期'].apply(self._excel_number_to_chinese_date)

        return df
