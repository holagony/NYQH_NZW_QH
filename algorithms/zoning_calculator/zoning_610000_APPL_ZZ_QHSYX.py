import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from osgeo import gdal

class APPL_ZZ:
    """苹果种植气候适宜性区划计算器 - 陕西苹果"""

    def __init__(self):
        pass

    def __call__(self, params):
        return self.calculate(params)

    def calculate(self, params):
        """执行种植气候适宜性区划计算"""
        config = params['config']
        self._algorithms = params['algorithms']

        return self._calculate_element(params)

    def _calculate_element(self, params):
        """
        计算苹果种植气候适宜性区划
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print(f'开始计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-流程')

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

        # 读取地形因子并叠加地形适宜性
        # result = self._apply_terrain_suitability(result, config, algorithmConfig)

        print(f'计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-区划完成')
        return result

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
        """保存综合指标统计信息，区分0值和缺测值"""
        try:
            # 缺测值掩码
            nodata_mask = np.isnan(data)
            nodata_count = np.sum(nodata_mask)
            
            # 有效值掩码（包括0）
            valid_mask = ~nodata_mask
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
                    'nodata_count': int(nodata_count),
                    'total_count': int(total_count),
                    'valid_ratio': float(valid_count / total_count),
                    'zero_count': int(np.sum(valid_data == 0)),
                    'one_count': int(np.sum(valid_data == 1))
                }
                
                # 保存为JSON文件
                import json
                stats_path = output_dir / f"{name}_stats.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                
                print(f"综合指标统计: 范围[{stats['min']:.4f}, {stats['max']:.4f}], "
                    f"均值{stats['mean']:.4f}, 有效{valid_count}/{total_count}, "
                    f"0值{stats['zero_count']}, 1值{stats['one_count']}")
            else:
                print(f"综合指标统计: 全部为缺测值，有效0/{total_count}")
                    
        except Exception as e:
            print(f"保存统计信息失败: {str(e)}")

    def _calculate_with_interpolation(self, station_indicators, station_coords,
                                             config, crop_config):
        """计算种植气候适宜性区划指标"""

        # 获取插值配置和顺序控制
        interpolation_config = crop_config.get("interpolation", {})
        interpolation_order = interpolation_config.get('order', 'before')
        interpolation_method = interpolation_config.get('method', 'lsm_idw')
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
        else:
            raise ValueError(f"不支持的插值顺序: {interpolation_order}")

    def _interpolate_before_calculate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先插值各指标，再计算适宜度并加权求和"""
        print("执行 before 插值顺序: 先插值各指标，再计算适宜度并加权求和")

        indicator_configs = crop_config.get("indicators", {})
        interpolated_indicators = {}

        # 将站点数据转换为DataFrame
        station_df = self._convert_station_data_to_dataframe(station_indicators, indicator_configs)
        print(f"站点数据DataFrame形状: {station_df.shape}")

        # 只处理X1-X5气候因子
        climate_indicators = [f"X{i}" for i in range(1, 6)]
        
        for indicator_name in climate_indicators:
            if indicator_name not in station_df.columns:
                print(f"警告: 指标 {indicator_name} 不在站点数据中，跳过")
                continue
                
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
        successful_indicators = [name for name in climate_indicators if
                                 name in interpolated_indicators and interpolated_indicators[name] is not None]

        if len(successful_indicators) == 0:
            raise ValueError("所有气候指标插值都失败了")

        print(f"成功插值气候指标: {successful_indicators}")

        # 对每个气候指标计算适宜度
        suitability_indicators = self._calculate_climate_suitability(interpolated_indicators, crop_config, config)
        
        # 计算气候适宜性综合评分
        composite_result = self._calculate_climate_composite(suitability_indicators, crop_config, config)
        return composite_result

    def _convert_station_data_to_dataframe(self, station_indicators: Dict, indicator_configs: Dict) -> pd.DataFrame:
        """将站点数据转换为DataFrame"""
        data_dict = {}
        
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

    def _calculate_climate_suitability(self, indicators: Dict[str, Any], crop_config: Dict[str, Any], 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """计算各气候指标的适宜度，并保存结果"""
        print("开始计算气候指标适宜度...")
        
        suitability_configs = crop_config.get("suitability_functions", {})
        suitability_indicators = {}
        
        # 创建适宜度输出目录
        suitability_dir = Path(config["resultPath"]) / "suitability"
        suitability_dir.mkdir(parents=True, exist_ok=True)
        
        for indicator_name, indicator_data in indicators.items():
            if indicator_data is None:
                continue
                
            data = indicator_data['data']
            suitability_config = suitability_configs.get(indicator_name, {})
            
            if not suitability_config:
                print(f"警告: 指标 {indicator_name} 没有适宜度函数配置，跳过")
                continue
            
            # 对数据进行适宜度计算
            suitability_data = self._apply_suitability_function(data, suitability_config)
            
            # 统计适宜度结果
            valid_mask = ~np.isnan(suitability_data)
            valid_count = np.sum(valid_mask)
            if valid_count > 0:
                valid_data = suitability_data[valid_mask]
                min_val = np.nanmin(valid_data)
                max_val = np.nanmax(valid_data)
                mean_val = np.nanmean(valid_data)
                print(f"指标 {indicator_name} 适宜度: 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
            
            # 保存适宜度结果
            suitability_result = {
                'data': suitability_data,
                'meta': indicator_data['meta']
            }
            
            # 保存为GeoTIFF文件
            output_path = suitability_dir / f"{indicator_name}_suitability.tif"
            self._save_geotiff_gdal(suitability_data, indicator_data['meta'], output_path)
            print(f"保存 {indicator_name} 适宜度结果: {output_path}")
            
            # 保存适宜度统计信息
            self._save_suitability_stats(suitability_data, suitability_dir, f"{indicator_name}_suitability")
            
            suitability_indicators[indicator_name] = {
                'data': suitability_data,
                'meta': indicator_data['meta']
            }
            
        return suitability_indicators

    def _save_suitability_stats(self, data: np.ndarray, output_dir: Path, name: str):
        """保存适宜度统计信息"""
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
                
        except Exception as e:
            print(f"保存适宜度统计信息失败: {str(e)}")

    def _apply_suitability_function(self, data: np.ndarray, suitability_config: Dict[str, Any]) -> np.ndarray:
        """应用分段函数计算适宜度，0值为有效值，缺测值为nan"""
        # 创建输出数组，初始化为nan（缺测值）
        suitability_data = np.full_like(data, np.nan, dtype=float)
        
        # 获取有效数据掩码（非nan的数据）
        valid_mask = ~np.isnan(data)
        
        if not np.any(valid_mask):
            print("警告: 输入数据全部为nan")
            return suitability_data
        
        # 获取分段函数配置
        segments = suitability_config.get("segments", [])
        
        if not segments:
            print("警告: 没有分段函数配置")
            return suitability_data
        
        # 首先处理所有明确的范围条件
        for segment in segments:
            # 获取分段条件
            condition = segment.get("condition", "")
            if not condition:
                continue
            
            # 获取计算公式
            formula = segment.get("formula", "")
            if not formula:
                continue
                
            # 解析条件
            mask = None
            
            try:
                if "<=" in condition and "<" not in condition.split("<=")[0]:
                    # 简单条件：x<=value
                    parts = condition.split("<=")
                    if len(parts) == 2:
                        var = parts[0].strip()
                        if var == "x":
                            threshold = float(parts[1].strip())
                            mask = data <= threshold
                elif "<" in condition and "<=" not in condition and "=" not in condition.split("<")[0]:
                    # 简单条件：x<value
                    parts = condition.split("<")
                    if len(parts) == 2:
                        var = parts[0].strip()
                        if var == "x":
                            threshold = float(parts[1].strip())
                            mask = data < threshold
                elif ">" in condition and "=" not in condition.split(">")[0]:
                    # 简单条件：x>value
                    parts = condition.split(">")
                    if len(parts) == 2:
                        var = parts[1].strip()
                        if var == "x":
                            threshold = float(parts[0].strip())
                            mask = data > threshold
                elif ">=" in condition:
                    # 简单条件：x>=value
                    parts = condition.split(">=")
                    if len(parts) == 2:
                        var = parts[1].strip()
                        if var == "x":
                            threshold = float(parts[0].strip())
                            mask = data >= threshold
                elif "<x<=" in condition or "<x<" in condition or "<=x<" in condition or "<=x<=" in condition:
                    # 区间条件：min < x <= max 等
                    import re
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', condition)
                    if len(numbers) >= 2:
                        left_num = float(numbers[0])
                        right_num = float(numbers[1])
                        
                        if "<x<=" in condition:
                            mask = (data > left_num) & (data <= right_num)
                        elif "<x<" in condition:
                            mask = (data > left_num) & (data < right_num)
                        elif "<=x<" in condition:
                            mask = (data >= left_num) & (data < right_num)
                        elif "<=x<=" in condition:
                            mask = (data >= left_num) & (data <= right_num)
            except Exception as e:
                print(f"解析条件 {condition} 时出错: {str(e)}")
                continue
            
            if mask is None:
                print(f"警告: 无法解析条件: {condition}")
                continue
            
            # 计算适宜度值
            try:
                if formula == "0":
                    # 0是有效值，不是缺测值
                    suitability_data[mask] = 0.0
                elif formula == "1":
                    suitability_data[mask] = 1.0
                elif "x" in formula:
                    # 解析包含x的公式
                    x_vals = data[mask]
                    if np.any(~np.isnan(x_vals)):
                        # 替换公式中的x，并计算
                        formula_expr = formula.replace("x", "x_vals")
                        result = eval(formula_expr)
                        # 确保结果为数组
                        if np.isscalar(result):
                            result = np.full_like(x_vals, result)
                        suitability_data[mask] = result
                    else:
                        print(f"警告: 在条件 {condition} 下没有有效数据")
                else:
                    # 尝试直接计算常数公式
                    constant_value = eval(formula)
                    suitability_data[mask] = constant_value
                    
            except Exception as e:
                print(f"计算公式 {formula} 时出错: {str(e)}")
                continue
        
        # 确保适宜度在0-1之间，但保持nan不变
        valid_suitability_mask = ~np.isnan(suitability_data)
        if np.any(valid_suitability_mask):
            suitability_data[valid_suitability_mask] = np.clip(
                suitability_data[valid_suitability_mask], 0.0, 1.0
            )
        
        # 统计结果
        valid_count = np.sum(valid_suitability_mask)
        total_count = data.size
        zero_count = np.sum(suitability_data[valid_suitability_mask] == 0)
        one_count = np.sum(suitability_data[valid_suitability_mask] == 1)
        
        print(f"适宜度计算结果: 有效{valid_count}/{total_count}, 0值{zero_count}, 1值{one_count}")
        
        return suitability_data

    def _calculate_climate_composite(self, suitability_indicators: Dict[str, Any],
                                    crop_config: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """计算气候适宜性综合评分，正确处理0值和缺测值"""
        print("计算气候适宜性综合评分")
        
        # 获取公式配置
        formula_config = crop_config.get("formula", {})
        formula_str = formula_config.get("formula", "")
        variables_config = formula_config.get("variables", {})
        
        if not formula_str:
            raise ValueError("未找到公式配置")
        
        # 准备变量数据
        variables_data = {}
        variable_shapes = {}
        
        for var_name, var_config in variables_config.items():
            print(f"处理变量: {var_name}")
            ref_name = var_config.get("ref", "")
            if ref_name in suitability_indicators:
                var_data = suitability_indicators[ref_name]['data']
                variables_data[var_name] = var_data
                variable_shapes[var_name] = var_data.shape
            else:
                print(f"警告: 变量 {var_name} 引用的指标 {ref_name} 不存在")
                continue
        
        # 检查所有变量形状是否一致
        shapes = list(variable_shapes.values())
        if len(set(shapes)) > 1:
            print(f"警告: 变量形状不一致: {variable_shapes}")
        
        # 使用第一个变量的形状创建结果数组
        first_var = list(variables_data.keys())[0]
        result_shape = variables_data[first_var].shape
        result_data = np.full(result_shape, np.nan, dtype=float)
        
        # 计算公式
        try:
            # 找到所有变量都有效的区域
            valid_mask = np.ones(result_shape, dtype=bool)
            
            for var_name, var_data in variables_data.items():
                # 注意：0是有效值，只有nan是缺测值
                var_valid_mask = ~np.isnan(var_data)
                valid_mask = valid_mask & var_valid_mask
            
            valid_count = np.sum(valid_mask)
            print(f"综合评分有效区域: {valid_count}/{result_data.size}")
            
            if valid_count > 0:
                # 为每个变量提取有效区域的数据
                local_env = {'np': np}
                
                for var_name, var_data in variables_data.items():
                    # 只提取有效区域的数据
                    local_env[var_name] = var_data[valid_mask]
                
                # 执行公式计算
                composite_values = eval(formula_str, {"__builtins__": {}}, local_env)
                
                # 确保结果为数组
                if np.isscalar(composite_values):
                    composite_values = np.full(valid_count, composite_values)
                
                # 将结果放回对应位置
                result_data[valid_mask] = composite_values
                
                # 计算统计信息
                valid_result_data = result_data[valid_mask]
                min_val = np.nanmin(valid_result_data)
                max_val = np.nanmax(valid_result_data)
                mean_val = np.nanmean(valid_result_data)
                zero_count = np.sum(valid_result_data == 0)
                
                print(f"气候适宜性综合评分统计: 范围[{min_val:.4f}, {max_val:.4f}], "
                    f"均值{mean_val:.4f}, 有效{valid_count}/{result_data.size}, 0值{zero_count}")
                
                # 检查是否有负值或大于1的值（不应该出现，但以防万一）
                if np.any(valid_result_data < 0) or np.any(valid_result_data > 1):
                    print(f"警告: 综合评分超出[0,1]范围: 最小值{min_val}, 最大值{max_val}")
                    # 强制限制在0-1之间
                    result_data[valid_mask] = np.clip(composite_values, 0.0, 1.0)
            else:
                print("警告: 没有有效区域计算综合评分")
                
        except Exception as e:
            print(f"气候适宜性综合评分计算失败: {str(e)}")
            raise
        
        # 保存综合评分结果
        composite_dir = Path(config["resultPath"]) / "composite_score"
        composite_dir.mkdir(parents=True, exist_ok=True)
        
        first_indicator = next(iter(suitability_indicators.values()))
        composite_result = {
            'data': result_data,
            'meta': first_indicator['meta']
        }
        
        # 保存综合评分为GeoTIFF
        composite_path = composite_dir / "climate_composite_score.tif"
        self._save_geotiff_gdal(result_data, first_indicator['meta'], composite_path)
        print(f"保存气候适宜性综合评分: {composite_path}")
        
        # 保存综合评分统计信息
        self._save_composite_stats(result_data, composite_dir, "climate_composite_score")
        
        return composite_result

    def _apply_terrain_suitability(self, climate_result: Dict[str, Any],
                                config: Dict[str, Any],
                                algorithmConfig: Dict[str, Any]) -> Dict[str, Any]:
        """应用地形适宜性叠加，正确处理0值和缺测值"""
        print("开始应用地形适宜性叠加...")
        
        # 获取地形因子文件路径
        slope_file_path = config.get("slopeFilePath", "")
        aspect_file_path = config.get("aspectFilePath", "")
        
        if not slope_file_path or not aspect_file_path:
            print("警告: 未找到地形因子文件路径，跳过地形适宜性叠加")
            return climate_result
        
        try:
            # 读取坡度数据
            print(f"读取坡度数据: {slope_file_path}")
            slope_ds = gdal.Open(slope_file_path)
            if slope_ds is None:
                print(f"错误: 无法打开坡度文件 {slope_file_path}")
                return climate_result
                
            slope_data = slope_ds.ReadAsArray()
            slope_band = slope_ds.GetRasterBand(1)
            slope_nodata = slope_band.GetNoDataValue()
            
            # 读取坡向数据
            print(f"读取坡向数据: {aspect_file_path}")
            aspect_ds = gdal.Open(aspect_file_path)
            if aspect_ds is None:
                print(f"错误: 无法打开坡向文件 {aspect_file_path}")
                slope_ds = None
                return climate_result
                
            aspect_data = aspect_ds.ReadAsArray()
            aspect_band = aspect_ds.GetRasterBand(1)
            aspect_nodata = aspect_band.GetNoDataValue()
            
            # 确保地形数据与气候数据尺寸一致
            climate_data = climate_result['data']
            if slope_data.shape != climate_data.shape:
                print(f"警告: 坡度数据形状 {slope_data.shape} 与气候数据形状 {climate_data.shape} 不匹配，跳过地形叠加")
                return climate_result
            
            # 计算坡度适宜度（0为不适宜，1为适宜）
            slope_suitability = np.where(slope_data < 35, 1, 0)
            if slope_nodata is not None:
                # 将原始nodata区域设为nan
                slope_suitability[slope_data == slope_nodata] = np.nan
            
            # 计算坡向适宜度
            # 条件: 坡向在112.5到292.5度之间或等于-1（平地）为1，否则为0
            aspect_suitability = np.where(
                ((aspect_data > 112.5) & (aspect_data <= 292.5)) | (aspect_data == -1),
                1, 0
            )
            if aspect_nodata is not None:
                # 将原始nodata区域设为nan
                aspect_suitability[aspect_data == aspect_nodata] = np.nan
            
            # 计算综合地形适宜度（两者都适宜才为1，否则为0）
            # 注意：只有两个因子都有效时才计算，任一为nan则结果为nan
            terrain_suitability = np.full_like(slope_suitability, np.nan, dtype=float)
            
            # 找到两个地形因子都有效的区域
            slope_valid = ~np.isnan(slope_suitability)
            aspect_valid = ~np.isnan(aspect_suitability)
            terrain_valid = slope_valid & aspect_valid
            
            if np.any(terrain_valid):
                # 在有效区域计算地形适宜度
                terrain_suitability[terrain_valid] = np.where(
                    (slope_suitability[terrain_valid] == 1) & (aspect_suitability[terrain_valid] == 1),
                    1, 0
                )
            
            # 统计地形适宜度
            valid_mask = ~np.isnan(terrain_suitability)
            valid_count = np.sum(valid_mask)
            if valid_count > 0:
                terrain_suitable = np.sum(terrain_suitability[valid_mask] == 1)
                terrain_unsuitable = np.sum(terrain_suitability[valid_mask] == 0)
                print(f"地形适宜度统计: 适宜区域 {terrain_suitable}/{valid_count}, 不适宜区域 {terrain_unsuitable}/{valid_count}")
            
            # 叠加地形适宜性：在地形不适宜的区域，将气候适宜性等级设为1（不适宜）
            result_data = climate_data.copy()
            
            # 找到地形不适宜的区域（地形适宜度为0）
            terrain_unsuitable_mask = (terrain_suitability == 0) & valid_mask
            
            # 在地形不适宜的区域，将气候适宜性等级设为1
            result_data[terrain_unsuitable_mask] = 1.0  # 1代表不适宜区
            
            # 统计叠加结果
            climate_suitable_count = np.sum((climate_data > 1) & valid_mask)
            final_suitable_count = np.sum((result_data > 1) & valid_mask)
            print(f"地形叠加后统计: 原适宜区域 {climate_suitable_count}, 地形叠加后适宜区域 {final_suitable_count}")
            
            # 保存地形叠加后的结果
            final_dir = Path(config["resultPath"]) / "final_result"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / "final_suitability_with_terrain.tif"
            self._save_geotiff_gdal(result_data, climate_result['meta'], final_path)
            print(f"保存地形叠加后的最终结果: {final_path}")
            
            # 清理资源
            slope_ds = None
            aspect_ds = None
            
            # 更新结果
            climate_result['data'] = result_data
            
            print("地形适宜性叠加完成")
            return climate_result
            
        except Exception as e:
            print(f"地形适宜性叠加失败: {str(e)}")
            return climate_result

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
        """使用GDAL保存为GeoTIFF文件，正确处理0值和缺测值"""
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

            # 数据类型映射 - 使用Float32以保留0值和nan
            datatype = gdal.GDT_Float32

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
            
            # 写入数据
            band.WriteArray(data)
            
            # 设置nodata值为nan
            band.SetNoDataValue(float('nan'))
            
            # 设置有效值范围（0-1）
            # band.SetMinimum(0.0)
            # band.SetMaximum(1.0)

            dataset = None
            print(f"GeoTIFF文件保存成功: {output_path}")

        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise
    
    # def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
    #     """使用GDAL保存为GeoTIFF文件"""
    #     try:
    #         from osgeo import gdal, osr

    #         if len(data.shape) == 1:
    #             if meta.get('width') and meta.get('height'):
    #                 data = data.reshape((meta['height'], meta['width']))
    #             else:
    #                 data = data.reshape((1, -1))
    #         elif len(data.shape) > 2:
    #             data = data.squeeze()

    #         height, width = data.shape

    #         # 数据类型映射
    #         dtype_map = {
    #             np.uint8: gdal.GDT_Byte,
    #             np.uint16: gdal.GDT_UInt16,
    #             np.int16: gdal.GDT_Int16,
    #             np.uint32: gdal.GDT_UInt32,
    #             np.int32: gdal.GDT_Int32,
    #             np.float32: gdal.GDT_Float32,
    #             np.float64: gdal.GDT_Float64,
    #         }
    #         datatype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

    #         driver = gdal.GetDriverByName('GTiff')
    #         dataset = driver.Create(
    #             str(output_path),
    #             width,
    #             height,
    #             1,
    #             datatype,
    #             ['COMPRESS=LZW']
    #         )

    #         if dataset is None:
    #             raise ValueError(f"无法创建文件: {output_path}")

    #         transform = meta.get('transform')
    #         dataset.SetGeoTransform(transform)

    #         crs = meta.get('crs')
    #         if crs is not None:
    #             dataset.SetProjection(crs)

    #         band = dataset.GetRasterBand(1)
    #         band.WriteArray(data)

    #         nodata = meta.get('nodata')
    #         if nodata is not None:
    #             band.SetNoDataValue(float(nodata))

    #         dataset = None
    #         print(f"GeoTIFF文件保存成功: {output_path}")

    #     except Exception as e:
    #         print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
    #         raise

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]