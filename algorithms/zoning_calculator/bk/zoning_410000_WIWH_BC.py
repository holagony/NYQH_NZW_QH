from pathlib import Path
import numpy as np
from typing import Dict, Any
import pandas as pd


# from algorithms.interpolation import InterpolateTool
# from algorithms.classification import ClassificationTool

class WIWH_BC:
    """计算河南小麦品质气候区划计算器"""
    def __init__(self):
        pass

    def calculate(self, params): # 修改
        """执行区划计算"""
        # 获取输入数据
        config = params['config']

        self._algorithms = params['algorithms']
        # 根据品质气候类型选择计算方式
        pest_type = config['element']

        if pest_type in ['CMB','TXB'] :
            return self._calculate_element(params)
        else:
            raise ValueError(f"不支持的品质区划类型: {pest_type}")

    def _calculate_element(self, params):
        """
        计算品质气候区划-蛋白质气候区划
         - 先计算站点综合风险指数再插值
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print(f'开始计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-新流程')

        result = self._calculate_with_interpolation_before(station_indicators, station_coords, config, algorithmConfig)

        # 保存分级前的综合指标
        composite_index_result = {
            'data': result['data'].copy(),
            'meta': result['meta'].copy()
        }
        self._save_composite_index(composite_index_result, config, "composite_index")

        # 分级
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'equal_interval')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(result['data'], classification)
        result['data'] = data

        print(f'计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-区划完成')

        return result

    def _calculate_with_interpolation_before(self, station_indicators, station_coords,
                                             config, crop_config):
        """支持两种插值顺序：先插值各指标再计算，或先计算再插值"""

        # 获取插值配置和顺序控制
        interpolation_config = crop_config.get("interpolation", {})
        interpolation_order = interpolation_config.get('order', 'before')  # before: 先插值后计算, after: 先计算后插值
        interpolation_method = interpolation_config.get('method', 'idw')
        interpolation_params = interpolation_config.get('params', {})

        print(f"使用插值顺序: {interpolation_order}")

        # 获取指标配置
        indicator_configs = crop_config.get("indicators", {})
        if not indicator_configs:
            raise ValueError("未找到指标配置")

        print(f"需要处理的指标: {list(indicator_configs.keys())}")

        # 获取插值算法
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        if interpolation_order == 'before':
            # 先插值各指标，再计算区划指标
            return self._interpolate_before_calculate(station_indicators, station_coords,
                                                      config, crop_config, interpolator,
                                                      interpolation_params)
        elif interpolation_order == 'after':
            # 先根据站点计算数值，再插值
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

        for indicator_name in indicator_configs.keys():
            print(f"正在插值指标: {indicator_name}")

            # 提取该指标的站点数值
            indicator_values = {}
            valid_count = 0

            for station_id, values in station_indicators.items():
                if isinstance(values, dict) and indicator_name in values:
                    value = values[indicator_name]
                else:
                    value = values

                if isinstance(value, (int, float)) and not np.isnan(value):
                    indicator_values[station_id] = value
                    valid_count += 1
                else:
                    indicator_values[station_id] = np.nan

            print(f"指标 {indicator_name} 的有效站点数: {valid_count}/{len(station_indicators)}")

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

                interpolated_dict = interpolator.execute(interpolation_data, interpolation_params)
                interpolated_indicators[indicator_name] = interpolated_dict
                print(f"指标 {indicator_name} 插值完成")

                # 保存中间结果
                self._save_intermediate_result(interpolated_dict, config, indicator_name)

            except Exception as e:
                print(f"指标 {indicator_name} 插值失败: {str(e)}")
                interpolated_indicators[indicator_name] = None

        # 检查是否所有必要指标都成功插值
        required_indicators = list(indicator_configs.keys())
        successful_indicators = [name for name in required_indicators if
                                 name in interpolated_indicators and interpolated_indicators[name] is not None]

        if len(successful_indicators) == 0:
            raise ValueError("所有指标插值都失败了")

        print(f"成功插值的指标: {successful_indicators}")

        # 使用插值后的栅格数据计算区划指标
        composite_result = self.calculate_grid(interpolated_indicators, crop_config)
        return composite_result


    # def _calculate_before_interpolate(self, station_indicators, station_coords,
    #                                   config, crop_config, interpolator, interpolation_params):
    #     """先根据站点计算数值，再插值"""
    #     print("执行 after 插值顺序: 先根据站点计算数值，再插值")
    #
    #     # 在站点级别计算综合指标
    #     station_composite_values = {}
    #     valid_station_count = 0
    #
    #     # 获取公式配置
    #     formula_config = crop_config.get("formula", {})
    #     if not formula_config:
    #         raise ValueError("未找到公式配置")
    #
    #     indicator_names = list(crop_config.get("indicators", {}).keys())
    #
    #     # 创建计算器（自动计算数据范围）
    #     calculator = StandardizationCalculator(station_indicators, indicator_names)
    #
    #     # 计算所有站点的隶属度
    #     new_station_indicators = calculator.calculate_all_stations_normalization(formula_config)
    #
    #     for station_id, station_values in new_station_indicators.items():
    #         try:
    #             station_values_normalized = station_values
    #             # 存储某个站的多个指标，与calculate_grid接口保持一致
    #             station_indicators_dict = {}
    #             for indicator_name in crop_config.get("indicators", {}).keys():
    #
    #                 if isinstance(station_values_normalized, dict) and indicator_name in station_values_normalized:
    #                     # 处理字典格式的指标数据（如食心虫的X1-X4）
    #                     station_indicators_dict[indicator_name] = station_values_normalized[indicator_name]
    #                 else:
    #                     # 处理直接数值格式
    #                     station_indicators_dict[indicator_name] = station_values_normalized
    #
    #             # 在站点级别计算综合指标
    #             composite_value = self.calculate_grid(station_indicators_dict, crop_config)
    #
    #             if composite_value and 'data' in composite_value and len(composite_value['data']) > 0:
    #                 station_composite_values[station_id] = composite_value['data'][0]
    #                 valid_station_count += 1
    #             else:
    #                 station_composite_values[station_id] = np.nan
    #                 print(f"站点 {station_id} 计算失败，结果为NaN")
    #
    #         except Exception as e:
    #             print(f"站点 {station_id} 计算异常: {str(e)}")
    #             station_composite_values[station_id] = np.nan
    #
    #     print(f"成功计算综合指标的站点数: {valid_station_count}/{len(station_indicators)}")
    #
    #     if valid_station_count == 0:
    #         raise ValueError("所有站点计算综合指标都失败了")
    #
    #     # 对综合指标结果进行插值
    #     try:
    #         interpolation_data = {
    #             'station_values': station_composite_values,
    #             'station_coords': station_coords,
    #             'dem_path': config.get("demFilePath", ""),
    #             'shp_path': config.get("shpFilePath", ""),
    #             'grid_path': config.get("gridFilePath", ""),
    #             'area_code': config.get("areaCode", "")
    #         }
    #
    #         composite_result = interpolator.execute(interpolation_data, interpolation_params)
    #         print("综合指标插值完成")
    #
    #         # 保存中间结果
    #         self._save_intermediate_result(composite_result, config, "composite_indicator")
    #
    #         return composite_result
    #
    #     except Exception as e:
    #         print(f"综合指标插值失败: {str(e)}")
    #         raise

    # def calculate_grid(self, indicators: Dict[str, Any],
    #                    crop_config: Dict[str, Any]) -> Any:
    #     """栅格级别的区划计算 - 支持站点级别和栅格级别计算"""
    #     print("执行区划计算")
    #
    #     # 获取公式配置
    #     formula_config = crop_config.get("formula", {})
    #     if not formula_config:
    #         raise ValueError("未找到公式配置")
    #
    #     # 处理简单引用配置
    #     if "ref" in formula_config:
    #         ref_name = formula_config["ref"]
    #         if ref_name in indicators:
    #             result = indicators[ref_name]
    #             print(f"直接返回指标 {ref_name} 的结果")
    #             return result
    #         else:
    #             raise ValueError(f"引用的指标 {ref_name} 不存在")
    #
    #     # 处理自定义公式
    #     formula_type = formula_config.get("type", "")
    #     formula_str = formula_config.get("formula", "")
    #     variables_config = formula_config.get("variables", {})
    #
    #     print(f"使用公式类型: {formula_type}")
    #     print(f"公式: {formula_str}")
    #     print(f"变量配置: {list(variables_config.keys())}")
    #
    #     if formula_type != "custom_formula" or not formula_str:
    #         raise ValueError("不支持的公式类型或公式为空")
    #
    #     # 检测计算级别：站点级别还是栅格级别
    #     calculation_level = self._detect_calculation_level(indicators)
    #     print(f"检测到计算级别: {calculation_level}")
    #
    #     if calculation_level == "station":
    #         # 站点级别计算
    #         return self._calculate_station_level(indicators, formula_str, variables_config, crop_config)
    #     else:
    #         # 栅格级别计算
    #         return self._calculate_grid_level(indicators, formula_str, variables_config, crop_config)
    def calculate_grid(self, indicators: Dict[str, Any],
                       crop_config: Dict[str, Any],
                       region_name: str = None) -> Any:
        """栅格级别的区划计算 - 支持多区域配置"""
        print("执行区划计算")

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

        # 选择区域公式
        if region_name and f"formula_{region_name}" in formula_config:
            # 使用指定区域的公式
            region_config = formula_config[f"formula_{region_name}"]
            print(f"使用指定区域公式: {region_name}")
        elif "formula_north" in formula_config:
            # 默认使用北方公式
            region_config = formula_config["formula_north"]
            print("使用默认区域公式: north")
        elif "formula_south" in formula_config:
            # 默认使用南方公式
            region_config = formula_config["formula_south"]
            print("使用默认区域公式: south")
        else:
            # 单区域配置
            region_config = formula_config
            print("使用单区域公式配置")

        formula_type = region_config.get("type", "")
        formula_str = region_config.get("formula", "")
        variables_config = region_config.get("variables", {})

        print(f"公式类型: {formula_type}")
        print(f"公式: {formula_str}")
        print(f"变量配置: {list(variables_config.keys())}")

        if formula_type != "custom_formula" or not formula_str:
            raise ValueError("不支持的公式类型或公式为空")

        # 检测计算级别：站点级别还是栅格级别
        calculation_level = self._detect_calculation_level(indicators)
        print(f"检测到计算级别: {calculation_level}")

        if calculation_level == "station":
            # 站点级别计算
            return self._calculate_station_level(indicators, formula_str, variables_config, crop_config)
        else:
            # 栅格级别计算
            return self._calculate_grid_level(indicators, formula_str, variables_config, crop_config)

    def _detect_calculation_level(self, indicators: Dict[str, Any]) -> str:
        """检测计算级别"""
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict) and 'data' in indicator_data:
                data_content = indicator_data['data']
                if isinstance(data_content, np.ndarray):
                    # 如果是多维数组，说明是栅格数据
                    if data_content.ndim > 1 or data_content.size > 100:
                        return "grid"
            # 如果是单值或者小数组，可能是站点数据
            if isinstance(indicator_data, (int, float)) or (
                    isinstance(indicator_data, np.ndarray) and indicator_data.size == 1):
                return "station"

        # 默认使用栅格级别
        return "grid"

    def _calculate_station_level(self, indicators: Dict[str, Any],
                                 formula_str: str, variables_config: Dict[str, Any],
                                 crop_config: Dict[str, Any]) -> Any:
        """站点级别计算"""
        print("执行站点级别计算")

        # 准备站点数据
        station_data = {}

        # 收集所有站点的数据
        for var_name, var_config in variables_config.items():
            if "ref" in var_config:
                ref_name = var_config["ref"]
                if ref_name in indicators:
                    indicator_data = indicators[ref_name]

                    # 提取数据值
                    if isinstance(indicator_data, (int, float)):
                        station_data[var_name] = indicator_data
                    elif isinstance(indicator_data, dict) and 'data' in indicator_data:
                        data_content = indicator_data['data']
                        if isinstance(data_content, np.ndarray) and data_content.size == 1:
                            station_data[var_name] = data_content[0]
                        else:
                            station_data[var_name] = data_content
                    elif isinstance(indicator_data, np.ndarray) and indicator_data.size == 1:
                        station_data[var_name] = indicator_data[0]
                    else:
                        station_data[var_name] = np.nan
                else:
                    station_data[var_name] = np.nan

        # 计算公式
        try:
            # 替换公式中的变量名
            local_formula = formula_str
            for var_name, var_value in station_data.items():
                if isinstance(var_value, (int, float)) and not np.isnan(var_value):
                    local_formula = local_formula.replace(var_name, str(var_value))
                else:
                    local_formula = local_formula.replace(var_name, "np.nan")

            # 安全计算
            result_value = eval(local_formula, {"np": np, "__builtins__": {}}, {})

            # 包装结果
            result = {
                'data': np.array([result_value]),
                'meta': {
                    'calculation_level': 'station',
                    'station_count': 1,
                    'description': '站点级别计算结果'
                }
            }

            print(f"站点级别计算结果: {result_value:.4f}")
            return result

        except Exception as e:
            print(f"站点级别公式计算失败: {str(e)}")
            raise

    def _calculate_grid_level(self, interpolated_indicators: Dict[str, Any],
                              formula_str: str, variables_config: Dict[str, Any],
                              crop_config: Dict[str, Any]) -> Any:
        """栅格级别计算"""
        print("执行栅格级别计算")

        # 准备变量数据
        variables_data = {}

        for var_name, var_config in variables_config.items():
            print(f"处理变量: {var_name}")
            var_value = self._compute_variable(var_config, interpolated_indicators)
            variables_data[var_name] = var_value

        # 计算综合指数
        try:
            result_grid = self._evaluate_formula(formula_str, variables_data)
        except Exception as e:
            print(f"公式计算失败: {str(e)}")
            raise

        # 使用第一个指标的元数据
        first_indicator = next(iter(interpolated_indicators.values()))
        result = {
            'data': result_grid,
            'meta': first_indicator['meta']
        }

        # 打印最终结果统计
        if 'data' in result:
            data = result['data']
            valid_count = np.sum(~np.isnan(data))
            total_count = data.size
            if valid_count > 0:
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                mean_val = np.nanmean(data)
                print(f"最终结果: {valid_count}/{total_count} 有效像素, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")

        return result

    def _compute_variable(self, var_config: Dict[str, Any], interpolated_indicators: Dict[str, Any]) -> np.ndarray:
        """计算单个变量的值"""
        var_type = var_config.get("type", "")

        if var_type == "standardize":
            # 标准化处理
            value_config = var_config.get("value", {})
            if "ref" in value_config:
                ref_name = value_config["ref"]
                if ref_name in interpolated_indicators:
                    grid_data = interpolated_indicators[ref_name]['data']
                    return self._standardize_grid(grid_data)
                else:
                    raise ValueError(f"引用的指标不存在: {ref_name}")
            else:
                raise ValueError("标准化配置缺少引用")

        elif "ref" in var_config:
            # 直接引用指标
            ref_name = var_config["ref"]
            if ref_name in interpolated_indicators:
                return interpolated_indicators[ref_name]['data']
            else:
                raise ValueError(f"引用的指标不存在: {ref_name}")

        elif var_type == "custom_formula":
            # 自定义公式（递归计算）
            formula_str = var_config.get("formula", "")
            sub_variables_config = var_config.get("variables", {})

            sub_variables_data = {}
            for sub_var_name, sub_var_config in sub_variables_config.items():
                sub_variables_data[sub_var_name] = self._compute_variable(sub_var_config, interpolated_indicators)

            return self._evaluate_formula(formula_str, sub_variables_data)

        else:
            # 常量值
            value = var_config.get("value", 0)
            # 创建一个与输入栅格相同大小的常量栅格
            first_grid = next(iter(interpolated_indicators.values()))['data']
            return np.full_like(first_grid, value)

    def _standardize_grid(self, grid_data: np.ndarray) -> np.ndarray:
        """对栅格数据进行标准化"""
        valid_mask = ~np.isnan(grid_data)
        if not np.any(valid_mask):
            return np.full_like(grid_data, np.nan)

        valid_values = grid_data[valid_mask]
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        if max_val == min_val:
            return np.full_like(grid_data, 0.5)

        standardized = np.full_like(grid_data, np.nan)
        standardized[valid_mask] = (valid_values - min_val) / (max_val - min_val)
        return standardized

    def _evaluate_formula(self, formula_str: str, variables_data: Dict[str, np.ndarray]) -> np.ndarray:
        """评估公式"""
        # 准备变量映射
        var_mapping = {}
        for var_name, var_data in variables_data.items():
            # 使用简化的变量名，避免公式中的复杂替换
            var_mapping[var_name] = var_data

        # 使用安全的公式评估
        try:
            # 创建局部环境，包含numpy函数和变量
            local_env = {'np': np}
            local_env.update(var_mapping)

            # 执行公式计算
            result = eval(formula_str, {"__builtins__": {}}, local_env)
            return result

        except Exception as e:
            print(f"公式评估失败: {formula_str}")
            print(f"可用变量: {list(variables_data.keys())}")
            raise ValueError(f"公式计算错误: {str(e)}")

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any],
                                  indicator_name: str) -> None:
        """保存中间结果 - 各个指标的插值结果"""
        try:
            print(f"保存中间结果: {indicator_name}")

            # 生成中间结果文件名
            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            # print(f"中间结果保存路径: {output_path}")

            # 使用与最终结果相同的保存逻辑
            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                print(f"警告: 中间结果 {indicator_name} 格式不支持，跳过保存")
                return
            meta["nodata"] = -32768
            # 保存为GeoTIFF
            self._save_geotiff_gdal(data, meta, output_path)

            # # 保存中间结果信息
            # self._save_intermediate_info(params, indicator_name, output_path)

            # print(f"中间结果 {indicator_name} 保存完成")

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")
            # 不抛出异常，继续处理其他指标
    def _numpy_to_gdal_dtype(self, numpy_dtype: np.dtype) -> int:
        """将numpy数据类型转换为GDAL数据类型"""
        from osgeo import gdal

        dtype_map = {
            np.bool_: gdal.GDT_Byte,
            np.uint8: gdal.GDT_Byte,
            np.uint16: gdal.GDT_UInt16,
            np.int16: gdal.GDT_Int16,
            np.uint32: gdal.GDT_UInt32,
            np.int32: gdal.GDT_Int32,
            np.float32: gdal.GDT_Float32,
            np.float64: gdal.GDT_Float64,
            np.complex64: gdal.GDT_CFloat32,
            np.complex128: gdal.GDT_CFloat64
        }

        for np_type, gdal_type in dtype_map.items():
            if np.issubdtype(numpy_dtype, np_type):
                return gdal_type

        # 默认使用Float32
        print(f"警告: 无法映射numpy数据类型 {numpy_dtype}，默认使用GDT_Float32")
        return gdal.GDT_Float32

    def _calculate_before_interpolate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先根据站点计算数值，再插值"""
        print("执行 after 插值顺序: 先根据站点计算数值，再插值")

        # 在站点级别计算综合指标
        station_composite_values = {}
        valid_station_count = 0

        # 获取公式配置
        formula_config = crop_config.get("formula", {})
        if not formula_config:
            raise ValueError("未找到公式配置")

        indicator_names = list(crop_config.get("indicators", {}).keys())

        station_file=config.get('stationFilePath')
        station_info = self._load_station_info(station_file)

        # 创建计算器（自动计算数据范围）
        calculator = StandardizationCalculator(station_indicators, indicator_names)

        # 计算所有站点的隶属度
        new_station_indicators = calculator.calculate_all_stations_normalization(station_info,formula_config)

        # 保存中间归一化结果
        print("保存中间归一化结果")
        element = config['element']
        file_name = f"intermediate_{element}_normalization.csv"
        # 保存中间归一化结果
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name
        self.convert_nested_dict_to_csv(new_station_indicators,output_path)

        # for station_id, station_values in new_station_indicators.items():
        #     try:
        #         station_values_normalized = station_values
        #         # 存储某个站的多个指标，与calculate_grid接口保持一致
        #         station_indicators_dict = {}
        #         for indicator_name in crop_config.get("indicators", {}).keys():
        #
        #             if isinstance(station_values_normalized, dict) and indicator_name in station_values_normalized:
        #                 # 处理字典格式的指标数据（如食心虫的X1-X4）
        #                 station_indicators_dict[indicator_name] = station_values_normalized[indicator_name]
        #             else:
        #                 # 处理直接数值格式
        #                 station_indicators_dict[indicator_name] = station_values_normalized
        #
        #         # 在站点级别计算综合指标
        #         composite_value = self.calculate_grid(station_indicators_dict, crop_config)
        #
        #         if composite_value and 'data' in composite_value and len(composite_value['data']) > 0:
        #             station_composite_values[station_id] = composite_value['data'][0]
        #             valid_station_count += 1
        #         else:
        #             station_composite_values[station_id] = np.nan
        #             print(f"站点 {station_id} 计算失败，结果为NaN")
        #
        #     except Exception as e:
        #         print(f"站点 {station_id} 计算异常: {str(e)}")
        #         station_composite_values[station_id] = np.nan
        #
        # print(f"成功计算综合指标的站点数: {valid_station_count}/{len(station_indicators)}")
        # 修改后的代码

        # 判断是否为多区域格式

        # 判断是否为多区域格式
        if isinstance(new_station_indicators, dict) and any(
                key in ['north', 'south'] for key in new_station_indicators.keys()):
            # 多区域格式处理
            for region_name, region_stations in new_station_indicators.items():
                print(f"处理 {region_name} 区域，共 {len(region_stations)} 个站点")

                for station_id, station_values in region_stations.items():
                    try:
                        # 存储某个站的多个指标，与calculate_grid接口保持一致
                        station_indicators_dict = {}

                        # 遍历station_values中的指标（只包含该区域有的指标）
                        if isinstance(station_values, dict):
                            for indicator_name, indicator_value in station_values.items():
                                # 直接使用station_values中的指标和值
                                station_indicators_dict[indicator_name] = indicator_value
                        else:
                            # 如果不是字典格式，设置为NaN
                            station_indicators_dict = {indicator_name: np.nan for indicator_name in
                                                       crop_config.get("indicators", {}).keys()}
                            print(f"警告: 站点 {station_id} 数据格式异常")

                        # 在站点级别计算综合指标，传递区域信息
                        composite_value = self.calculate_grid(
                            station_indicators_dict,
                            crop_config,
                            region_name=region_name  # 传递区域名称
                        )

                        if composite_value and 'data' in composite_value and len(composite_value['data']) > 0:
                            station_composite_values[station_id] = composite_value['data'][0]
                            valid_station_count += 1
                        else:
                            station_composite_values[station_id] = np.nan
                            print(f"站点 {station_id} 计算失败，结果为NaN")

                    except Exception as e:
                        station_composite_values[station_id] = np.nan
                        print(f"站点 {station_id} 计算异常: {str(e)}")

        else:
            # 单区域格式处理（保持原有逻辑）
            for station_id, station_values in new_station_indicators.items():
                try:
                    station_values_normalized = station_values
                    # 存储某个站的多个指标，与calculate_grid接口保持一致
                    station_indicators_dict = {}

                    # 遍历station_values中的指标
                    if isinstance(station_values_normalized, dict):
                        for indicator_name, indicator_value in station_values_normalized.items():
                            station_indicators_dict[indicator_name] = indicator_value
                    else:
                        # 处理直接数值格式
                        for indicator_name in crop_config.get("indicators", {}).keys():
                            station_indicators_dict[indicator_name] = station_values_normalized

                    # 在站点级别计算综合指标
                    composite_value = self.calculate_grid(station_indicators_dict, crop_config)

                    if composite_value and 'data' in composite_value and len(composite_value['data']) > 0:
                        station_composite_values[station_id] = composite_value['data'][0]
                        valid_station_count += 1
                        print(f"站点 {station_id} 计算成功: {composite_value['data'][0]:.4f}")
                    else:
                        station_composite_values[station_id] = np.nan
                        print(f"站点 {station_id} 计算失败，结果为NaN")

                except Exception as e:
                    print(f"站点 {station_id} 计算异常: {str(e)}")
                    station_composite_values[station_id] = np.nan

        print(f"成功计算 {valid_station_count} 个站点的综合指标")

        if valid_station_count == 0:
            raise ValueError("所有站点计算综合指标都失败了")

        # 保存中间归一化结果
        print("保存综合指标计算结果-单站")
        element = config['element']
        file_name = f"{element}_station_composite_index.csv"
        intermediate_dir = Path(config["resultPath"]) / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_path = intermediate_dir / file_name
        self.convert_nested_dict_to_csv(station_composite_values, output_path)

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

            composite_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合指标插值完成")

            # 保存中间结果
            self._save_intermediate_result(composite_result, config, "composite_indicator")

            return composite_result

        except Exception as e:
            print(f"综合指标插值失败: {str(e)}")
            raise

    def _load_station_info(self,station_file):
        """加载站点信息文件"""
        _station_info_cache = {}
        try:
            # 读取站点stationFilePath信息文件（GBK编码）
            station_df = pd.read_csv(station_file, encoding='gbk')

            # 处理列名可能的空格问题
            station_df.columns = station_df.columns.str.strip()

            # 将站点信息缓存到字典中
            for _, row in station_df.iterrows():
                station_id = str(row['站号']).strip()
                _station_info_cache[station_id] = {
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

            print(f"成功加载 {len(_station_info_cache)} 个站点的信息")
            return _station_info_cache

        except Exception as e:
            print(f"加载站点信息文件失败: {str(e)}")

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

    # def convert_nested_dict_to_csv(self, data_dict,output_path):
    #     """将嵌套字典转换为适合CSV的格式并保存"""
    #
    #     # 将嵌套字典转换为适合CSV的格式
    #     rows = []
    #     for station_id, indicators in data_dict.items():
    #         row = {'station_id': station_id}
    #         row.update(indicators)  # 将内部字典的键值对添加到行中
    #         rows.append(row)
    #
    #     # 创建DataFrame并保存为CSV
    #     df = pd.DataFrame(rows)
    #
    #     # 设置station_id为索引（可选）
    #     df.set_index('station_id', inplace=True)
    #
    #     # 保存为CSV文件
    #     df.to_csv(output_path, encoding='utf-8-sig')
    #
    #     print(f"中间归一化结果已保存: {output_path}")
    #     print(f"保存了 {len(df)} 个站点的数据")
    #
    #     return output_path  # 返回保存的文件路径
    def convert_nested_dict_to_csv(self, data_dict, output_path):
        """将嵌套字典转换为适合CSV的格式并保存"""

        # 将嵌套字典转换为适合CSV的格式
        rows = []
        for station_id, indicators in data_dict.items():
            row = {'station_id': station_id}

            # 检查 indicators 的类型
            if isinstance(indicators, dict):
                # 如果 indicators 是字典，直接更新
                row.update(indicators)
            else:
                # 如果 indicators 是单个数值，将其作为单独的一列
                row['value'] = indicators

            rows.append(row)

        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(rows)

        # 设置station_id为索引（可选）
        df.set_index('station_id', inplace=True)

        # 保存为CSV文件
        df.to_csv(output_path, encoding='utf-8-sig')

        print(f"中间归一化结果已保存: {output_path}")
        print(f"保存了 {len(df)} 个站点的数据")

        return output_path  # 返回保存的文件路径

'''
原始数据 station_indicators
     ↓
计算全局极值范围 (1961-2023所有数据)
     ↓
计算各站点年平均值  
     ↓
分析公式系数 → 确定相关关系
     ↓
归一化计算 (基于全局极值和年平均值)
     ↓
输出 new_station_indicators
'''


class StandardizationCalculator:
    def __init__(self, station_indicators, indicator_names):
        """
        初始化标准化计算器

        Parameters:
        station_indicators: 站点数据字典
        indicator_names: 指标名称列表，如 ['Tmax', 'Tmin', 'Pred']
        """
        self.station_indicators = station_indicators
        self.indicator_names = indicator_names
        self.data_range = self._calculate_data_range_from_all_years()
        self._print_data_range_info()

    def _calculate_data_range_from_all_years(self):
        """
        根据所有站点所有年份的原始数据计算每个指标的最小最大值

        Returns:
        data_range: 包含每个指标min和max的字典
        """
        data_range = {}

        # 初始化每个指标的最小最大值
        for indicator_name in self.indicator_names:
            data_range[indicator_name] = {
                'min': float('inf'),
                'max': float('-inf')
            }

        # 遍历所有站点所有年份的原始数据
        for station_id, station_data in self.station_indicators.items():
            for indicator_name in self.indicator_names:
                if indicator_name in station_data:
                    for year, value in station_data[indicator_name].items():
                        # 跳过NaN值
                        if not np.isnan(value):
                            # 更新最小值
                            if value < data_range[indicator_name]['min']:
                                data_range[indicator_name]['min'] = value
                            # 更新最大值
                            if value > data_range[indicator_name]['max']:
                                data_range[indicator_name]['max'] = value

        # 处理没有有效数据的情况
        for indicator_name in self.indicator_names:
            if data_range[indicator_name]['min'] == float('inf'):
                data_range[indicator_name]['min'] = 0
                data_range[indicator_name]['max'] = 1
                print(f"警告: 指标 {indicator_name} 没有有效数据，使用默认范围 [0, 1]")
            elif data_range[indicator_name]['min'] == data_range[indicator_name]['max']:
                # 如果所有值都相同，添加一个小偏移量避免除零
                data_range[indicator_name]['min'] -= 0.1
                data_range[indicator_name]['max'] += 0.1
                print(f"警告: 指标 {indicator_name} 所有值相同，调整范围避免除零")

        return data_range

    def _calculate_annual_means(self):
        """
        计算每个站点每个指标的年平均值

        Returns:
        annual_means: 字典，格式为 {station_id: {indicator_name: mean_value}}
        """
        annual_means = {}

        for station_id, station_data in self.station_indicators.items():
            annual_means[station_id] = {}

            for indicator_name in self.indicator_names:
                if indicator_name in station_data:
                    # 收集所有年份的有效值
                    values = []
                    for year, value in station_data[indicator_name].items():
                        if not np.isnan(value):
                            values.append(value)

                    # 计算平均值
                    if values:
                        annual_means[station_id][indicator_name] = np.mean(values)
                    else:
                        annual_means[station_id][indicator_name] = np.nan
                else:
                    annual_means[station_id][indicator_name] = np.nan

        return annual_means

    def _print_data_range_info(self):
        """打印数据范围信息"""
        for indicator_name, range_info in self.data_range.items():
            # 统计总数据点数量
            total_points = 0
            valid_points = 0

            for station_id, station_data in self.station_indicators.items():
                if indicator_name in station_data:
                    for year, value in station_data[indicator_name].items():
                        total_points += 1
                        if not np.isnan(value):
                            valid_points += 1

            print(f"{indicator_name}:")
            print(f"  历史范围: [{range_info['min']:.4f}, {range_info['max']:.4f}]")
            print(f"  有效数据点: {valid_points}/{total_points} ({valid_points / max(total_points, 1) * 100:.1f}%)")
            print()

    def calculate_membership(self, X, ref_var, correlation_type="positive"):
        """
        计算隶属度值 U(X)

        Parameters:
        X: 输入值（年平均值）
        ref_var: 参考变量名
        correlation_type: 相关类型 "positive" 或 "negative"
        """
        if ref_var not in self.data_range:
            raise ValueError(f"未找到变量 {ref_var} 的数据范围")

        X_min = self.data_range[ref_var]['min']  # 1961-2023所有年份最小值
        X_max = self.data_range[ref_var]['max']  # 1961-2023所有年份最大值

        if X_max == X_min:
            return 0.5  # 避免除零错误

        # 处理NaN值
        if np.isnan(X):
            return np.nan

        if correlation_type == "positive":
            # 正相关公式: U(X) = (X - X_min) / (X_max - X_min)
            return (X - X_min) / (X_max - X_min)
        else:
            # 负相关公式: U(X) = (X_max - X) / (X_max - X_min)
            return (X_max - X) / (X_max - X_min)

    def analyze_correlation_from_formula(self, formula_config):
        """
        根据公式配置自动分析相关关系

        Parameters:
        formula_config: 公式配置字典

        Returns:
        correlation_config: 相关关系配置字典
        """
        formula_str = formula_config["formula"]
        variables_config = formula_config["variables"]

        # 清理公式字符串
        clean_formula = formula_str.replace(' ', '')

        correlation_config = {}

        for var_name, var_config in variables_config.items():
            indicator_name = var_config["ref"]

            if var_name in clean_formula:
                # 找到变量在公式中的位置
                var_index = clean_formula.find(var_name)

                if var_index == 0:
                    # 变量在开头，系数为1（正）
                    correlation_config[indicator_name] = "positive"
                else:
                    # 查找系数部分
                    coeff_part = clean_formula[:var_index]

                    # 从右向左查找运算符
                    last_operator_pos = -1
                    for i in range(len(coeff_part) - 1, -1, -1):
                        if coeff_part[i] in ['+', '-']:
                            last_operator_pos = i
                            break

                    if last_operator_pos >= 0:
                        coeff_str = coeff_part[last_operator_pos:]
                    else:
                        coeff_str = coeff_part

                    if coeff_str.startswith('-'):
                        correlation_config[indicator_name] = "negative"
                    else:
                        correlation_config[indicator_name] = "positive"
            else:
                # 变量不在公式中，默认正相关
                correlation_config[indicator_name] = "positive"

        print("自动分析的相关关系:", correlation_config)
        return correlation_config

    def _is_multi_region_formula(self,formula_config):
        """
        判断是否为多区域公式配置
        """
        if not isinstance(formula_config, dict):
            return False

        # 检查是否包含区域公式键（formula_north, formula_south等）
        region_keys = [key for key in formula_config.keys() if key.startswith('formula_')]

        # 如果有区域公式键，且数量>=2，则是多区域配置
        if len(region_keys) >= 1:
            return True

        # 如果没有区域公式键，但有type和formula字段，则是单区域配置
        if 'type' in formula_config and 'formula' in formula_config:
            return False

        # 其他情况默认为单区域
        return False

    # def calculate_all_stations_normalization(self, formula_config):
    #     """
    #     计算所有站点的结果，根据standardize字段决定是否进行归一化
    #
    #     Parameters:
    #     formula_config: 公式配置字典
    #
    #     Returns:
    #     new_station_indicators: 包含结果值的字典
    #     """
    #
    #     # 检查是否需要进行归一化
    #     standardize = formula_config.get("standardize", "True").lower() == "true"
    #
    #     # 计算年平均值
    #     annual_means = self._calculate_annual_means()
    #
    #     # 新的station_indicators结构
    #     new_station_indicators = {}
    #
    #     if standardize:
    #         # 需要归一化：自动分析相关关系并进行归一化
    #         correlation_config = self.analyze_correlation_from_formula(formula_config)
    #         print(f"进行归一化计算，相关关系: {correlation_config}")
    #     else:
    #         # 不需要归一化
    #         print("不进行归一化，直接返回年平均值")
    #
    #     for station_id, means in annual_means.items():
    #         # 初始化新结构
    #         new_station_indicators[station_id] = {}  # 只包含结果值
    #
    #         # 计算结果值
    #         if standardize:
    #             # 计算归一化值
    #             for indicator_name in self.indicator_names:
    #                 mean_value = means.get(indicator_name, np.nan)
    #
    #                 if not np.isnan(mean_value) and indicator_name in correlation_config:
    #                     correlation_type = correlation_config[indicator_name]
    #                     normalized_value = self.calculate_membership(
    #                         mean_value, indicator_name, correlation_type
    #                     )
    #                     new_station_indicators[station_id][indicator_name] = normalized_value
    #                 else:
    #                     new_station_indicators[station_id][indicator_name] = np.nan
    #         else:
    #             # 直接使用年平均值作为结果值
    #             for indicator_name in self.indicator_names:
    #                 mean_value = means.get(indicator_name, np.nan)
    #                 new_station_indicators[station_id][indicator_name] = mean_value
    #
    #         # # 统计有效值数量
    #         # valid_count = sum(1 for value in new_station_indicators[station_id].values()
    #         #                   if not np.isnan(value))
    #         # print(f"站点 {station_id} 完成，有效结果指标: {valid_count}/{len(self.indicator_names)}")
    #
    #     return new_station_indicators
    def calculate_all_stations_normalization(self, station_info, formula_config):
        """
        计算所有站点的结果，根据standardize字段决定是否进行归一化
        支持分区域计算

        Parameters:
        station_info: 站点信息字典
        formula_config: 公式配置字典

        Returns:
        new_station_indicators: 包含结果值的字典
        """
        # 检查是否需要进行归一化
        standardize = formula_config.get("standardize", "True").lower() == "true"

        # 检查是否有多个公式配置（分区域）- 使用新的判断逻辑
        is_multi_region = self._is_multi_region_formula(formula_config)

        if is_multi_region:
            print("检测到多区域公式配置，按地区分配计算")
            # 定义南方城市列表
            south_cities = ['南阳市', '驻马店市', '信阳市']

            # 分离站点到不同区域
            north_stations = {}
            south_stations = {}

            for station_id, info in station_info.items():
                city_name = info.get('city', '')

                # 判断是否属于南方城市
                if any(south_city in city_name for south_city in south_cities):
                    south_stations[station_id] = info
                    # print(f"站点 {station_id}({info.get('station_name', '')}) 属于南方区域")
                else:
                    north_stations[station_id] = info
                    # print(f"站点 {station_id}({info.get('station_name', '')}) 属于北方区域")

            # 计算所有站点的年平均值（南北统一计算）
            all_annual_means = self._calculate_annual_means()

            # 分别计算不同区域的归一化结果
            results = {}

            # 计算北方区域
            if north_stations and 'formula_north' in formula_config:
                print(f"\n计算北方区域，共 {len(north_stations)} 个站点")
                north_formula = formula_config['formula_north']
                north_results = {}

                if standardize:
                    correlation_config = self.analyze_correlation_from_formula(north_formula)
                    print(f"北方区域归一化，相关关系: {correlation_config}")

                for station_id in north_stations.keys():
                    if station_id not in all_annual_means:
                        continue

                    means = all_annual_means[station_id]
                    station_results = {}

                    # 计算结果值
                    if standardize:
                        # 计算归一化值
                        for var_name, var_config in north_formula["variables"].items():
                            indicator_name = var_config["ref"]
                            mean_value = means.get(indicator_name, np.nan)

                            if not np.isnan(mean_value) and indicator_name in correlation_config:
                                correlation_type = correlation_config[indicator_name]
                                normalized_value = self.calculate_membership(
                                    mean_value, indicator_name, correlation_type
                                )
                                station_results[indicator_name] = normalized_value
                            else:
                                station_results[indicator_name] = np.nan
                    else:
                        # 直接使用年平均值
                        for var_name, var_config in north_formula["variables"].items():
                            indicator_name = var_config["ref"]
                            mean_value = means.get(indicator_name, np.nan)
                            station_results[indicator_name] = mean_value

                    north_results[station_id] = station_results

                results['north'] = north_results

            # 计算南方区域
            if south_stations and 'formula_south' in formula_config:
                print(f"\n计算南方区域，共 {len(south_stations)} 个站点")
                south_formula = formula_config['formula_south']
                south_results = {}

                if standardize:
                    correlation_config = self.analyze_correlation_from_formula(south_formula)
                    print(f"南方区域归一化，相关关系: {correlation_config}")

                for station_id in south_stations.keys():
                    if station_id not in all_annual_means:
                        continue

                    means = all_annual_means[station_id]
                    station_results = {}

                    # 计算结果值
                    if standardize:
                        # 计算归一化值
                        for var_name, var_config in south_formula["variables"].items():
                            indicator_name = var_config["ref"]
                            mean_value = means.get(indicator_name, np.nan)

                            if not np.isnan(mean_value) and indicator_name in correlation_config:
                                correlation_type = correlation_config[indicator_name]
                                normalized_value = self.calculate_membership(
                                    mean_value, indicator_name, correlation_type
                                )
                                station_results[indicator_name] = normalized_value
                            else:
                                station_results[indicator_name] = np.nan
                    else:
                        # 直接使用年平均值
                        for var_name, var_config in south_formula["variables"].items():
                            indicator_name = var_config["ref"]
                            mean_value = means.get(indicator_name, np.nan)
                            station_results[indicator_name] = mean_value

                    south_results[station_id] = station_results

                results['south'] = south_results

            return results

        else:
            print("单一公式配置，进行全局计算")
            # 原有的单一公式处理逻辑
            standardize = formula_config.get("standardize", "True").lower() == "true"

            # 计算年平均值
            annual_means = self._calculate_annual_means()

            # 新的station_indicators结构
            new_station_indicators = {}

            if standardize:
                # 需要归一化：自动分析相关关系
                correlation_config = self.analyze_correlation_from_formula(formula_config)
                print(f"进行归一化计算，相关关系: {correlation_config}")
            else:
                print("不进行归一化，直接返回年平均值")

            for station_id, means in annual_means.items():
                # 初始化新结构
                new_station_indicators[station_id] = {}

                # 计算结果值
                if standardize:
                    # 计算归一化值
                    for indicator_name in self.indicator_names:
                        mean_value = means.get(indicator_name, np.nan)

                        if not np.isnan(mean_value) and indicator_name in correlation_config:
                            correlation_type = correlation_config[indicator_name]
                            normalized_value = self.calculate_membership(
                                mean_value, indicator_name, correlation_type
                            )
                            new_station_indicators[station_id][indicator_name] = normalized_value
                        else:
                            new_station_indicators[station_id][indicator_name] = np.nan
                else:
                    # 直接使用年平均值作为结果值
                    for indicator_name in self.indicator_names:
                        mean_value = means.get(indicator_name, np.nan)
                        new_station_indicators[station_id][indicator_name] = mean_value

            return new_station_indicators




