from pathlib import Path
import numpy as np
from typing import Dict, Any


# from algorithms.interpolation import InterpolateTool
# from algorithms.classification import ClassificationTool

class WIWH_PZ:
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

        if pest_type in ['DBZHL','SMJHL','MTWDSJ','ZLRZ'] :
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


    def _calculate_before_interpolate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先根据站点计算数值，再插值"""
        print("执行 after 插值顺序: 先根据站点计算数值，再插值")

        # 在站点级别计算综合指标
        station_composite_values = {}
        valid_station_count = 0

        # print(station_indicators)
        for station_id, station_values in station_indicators.items():
            try:
                # 存储某个站的多个指标，与calculate_grid接口保持一致
                station_indicators_dict = {}
                for indicator_name in crop_config.get("indicators", {}).keys():

                    if isinstance(station_values, dict) and indicator_name in station_values:
                        # 处理字典格式的指标数据（如食心虫的X1-X4）
                        station_indicators_dict[indicator_name] = station_values[indicator_name]
                    else:
                        # 处理直接数值格式
                        station_indicators_dict[indicator_name] = station_values

            except Exception as e:
                print(f"站点 {station_id} 计算异常: {str(e)}")
                station_composite_values[station_id] = np.nan
        # print(station_composite_values)
        # breakpoint()

        # 在站点级别计算综合指标
        composite_value = self.calculate_grid(station_indicators_dict, crop_config)

        if composite_value and 'data' in composite_value and len(composite_value['data']) > 0:
            station_composite_values[station_id] = composite_value['data'][0]
            valid_station_count += 1
        else:
            station_composite_values[station_id] = np.nan
            print(f"站点 {station_id} 计算失败，结果为NaN")

        print(f"成功计算综合指标的站点数: {valid_station_count}/{len(station_indicators)}")

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

            composite_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合指标插值完成")

            # 保存中间结果
            self._save_intermediate_result(composite_result, config, "composite_indicator")

            return composite_result

        except Exception as e:
            print(f"综合指标插值失败: {str(e)}")
            raise

    def calculate_grid(self, indicators: Dict[str, Any],
                       crop_config: Dict[str, Any]) -> Any:
        """栅格级别的区划计算 - 支持站点级别和栅格级别计算"""
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

        # 处理自定义公式
        formula_type = formula_config.get("type", "")
        formula_str = formula_config.get("formula", "")
        variables_config = formula_config.get("variables", {})

        print(f"使用公式类型: {formula_type}")
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

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
        """使用GDAL保存为GeoTIFF文件"""
        try:
            from osgeo import gdal, osr

            # 确保数据是2D的
            if len(data.shape) == 1:
                # 如果是1D数据，需要知道宽度和高度才能重塑
                if meta.get('width') and meta.get('height'):
                    data = data.reshape((meta['height'], meta['width']))
                else:
                    # 如果不知道形状，创建为1行N列
                    data = data.reshape((1, -1))
            elif len(data.shape) > 2:
                data = data.squeeze()  # 移除单维度

            # 获取数据形状
            height, width = data.shape

            # 根据输入数据的 dtype 确定 GDAL 数据类型
            if data.dtype == np.uint8:
                datatype = gdal.GDT_Byte
            elif data.dtype == np.uint16:
                datatype = gdal.GDT_UInt16
            elif data.dtype == np.int16:
                datatype = gdal.GDT_Int16
            elif data.dtype == np.uint32:
                datatype = gdal.GDT_UInt32
            elif data.dtype == np.int32:
                datatype = gdal.GDT_Int32
            elif data.dtype == np.float32:
                datatype = gdal.GDT_Float32
            elif data.dtype == np.float64:
                datatype = gdal.GDT_Float64
            else:
                datatype = gdal.GDT_Float32  # 默认情况

            # 创建GeoTIFF文件
            driver = gdal.GetDriverByName('GTiff')

            # 创建数据集
            dataset = driver.Create(
                str(output_path),
                width,
                height,
                1,  # 波段数
                datatype,
                ['COMPRESS=LZW']  # 使用LZW压缩
            )

            if dataset is None:
                raise ValueError(f"无法创建文件: {output_path}")

            # 设置地理变换参数
            transform = meta.get('transform')
            dataset.SetGeoTransform(transform)

            # 设置投影
            crs = meta.get('crs')
            if crs is not None:
                dataset.SetProjection(crs)
            else:
                print("警告: 没有坐标参考系统信息")

            # 获取波段并写入数据
            band = dataset.GetRasterBand(1)
            band.WriteArray(data)

            # 设置无数据值
            nodata = meta.get('nodata')
            if nodata is not None:
                band.SetNoDataValue(float(nodata))

            # 关闭数据集，确保数据写入磁盘
            dataset = None

            print(f"GeoTIFF文件保存成功: {output_path}")

        except ImportError as e:
            print(f"导入GDAL失败: {str(e)}")
        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise

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







