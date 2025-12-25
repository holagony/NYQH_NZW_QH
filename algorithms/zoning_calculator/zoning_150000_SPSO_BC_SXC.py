from pathlib import Path 
import numpy as np
from typing import Dict, Any
import importlib

class SPSO_BC:
    """内蒙古大豆病虫害区划计算器"""
    
    def __init__(self):
        pass
    
    def calculate(self, params):
        """执行区划计算"""
        # 获取输入数据
        config = params['config']
        self._algorithms = params['algorithms']
        
        # 根据病虫害类型选择计算方式
        # pest_type = config['element']
        
        return self._calculate_bean_moth(params)
        # if pest_type == 'JHB':
        #     return self._calculate_sclerotinia(params)
        # elif pest_type == 'SXC':
        #     return self._calculate_bean_moth(params)
        # else:
        #     raise ValueError(f"不支持的病虫害类型: {pest_type}")
    
    def _calculate_sclerotinia(self, params):
        """计算菌核病风险"""
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        # 提取菌核病相关指标
        sclerotinia_indicators = {}
        for station_id, indicators in station_indicators.items():
            if 'D' in indicators:  # 菌核病风险天数
                sclerotinia_indicators[station_id] = indicators['D']
            else:
                sclerotinia_indicators[station_id] = np.nan
        
        result = self._calculate_with_interpolation_before(sclerotinia_indicators, station_coords, config, algorithmConfig)

        # 分级
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{classification_method}")        
        data = classifier.execute(result['data'], classification) 
        result['data'] = data
        return result
    
    def _calculate_bean_moth(self, params):
        """计算食心虫风险 - 先计算站点综合风险指数再插值"""
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']
        
        print("开始计算大豆食心虫风险 - 新流程：先计算站点综合风险指数")
        
        # 第一步：在站点级别计算食心虫指标 X1~X4
        print("第一步：在站点级别计算食心虫指标 X1~X4")
        bean_moth_indicators = self._calculate_bean_moth_indicators_station(station_indicators)
        
        # 第二步：在站点级别计算食心虫综合风险指数F
        print("第二步：在站点级别计算食心虫综合风险指数F")
        bean_moth_risk_station = self._calculate_bean_moth_risk_station(bean_moth_indicators, algorithmConfig)
        
        # 第三步：对综合风险指数F进行插值
        print("第三步：对综合风险指数F进行插值")
        interpolated_risk = self._interpolate_bean_moth_risk(bean_moth_risk_station, station_coords, config, algorithmConfig)
        
        # 第四步：对插值结果进行分类
        print("第四步：对插值结果进行分类")
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'custom_thresholds')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        
        classified_data = classifier.execute(interpolated_risk['data'], classification)
        
        # 准备最终结果
        result = {
            'data': classified_data,
            'meta': interpolated_risk['meta'],
            'type': 'bean_moth_risk',
            'process': 'station_level_calculation'
        }
        
        print("大豆食心虫风险计算完成")
        return result
    
    def _calculate_bean_moth_indicators_station(self, station_indicators):
        """在站点级别计算食心虫指标 X1~X4"""
        bean_moth_indicators = {}
        
        for station_id, indicators in station_indicators.items():
            station_data = {}
            
            # 获取基础指标
            T1 = indicators.get('T1', np.nan)
            R1 = indicators.get('R1', np.nan)
            D1 = indicators.get('D1', np.nan)
            T2 = indicators.get('T2', np.nan)
            D2 = indicators.get('D2', np.nan)
            
            # 计算 X1 - 温度因子（需要T1和R1）
            station_data['X1'] = self._calculate_bean_moth_X1_station(T1, R1)
            
            # 计算 X2 - 降水因子（需要D1）
            station_data['X2'] = self._calculate_bean_moth_X2_station(D1)
            
            # 计算 X3 - 低温因子（需要T2）
            station_data['X3'] = self._calculate_bean_moth_X3_station(T2)
            
            # 计算 X4 - 温湿组合因子（需要D2）
            station_data['X4'] = self._calculate_bean_moth_X4_station(D2)
            
            bean_moth_indicators[station_id] = station_data
        
        # 统计各指标的有效站点数
        self._print_station_indicator_stats(bean_moth_indicators)
        
        return bean_moth_indicators
    
    def _calculate_bean_moth_X1_station(self, T1, R1):
        """在站点级别计算 X1 - 温度因子（基于T1和R1）"""
        """
        等级赋值规则：
        适宜(3): T1≥14且R1≤5
        较适宜(2): T1≥14
        不适宜(1): 以上2个条件都不满足
        """
        if np.isnan(T1) or np.isnan(R1):
            return np.nan
        
        # 应用等级赋值规则
        if T1 >= 14 and R1 <= 5:
            return 3  # 适宜
        elif T1 >= 14:
            return 2  # 较适宜
        else:
            return 1  # 不适宜
    
    def _calculate_bean_moth_X2_station(self, D1):
        """在站点级别计算 X2 - 降水因子（基于D1）"""
        """
        等级赋值规则：
        适宜(3): D1≤1
        较适宜(2): 1＜D1≤5
        不适宜(1): D1＞5
        """
        if np.isnan(D1):
            return np.nan
        
        # 应用等级赋值规则
        if D1 <= 1:
            return 3  # 适宜
        elif 1 < D1 <= 5:
            return 2  # 较适宜
        else:
            return 1  # 不适宜
    
    def _calculate_bean_moth_X3_station(self, T2):
        """在站点级别计算 X3 - 低温因子（基于T2）"""
        """
        等级赋值规则：
        适宜(3): T2≥17
        较适宜(2): 15≤T2＜17
        不适宜(1): T2＜15
        """
        if np.isnan(T2):
            return np.nan
        
        # 应用等级赋值规则
        if T2 >= 17:
            return 3  # 适宜
        elif 15 <= T2 < 17:
            return 2  # 较适宜
        else:
            return 1  # 不适宜
    
    def _calculate_bean_moth_X4_station(self, D2):
        """在站点级别计算 X4 - 温湿组合因子（基于D2）"""
        """
        等级赋值规则：
        适宜(3): D2≥15
        较适宜(2): 10≤D2＜15
        不适宜(1): D2＜10
        """
        if np.isnan(D2):
            return np.nan
        
        # 应用等级赋值规则
        if D2 >= 15:
            return 3  # 适宜
        elif 10 <= D2 < 15:
            return 2  # 较适宜
        else:
            return 1  # 不适宜
    
    def _calculate_bean_moth_risk_station(self, bean_moth_indicators, crop_config):
        """在站点级别计算食心虫综合风险指数F"""
        # 获取配置中的公式
        formula_config = crop_config.get("formula", {})
        if not formula_config:
            raise ValueError("未找到公式配置")
        
        formula_type = formula_config.get("type", "")
        formula_str = formula_config.get("formula", "")
        
        if formula_type != "custom_formula" or not formula_str:
            raise ValueError("不支持的公式类型或公式为空")
        
        print(f"使用公式计算站点综合风险指数: {formula_str}")
        
        # 计算每个站点的综合风险指数F
        bean_moth_risk = {}
        valid_count = 0
        
        for station_id, indicators in bean_moth_indicators.items():
            X1 = indicators.get('X1', np.nan)
            X2 = indicators.get('X2', np.nan)
            X3 = indicators.get('X3', np.nan)
            X4 = indicators.get('X4', np.nan)
            
            # 检查所有X指标是否有效
            if not np.isnan(X1) and not np.isnan(X2) and not np.isnan(X3) and not np.isnan(X4):
                # 使用公式计算综合风险指数F
                try:
                    F = self._evaluate_formula_station(
                        formula_str, 
                        {
                            'X1': X1,
                            'X2': X2,
                            'X3': X3,
                            'X4': X4
                        }
                    )
                    bean_moth_risk[station_id] = F
                    valid_count += 1
                except Exception as e:
                    print(f"站点 {station_id} 综合风险指数计算失败: {str(e)}")
                    bean_moth_risk[station_id] = np.nan
            else:
                bean_moth_risk[station_id] = np.nan
        
        print(f"站点综合风险指数计算完成，有效站点数: {valid_count}/{len(bean_moth_indicators)}")
        
        # 统计综合风险指数的范围
        if valid_count > 0:
            values = [v for v in bean_moth_risk.values() if not np.isnan(v)]
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values) / len(values)
            print(f"综合风险指数范围: [{min_val:.4f}, {max_val:.4f}], 均值: {mean_val:.4f}")
        
        return bean_moth_risk
    
    def _evaluate_formula_station(self, formula_str: str, variables: Dict[str, float]) -> float:
        """在站点级别评估公式 - 支持标量数据计算"""
        try:
            # 创建局部环境，包含数学函数和变量
            local_env = {
                'sqrt': np.sqrt,
                'log': np.log,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'arcsin': np.arcsin,
                'arccos': np.arccos,
                'arctan': np.arctan,
                'abs': abs,
                'min': min,
                'max': max
            }
            
            # 添加变量到环境
            local_env.update(variables)
            
            # 执行公式计算
            result = eval(formula_str, {"__builtins__": {}}, local_env)
            
            return float(result)
            
        except Exception as e:
            print(f"站点级别公式评估失败: {formula_str}")
            print(f"可用变量: {list(variables.keys())}")
            raise ValueError(f"公式计算错误: {str(e)}")
    
    def _interpolate_bean_moth_risk(self, bean_moth_risk_station, station_coords, config, crop_config):
        """对食心虫综合风险指数进行插值"""
        interpolation = crop_config.get("interpolation")
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        
        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")
        
        print(f"使用 {interpolation_method} 方法对综合风险指数进行插值")
        
        # 准备插值数据
        interpolation_data = {
            'station_values': bean_moth_risk_station,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")
        }
        
        # 执行插值
        try:
            interpolated_result = interpolator.execute(interpolation_data, interpolation_params)
            print("综合风险指数插值完成")
            
            # 保存中间结果
            self._save_intermediate_result(interpolated_result, config, "bean_moth_risk_interpolated")
            
            return interpolated_result
            
        except Exception as e:
            print(f"综合风险指数插值失败: {str(e)}")
            raise
    
    def _print_station_indicator_stats(self, bean_moth_indicators):
        """打印站点级别指标统计信息"""
        x1_values = []
        x2_values = []
        x3_values = []
        x4_values = []
        
        for station_id, indicators in bean_moth_indicators.items():
            x1 = indicators.get('X1', np.nan)
            x2 = indicators.get('X2', np.nan)
            x3 = indicators.get('X3', np.nan)
            x4 = indicators.get('X4', np.nan)
            
            if not np.isnan(x1):
                x1_values.append(x1)
            if not np.isnan(x2):
                x2_values.append(x2)
            if not np.isnan(x3):
                x3_values.append(x3)
            if not np.isnan(x4):
                x4_values.append(x4)
        
        print("站点级别指标统计:")
        if x1_values:
            print(f"  X1: 有效站点 {len(x1_values)}, 等级分布 - 适宜(3):{x1_values.count(3)}, 较适宜(2):{x1_values.count(2)}, 不适宜(1):{x1_values.count(1)}")
        if x2_values:
            print(f"  X2: 有效站点 {len(x2_values)}, 等级分布 - 适宜(3):{x2_values.count(3)}, 较适宜(2):{x2_values.count(2)}, 不适宜(1):{x2_values.count(1)}")
        if x3_values:
            print(f"  X3: 有效站点 {len(x3_values)}, 等级分布 - 适宜(3):{x3_values.count(3)}, 较适宜(2):{x3_values.count(2)}, 不适宜(1):{x3_values.count(1)}")
        if x4_values:
            print(f"  X4: 有效站点 {len(x4_values)}, 等级分布 - 适宜(3):{x4_values.count(3)}, 较适宜(2):{x4_values.count(2)}, 不适宜(1):{x4_values.count(1)}")
    
         
    def _calculate_with_interpolation_before(self, station_indicators, station_coords,
                                            config, crop_config):
        """先插值各指标，再计算区划指标 - 修改以支持食心虫计算"""
        print("使用 before 插值顺序: 先插值各指标，再计算区划指标")
        
        # 获取配置中的指标名称
        indicator_configs = crop_config.get("indicators", {})
        interpolation = crop_config.get("interpolation")
        
        if not indicator_configs:
            raise ValueError("未找到指标配置")
        
        print(f"需要插值的指标: {list(indicator_configs.keys())}")

        # 获取插值算法
        interpolation_method = interpolation.get('method','lsm_idw')
        interpolation_params = interpolation.get('params',{})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        
        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")
        
        # 对每个指标分别进行插值
        interpolated_indicators = {}
        
        for indicator_name in indicator_configs.keys():
            print(f"正在插值指标: {indicator_name}")
            
            # 提取该指标的站点数值
            indicator_values = {}
            valid_count = 0
            
            for station_id, values in station_indicators.items():
                if isinstance(values, dict) and indicator_name in values:
                    # 处理字典格式的指标数据（如食心虫的X1-X4）
                    value = values[indicator_name]
                else:
                    # 处理直接数值格式
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
                # 准备插值所需的数据
                interpolation_data = {
                    'station_values': indicator_values,
                    'station_coords': station_coords,
                    'dem_path': config.get("demFilePath",""),
                    'shp_path': config.get("shpFilePath",""),
                    'grid_path': config.get("gridFilePath",""),
                    'area_code': config.get("areaCode","")
                }                
                # 执行插值
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
        successful_indicators = [name for name in required_indicators if name in interpolated_indicators and interpolated_indicators[name] is not None]
        
        if len(successful_indicators) == 0:
            raise ValueError("所有指标插值都失败了")
        
        print(f"成功插值的指标: {successful_indicators}")
        
        # 使用插值后的栅格数据计算区划指标
        composite_result = self.calculate_grid(interpolated_indicators, crop_config)
        return composite_result
    
    def calculate_grid(self, interpolated_indicators: Dict[str, Any],
                      crop_config: Dict[str, Any]) -> Any:
        """栅格级别的区划计算 - 支持食心虫综合指数计算"""
        print("执行栅格级别的区划计算")
        
        # 获取公式配置
        formula_config = crop_config.get("formula", {})
        if not formula_config:
            raise ValueError("未找到公式配置")
        
        # 处理简单引用配置
        if "ref" in formula_config:
            ref_name = formula_config["ref"]
            if ref_name in interpolated_indicators:
                result = interpolated_indicators[ref_name]
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
   
    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], 
                                indicator_name: str) -> None:
        """保存中间结果 - 各个指标的插值结果"""
        try:
            print(f"保存中间结果: {indicator_name}")
            
            # 生成中间结果文件名
            file_name = indicator_name+".tif"
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


    # def _save_geotiff(self, data: np.ndarray, meta: Dict, output_path: str,nodata=-32767):
    #     """保存GeoTIFF文件"""
    #     from osgeo import gdal
        
    #     driver = gdal.GetDriverByName('GTiff')
    #     dataset = driver.Create(
    #         output_path,
    #         meta['width'],
    #         meta['height'],
    #         1,
    #         gdal.GDT_Float32,
    #         ['COMPRESS=LZW']
    #     )
        
    #     dataset.SetGeoTransform(meta['transform'])
    #     dataset.SetProjection(meta['crs'])
        
    #     band = dataset.GetRasterBand(1)
    #     band.WriteArray(data)
    #     band.SetNoDataValue(nodata)
        
    #     band.FlushCache()
    #     dataset = None
        
    #     self.fjson.log(f"GeoTIFF文件已保存: {output_path}")
    
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
          
    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]