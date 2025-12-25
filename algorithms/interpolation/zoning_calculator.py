import numpy as np
from typing import Dict, Any
from algorithms.interpolation import InterpolateTool
from algorithms.classification import ClassificationTool
from algorithms.indicators import IndicatorTool

class GenericZoningCalculator:
    """通用区划计算器 - 支持完全参数化配置"""
    
    @staticmethod
    def calculate(params: Dict[str, Any]) -> Dict[str, Any]:
        """执行通用区划计算"""
        # 获取算法配置
        algorithm_config = params.get('algorithmConfig', {})
        
        # 获取输入数据
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        grid_path = params['grid_path']
        dem_path = params.get('dem_path')
        
        # 获取插值配置
        interpolation_config = algorithm_config.get('interpolation', {})
        interpolation_method = interpolation_config.get('method', 'idw')
        interpolation_params = interpolation_config.get('params', {})
        
        # 获取公式配置
        formula_config = algorithm_config.get('formula', {})
        
        # 获取分类配置
        classification_config = algorithm_config.get('classification', {})
        classification_method = classification_config.get('method', 'equal_interval')
        classification_params = classification_config.get('params', {})
        
        # 计算综合指标
        composite_indicators = GenericZoningCalculator._calculate_composite_indicators(
            station_indicators, formula_config
        )
        
        # 插值
        interpolation_data = {
            'station_values': composite_indicators,
            'station_coords': station_coords,
            'grid_path': grid_path,
            'dem_path': dem_path
        }
        
        # 选择插值方法
        if interpolation_method == 'idw':
            interpolated = InterpolateTool.idw_interpolation(interpolation_data, interpolation_params)
        elif interpolation_method == 'lsm_idw':
            interpolated = InterpolateTool.lsm_idw_interpolation(interpolation_data, interpolation_params)
        elif interpolation_method == 'kriging':
            interpolated = InterpolateTool.kriging_interpolation(interpolation_data, interpolation_params)
        else:
            interpolated = InterpolateTool.idw_interpolation(interpolation_data, interpolation_params)
        
        # 分级
        classified_data = GenericZoningCalculator._classify_data(
            interpolated['data'], classification_method, classification_params
        )
        
        return {
            'data': classified_data,
            'meta': interpolated['meta'],
            'type': formula_config.get('description', 'composite_index'),
            'classification_method': classification_method,
            'formula_used': formula_config.get('expression', '')
        }
    
    @staticmethod
    def _calculate_composite_indicators(station_indicators: Dict[str, Any], 
                                      formula_config: Dict[str, Any]) -> Dict[str, float]:
        """计算综合指标"""
        formula_type = formula_config.get('type', 'linear_combination')
        
        if formula_type == 'linear_combination':
            return GenericZoningCalculator._linear_combination(station_indicators, formula_config)
        elif formula_type == 'custom_formula':
            return GenericZoningCalculator._custom_formula(station_indicators, formula_config)
        else:
            raise ValueError(f"不支持的公式类型: {formula_type}")
    
    @staticmethod
    def _linear_combination(station_indicators: Dict[str, Any], 
                          formula_config: Dict[str, Any]) -> Dict[str, float]:
        """线性组合公式"""
        expression = formula_config['expression']
        coefficients = formula_config['coefficients']
        variables_config = formula_config['variables']
        
        composite_indicators = {}
        
        for station_id, indicators in station_indicators.items():
            # 准备变量字典
            var_dict = coefficients.copy()  # 先加入系数
            
            for var_name, var_config in variables_config.items():
                if 'ref' in var_config:
                    ref_name = var_config['ref']
                    if ref_name in indicators:
                        var_dict[var_name] = indicators[ref_name]
                    else:
                        var_dict[var_name] = np.nan
                else:
                    var_dict[var_name] = var_config.get('value', 0)
            
            # 计算表达式
            try:
                # 安全地计算表达式
                composite_value = GenericZoningCalculator._safe_eval(expression, var_dict)
                composite_indicators[station_id] = composite_value
            except Exception as e:
                print(f"站点 {station_id} 公式计算失败: {str(e)}")
                composite_indicators[station_id] = np.nan
        
        return composite_indicators
    
    @staticmethod
    def _custom_formula(station_indicators: Dict[str, Any], 
                       formula_config: Dict[str, Any]) -> Dict[str, float]:
        """自定义公式"""
        formula = formula_config['expression']
        variables_config = formula_config.get('variables', {})
        
        composite_indicators = {}
        
        for station_id, indicators in station_indicators.items():
            # 准备变量字典
            var_dict = {}
            for var_name, var_config in variables_config.items():
                if 'ref' in var_config:
                    ref_name = var_config['ref']
                    if ref_name in indicators:
                        var_dict[var_name] = indicators[ref_name]
                    else:
                        var_dict[var_name] = np.nan
                else:
                    var_dict[var_name] = var_config.get('value', 0)
            
            # 替换公式中的变量
            custom_formula = formula
            for var_name, var_value in var_dict.items():
                custom_formula = custom_formula.replace(var_name, str(var_value))
            
            # 计算表达式
            try:
                composite_value = eval(custom_formula)
                composite_indicators[station_id] = composite_value
            except Exception as e:
                print(f"站点 {station_id} 自定义公式计算失败: {str(e)}")
                composite_indicators[station_id] = np.nan
        
        return composite_indicators
    
    @staticmethod
    def _safe_eval(expression: str, variables: Dict[str, float]) -> float:
        """安全地计算数学表达式"""
        # 限制可用的函数和操作
        allowed_names = {
            'abs': abs, 'min': min, 'max': max, 'round': round,
            'sum': sum, 'pow': pow, 'sqrt': np.sqrt, 'log': np.log,
            'exp': np.exp, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'pi': np.pi, 'e': np.e
        }
        
        # 添加变量
        allowed_names.update(variables)
        
        # 编译表达式
        code = compile(expression, '<string>', 'eval')
        
        # 验证名称
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of {name} not allowed")
        
        return eval(code, {'__builtins__': {}}, allowed_names)
    
    @staticmethod
    def _classify_data(data: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
        """数据分级"""
        if method == 'equal_interval':
            return ClassificationTool.equal_interval(data, params)
        elif method == 'natural_breaks':
            return ClassificationTool.natural_breaks(data, params)
        elif method == 'custom_thresholds':
            return ClassificationTool.custom_thresholds(data, params)
        else:
            return ClassificationTool.equal_interval(data, params)