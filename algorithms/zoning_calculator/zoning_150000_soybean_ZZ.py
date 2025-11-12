import numpy as np
from typing import Dict, Any
from algorithms.interpolation import InterpolateTool
from algorithms.classification import ClassificationTool

class SoybeanPlantingZoning:
    """内蒙古大豆种植区划计算器"""
    
    @staticmethod
    def calculate(params: Dict[str, Any]) -> Dict[str, Any]:
        """执行种植区划计算"""
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        config = params['config']
        grid_path = config['gridFilePath']
        dem_path = config['demFilePath']
        algorithm_config = params['algorithmConfig']
        
        # 计算种植适宜性指数
        suitability_indicators = {}
        
        # for station_id, indicators in station_indicators.items():
        #     # 提取种植适宜性相关指标
        #     GDD = indicators.get('GDD', np.nan)  # 活动积温
        #     Precip = indicators.get('Precip_Total', np.nan)  # 降水总量
        #     Frost = indicators.get('Frost_Free_Days', np.nan)  # 无霜期
        #     Sunshine = indicators.get('Sunshine_Hours', np.nan)  # 日照时数
            
        #     # # 标准化处理
        #     # GDD_norm = SoybeanPlantingZoning._normalize_value(GDD, 1800, 2800)
        #     # Precip_norm = SoybeanPlantingZoning._normalize_value(Precip, 300, 600)
        #     # Frost_norm = SoybeanPlantingZoning._normalize_value(Frost, 120, 150)
        #     # Sunshine_norm = SoybeanPlantingZoning._normalize_value(Sunshine, 800, 1200)
            
        #     # 计算适宜性指数
        #     if not np.isnan(GDD_norm) and not np.isnan(Precip_norm) and \
        #        not np.isnan(Frost_norm) and not np.isnan(Sunshine_norm):
        #         suitability = (0.35 * GDD_norm + 0.25 * Precip_norm + 
        #                      0.20 * Frost_norm + 0.20 * Sunshine_norm)
        #         suitability_indicators[station_id] = suitability
        #     else:
        #         suitability_indicators[station_id] = np.nan
        
        # 插值
        interpolation_config = algorithm_config.get('interpolation', {})
        interpolation_method = interpolation_config.get('method', 'lsm_idw')
        
        interpolation_data = {
            'station_values': suitability_indicators,
            'station_coords': station_coords,
            'grid_path': grid_path,
            'dem_path': dem_path
        }
        
        if interpolation_method == 'idw':
            interpolated = InterpolateTool.idw_interpolation(interpolation_data, interpolation_config.get('params', {}))
        elif interpolation_method == 'lsm_idw':
            interpolated = InterpolateTool.lsm_idw_interpolation(interpolation_data, interpolation_config.get('params', {}))
        else:
            interpolated = InterpolateTool.idw_interpolation(interpolation_data, interpolation_config.get('params', {}))
        
        # 分级
        classification_config = algorithm_config.get('classification', {})
        classification_method = classification_config.get('method', 'custom_thresholds')
        
        if classification_method == 'custom_thresholds':
            classified_data = ClassificationTool.custom_thresholds(
                interpolated['data'], classification_config.get('params', {})
            )
        else:
            # 默认分级标准
            classification_params = {
                'thresholds': [
                    {"level": 1, "label": "不适宜区", "max": 0.3},
                    {"level": 2, "label": "次适宜区", "min": 0.3, "max": 0.5},
                    {"level": 3, "label": "适宜区", "min": 0.5, "max": 0.7},
                    {"level": 4, "label": "最适宜区", "min": 0.7}
                ]
            }
            classified_data = ClassificationTool.custom_thresholds(interpolated['data'], classification_params)
        
        return {
            'data': classified_data,
            'meta': interpolated['meta'],
            'type': 'planting_suitability'
        }
    
    @staticmethod
    def _normalize_value(value: float, min_val: float, max_val: float) -> float:
        """标准化数值到0-1范围"""
        if np.isnan(value):
            return np.nan
        
        # 确保在合理范围内
        value = max(min(value, max_val), min_val)
        
        # 线性归一化
        return (value - min_val) / (max_val - min_val)