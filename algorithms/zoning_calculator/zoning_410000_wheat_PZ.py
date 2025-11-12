from pathlib import Path
import numpy as np
from typing import Dict, Any


# from algorithms.interpolation import InterpolateTool
# from algorithms.classification import ClassificationTool

class WheatQualityZoning:
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

        if pest_type == 'sclerotinia':
            return self._calculate_sclerotinia(params)
        elif pest_type == 'bean_moth':
            return self._calculate_bean_moth(params)
        else:
            raise ValueError(f"不支持的病虫害类型: {pest_type}")

    def _calculate_protein(self, params):
        """
        计算品质气候区划-蛋白质气候区划
         - 先计算站点综合风险指数再插值
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print("开始计算蛋白质气候区划 - 新流程：先计算站点综合风险指数")

        # 第一步：在站点级别计算食心虫指标 X1~X4
        print("第一步：在站点级别计算蛋白质气候区划指标 X1~X4")
        bean_moth_indicators = self._calculate_protein_indicators_station(station_indicators)

        # 第二步：在站点级别计算食心虫综合风险指数F
        print("第二步：在站点级别计算蛋白质气候区划指数F")
        bean_moth_risk_station = self._calculate_bean_moth_risk_station(bean_moth_indicators, algorithmConfig)

        # 第三步：对综合风险指数F进行插值
        print("第三步：对蛋白质气候区划指数F进行插值")
        interpolated_risk = self._interpolate_bean_moth_risk(bean_moth_risk_station, station_coords, config,
                                                             algorithmConfig)

        # 第四步：对插值结果进行分类
        print("第四步：对插值结果进行分类")
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'equal_interval')
        classifier = self._get_algorithm(f"classification.{classification_method}")

        classified_data = classifier.execute(interpolated_risk['data'], classification)

        # 准备最终结果
        result = {
            'data': classified_data,
            'meta': interpolated_risk['meta'],
            'type': 'protein_climate_zone',
            'process': 'station_level_calculation'
        }

        print("计算蛋白质气候区划完成")
        return result

    def _calculate_protein_indicators_station(self, station_indicators):
        """在站点级别蛋白质气候区划指标 X1~X4"""
        bean_moth_indicators = {}

        for station_id, indicators in station_indicators.items():
            station_data = {}

            # 获取基础指标
            Tmax = indicators.get('Tmax', np.nan)
            Tmin = indicators.get('Tmin', np.nan)
            Pred = indicators.get('Pred', np.nan)
            AT10 = indicators.get('AT10', np.nan)

            # 计算 X1 - 高温
            station_data['X1'] = Tmax

            # 计算 X2 - 降水日数
            station_data['X2'] = Pred

            # 计算 X3 - 低温
            station_data['X3'] = Tmin

            # 计算 X4 - 活动积温
            station_data['X4'] = AT10

            bean_moth_indicators[station_id] = station_data

        # 统计各指标的有效站点数
        self._print_station_indicator_stats(bean_moth_indicators)

        return bean_moth_indicators

    def _calculate_bean_moth_risk_station(self, bean_moth_indicators, crop_config):
        """在站点级别计算蛋白质气候区划指数F"""
        # 获取配置中的公式
        formula_config = crop_config.get("formula", {})
        if not formula_config:
            raise ValueError("未找到公式配置")

        formula_type = formula_config.get("type", "")
        formula_str = formula_config.get("formula", "")

        if formula_type != "custom_formula" or not formula_str:
            raise ValueError("不支持的公式类型或公式为空")

        print(f"使用公式计算站点综合风险指数: {formula_str}")

        # 计算每个站点的蛋白质气候区划指数F
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

        # 统计蛋白质气候区划指数的范围
        if valid_count > 0:
            values = [v for v in bean_moth_risk.values() if not np.isnan(v)]
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values) / len(values)
            print(f"蛋白质气候区划指数范围: [{min_val:.4f}, {max_val:.4f}], 均值: {mean_val:.4f}")

        return bean_moth_risk
