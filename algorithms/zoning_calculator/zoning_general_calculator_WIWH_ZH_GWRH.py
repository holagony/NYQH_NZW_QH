import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 把当前脚本目录加入搜索路径

from zoning_general_calculator_ZH_GWRH import ZH_GWRH

class GenericZoningCalculator(ZH_GWRH):
    """兼容接口：支持 KORG_ZZ() 直接调用"""

    def __call__(self, params):
        """让实例可调用：calc(params) == calc.calculate(params)"""
        return self.calculate(params)