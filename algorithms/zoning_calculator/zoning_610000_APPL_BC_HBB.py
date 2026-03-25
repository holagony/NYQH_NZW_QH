import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 把当前脚本目录加入搜索路径
from zoning_610000_APPL_CL_QXCL import APPL_CL

class APPL_BC(APPL_CL):
    """苹果褐斑病区划计算器 - 兼容接口"""
    
    def __init__(self):
        super().__init__()

    def __call__(self, params):
        """让实例可调用：calc(params) == calc.calculate(params)"""
        return self.calculate(params)