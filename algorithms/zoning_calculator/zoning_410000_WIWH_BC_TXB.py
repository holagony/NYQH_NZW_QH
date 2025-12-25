from pathlib import Path
import numpy as np
from typing import Dict, Any
import pandas as pd


# from algorithms.interpolation import InterpolateTool
# from algorithms.classification import ClassificationTool

from zoning_410000_WIWH_BC_CMB import WIWH_BC as WIWH_BC_CMB

class WIWH_BC(WIWH_BC_CMB):
    """兼容接口：支持 WIWH_BC() 直接调用"""
    
    def __call__(self, params):
        """让实例可调用：calc(params) == calc.calculate(params)"""
        return self.calculate(params)
    