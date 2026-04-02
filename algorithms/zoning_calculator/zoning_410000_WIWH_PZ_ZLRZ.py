from pathlib import Path
import numpy as np
from typing import Dict, Any
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 把当前脚本目录加入搜索路径

from zoning_410000_WIWH_PZ_DBZHL import WIWH_PZ as WIWH_PZ_DBZHL

class WIWH_PZ(WIWH_PZ_DBZHL):
    """兼容接口：支持 WIWH_BC() 直接调用"""
    
    def __call__(self, params):
        """让实例可调用：calc(params) == calc.calculate(params)"""
        return self.calculate(params)
