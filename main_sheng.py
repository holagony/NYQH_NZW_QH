#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
农作物区划主程序
"""
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from algorithms.common_tool.json_tool import JsonTool

from arspy import glo
from arspy.id import product, period, fmt
import warnings
warnings.filterwarnings('ignore')

from algorithms.zoning_processor_sheng import ZoningProcessor

def main(json_data):
    """
    执行主程序
    """
    rjson = glo.get_value("rjson")
    fjson = glo.get_value("fjson")
    
    try:
        # 创建处理器
        processor = ZoningProcessor(json_data, fjson, rjson)
        
        # 执行处理
        success = processor.process()
        
        if success:
            rjson.info('status', ['0', '区划执行成功!'])
            fjson.info("算法执行成功")
        else:
            rjson.info('status', ['1', '区划执行失败!'])
            
    except Exception as e:
        fjson.info(f"算法执行异常: {str(e)}", codeId='1')
        rjson.info('status', ['1', "算法执行失败"])
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # 默认配置文件路径
        json_path = "inputJson_WIWH_PZ.json"
    
    # 读取json文件
    obj = JsonTool()
    json_data = obj.readJson(json_path)
    
    main(json_data)