#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @date    : 2021/3/18 9:43
  @author  : baizhaofeng
  @email   : zfengbai@gmail.com
  @file    : app.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import traceback
from arspy.api import ResultJson, FlowJson
from arspy import glo
from config import algName, keySteps, labels
from main import main

def app(jsonpath):
    starttime = time.time()

    # 解析传入json文件
    with open(jsonpath, "rb") as fid:
        jsondict = json.load(fid)

    taskid = jsondict['taskId']
    rjson = ResultJson(jsondict["resultJsonPath"], taskid)  # 记录算法运行结果的json
    fjson = FlowJson(algName, taskid, keySteps, jsondict["resultFlowPath"], jsondict["resultLogPath"])  # 记录算法运行流程的json和日志

    # 设置为全局变量，方便在其他模块调用
    glo._init()  # 主模块初始化
    glo.set_value("rjson", rjson)
    glo.set_value("fjson", fjson)

    try:
        main(jsondict)

    except KeyError as ke:
        fjson.info(str(ke), codeId='1')
        rjson.info('status', ['2', "参数传入异常!"])
        sys.exit()

    except Exception as ex:
        print(traceback.format_exc())
        fjson.info(str(ex), codeId='1')
        rjson.info('status', ['1', "算法执行失败"])
    else:
        # 将状态写入json文件,0表示成功
        rjson.info('status', ['0', '%s 执行成功 !' % algName])

        endtime = time.time()
        costtime = endtime - starttime
        fjson.log("算法执行成功，总用时：%s秒" % (str(round(costtime, 2))))
    finally:
        rjson.write()


if __name__ == '__main__':
    # 传入json文件
    try:
        jsonPath = sys.argv[1]
        if len(sys.argv) > 1:
            jsonPath = sys.argv[1]
    except:
        jsonPath=r"C:\Users\mjynj\Desktop\NYQH\NYQH_NZW_QH\inputJson_soybean_ZH_cwdi.json"
    # 调用
    sys.exit(app(jsonPath))
