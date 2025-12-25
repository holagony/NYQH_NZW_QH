#!/usr/bin/env python
# @Time: 2020/08/13 10:50
# @Author : baizhaofeng
# @Email  : zfengbai@gmail.com
# @File   : api.py

import os
import json
import datetime


class ResultJson:
    def __init__(self, filename, taskid):
        """
        :param out_jsonfile: 输出的json文件地址
        """
        self.filename = filename
        self.execute_info = {"status": "", "message": "", "taskId": taskid,
                             "productionTime": (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime(
                                 "%Y-%m-%d %H:%M:%S CST"),
                             "result": []}

    def info(self, mark, infor):
        """
        结果状态json文件输出信息
        :param mark:
        :param infor:
        """
        if mark == "status":
            self.execute_info['status'] = infor[0]
            self.execute_info['message'] = infor[1]
        elif mark == "result":
            tmprecord = {"filePath": infor[0], "productIdENG": infor[1], "productIdCHN": infor[2],
                         "productLevel": infor[3], "productFormat": infor[4]}
            self.execute_info["result"].append(tmprecord)
        elif mark == "result_value":
            self.execute_info.update({"result_value":infor[0]})
        elif mark == "meta":
            self.execute_info["satellite"] = infor[0]
            self.execute_info["sensor"] = infor[1]
        elif mark == "coord":  # 传参顺序：[分辨率,minLon,minLat,maxLon,maxLat]
            self.execute_info["resolution"] = infor[0]  # 单位：m
            self.execute_info["bottomLeftLongitude"] = infor[1]  # 单位：°
            self.execute_info["topLeftLongitude"] = infor[1]  # 单位：°
            self.execute_info["bottomLeftLatitude"] = infor[2]  # 单位：°
            self.execute_info["bottomRightLatitude"] = infor[2]  # 单位：°
            self.execute_info["topRightLongitude"] = infor[3]  # 单位：°
            self.execute_info["bottomRightLongitude"] = infor[3]  # 单位：°
            self.execute_info["topLeftLatitude"] = infor[4]  # 单位：°
            self.execute_info["topRightLatitude"] = infor[4]  # 单位：°
        elif mark == "time":
            self.execute_info["obsStartTime"] = infor[0]  # 格式："%Y-%m-%d %H:%M:%S CST"
            self.execute_info["obsEndTime"] = infor[1]  # 格式："%Y-%m-%d %H:%M:%S CST"
        else:
            self.execute_info[mark] = infor

    def write(self):
        fileDir = os.path.dirname(self.filename)
        if not os.path.isdir(fileDir):
            os.makedirs(fileDir, exist_ok=True)

        with open(self.filename, "w", encoding='utf-8') as f:
            json.dump(self.execute_info, f, ensure_ascii=False)


class FlowJson:
    cnt_num_times = 0

    def __init__(self, product_name, taskid, keysteps, out_jsonfile, logfile):
        """
        流程json文件初始化
        :param product_name: 算法名称
        :param keysteps: 算法关键执行步骤
        :param out_jsonfile: 输出json文件
        :param logfile: 输出log文件
        """
        self.filename = out_jsonfile
        self.logfilename = logfile
        self.execute_info = {"product": product_name, "taskId": taskid, "step": []}
        for i, keystep in enumerate(keysteps):
            keystepstr = {"stepName": keystep, "stepNo": str(i), "status": "255", "log": "", "timeStamp": ""}
            self.execute_info['step'].append(keystepstr)

        fileDir = os.path.dirname(out_jsonfile)
        if not os.path.isdir(fileDir):
            os.makedirs(fileDir, exist_ok=True)
        with open(out_jsonfile, "w", encoding='utf-8') as f:
            json.dump(self.execute_info, f, ensure_ascii=False)

    def log(self, info):
        infos = "%s: %s \n" % (
            (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S CST"), info)
        print(infos)
        with open(self.logfilename, 'a+', encoding='utf-8') as f:
            f.write(infos)

    def info(self, info, codeId='0'):
        """
        输出流程及log日志
        :param info:
        """
        FlowJson.cnt_num_times += 1

        with open(self.filename, "r+", encoding="utf-8") as f:
            item = json.load(f)
            content = item["step"]
            stepnum = self.cnt_num_times - 1

            content[stepnum]["status"] = codeId
            content[stepnum]["log"] = info
            content[stepnum]["timeStamp"] = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime(
                "%Y-%m-%d %H:%M:%S CST")
            self.log(info)
            f.seek(0)
            f.write(json.dumps(item, ensure_ascii=False))
            f.truncate()
