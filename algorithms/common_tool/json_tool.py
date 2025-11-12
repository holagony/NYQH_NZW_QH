#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
解析、输出json工具
@Version<1> 2021-03-17 Created by lyb
"""

import json

class JsonTool:

    @staticmethod
    def writeJson(jsonfile=None, info_dict=None):
        """
        将字典写进Json
        :param jsonfile: str, json文件
        :param info_dict: dict, 信息
        :return:
        """
        json_str = json.dumps(info_dict, indent=1, ensure_ascii=False)
        with open(jsonfile, 'w') as json_file:
            json_file.write(json_str)

    @staticmethod
    def readJson(jsonfile=None):
        """
        读取json文件
        :param jsonfile: str, json文件
        :return:
        """
        with open(jsonfile, 'r', encoding='utf-8')as fp:
            json_data = json.load(fp)
        return json_data

if __name__ == "__main__":
    jsonfile = r"D:\Project\交接\lyn244.json"
    info_dict = {"label": "haha", "data": 234, "score": [{"aa":1,"bb":33}]}
    print(info_dict)
    # info_dict["score"] = 223
    print(info_dict)
    obj = JsonTool()
    obj.writeJson(jsonfile, info_dict)
    print(1)