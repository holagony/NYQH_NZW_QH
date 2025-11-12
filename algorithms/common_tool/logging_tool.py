#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打印日志
@Version<1> 2021-02-28 Create by LYB
"""

import os
import time
import pytz
import logging
from datetime import datetime
from logging import handlers

class Logger():

    # 日志级别关系映射
    level_relations = {
                        'debug':logging.DEBUG,
                        'info':logging.INFO,
                        'warning':logging.WARNING,
                        'error':logging.ERROR,
                        'crit':logging.CRITICAL,
                       }

    def __init__(self, logfile=None):
        """
        :param logfile: str,日志文件路径
        """
        self.logfile = logfile

    def logmessage(self, message="", level='info', when='D', backCount=3):
        """
        打印日志
        :param message: str, 日志信息
        :param level: str, 日志级别
        :param when: str, 时间
        :param backCount: int, 参数
        :return:
        """
        timestr = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        if self.logfile is None:
            self.logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), timestr+".log")
        log_dir = os.path.dirname(self.logfile)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fmt = '%(asctime)s'+': %(message)s'
        uct_date = datetime.utcnow()
        cst_zo = pytz.timezone("Asia/Shanghai")
        self.logger = logging.getLogger(self.logfile)
        cst_date = cst_zo.fromutc(uct_date).strftime("%Y-%m-%d %H:%M:%S %Z")
        format_str = logging.Formatter(fmt, cst_date)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        # sh = logging.StreamHandler()#往屏幕上输出
        # sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=self.logfile,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)#设置文件里写入的格式
        # self.logger.addHandler(sh) #把对象加到logger里
        print(cst_date+": "+message)
        self.logger.addHandler(th)
        self.logger.info(message)
        # self.logger.removeHandler(sh)
        self.logger.removeHandler(th)

if __name__ == '__main__':
    logefile = r"D:\HT_Project\风险灾害项目\算法测试\算法测试.log"
    obj = Logger(logfile=logefile)
    obj.logmessage("日志内容")