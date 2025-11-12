#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据等级划分
@Version <1> 2021-03-12 Created by lyb
"""

import jenkspy
import numpy as np

class RankAlg:

    @staticmethod
    def divideRankByPercent(data, rank_content, rank_field, rank_descript_field,
                            min_value_field, max_value_field, nodata=0):
        """
         百分位划分等级
         :param data: list或tuple， array, 要分级的数据
         :param rank_content: dict,数据划分等级信息 {rank_field:[1,2,3,4], rank_descript_field:["高", "较高",  "较低", "低"]，
                                                    max_value_field:[100,90,70,30], min_value_field:[90,70,30,0]}
         :param rank_field: str, 划分等级的字段名, 包含在rank_content中
         :param rank_descript_field: str, 等级描述字段名, 包含在rank_content中
         :param min_value_field: str, 最小值字段, 包含在rank_content中
         :param max_value_field：str,最大值字段, 包含在rank_content中
         :param nodata：float, 无效值
         :return:
         """
        index_data = np.array(data)
        # 划分等级的个数
        rank_numbers = len(rank_content[rank_field])
        new_data = index_data[index_data != nodata]
        # new_data = index_data[(index_data != -999) & (index_data != 0)]
        if new_data.size<=1:
            rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
            descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size), dtype='<U10')
        else:
            if (min_value_field in rank_content) & (max_value_field in rank_content):
                rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
                descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size), dtype='<U10')
                for i in range(rank_numbers):
                    rank = int(rank_content[rank_field][i])
                    rank_descript = rank_content[rank_descript_field][i]
                    rank_min = rank_content[min_value_field][i]
                    rank_max = rank_content[max_value_field][i]
                    if (rank_min is None) & (rank_max is not None):
                        max_value = np.nanpercentile(new_data, int(rank_max))
                        rankdata[(index_data < max_value)] = rank
                        descriptdata[(index_data < max_value)] = rank_descript
                    if (rank_min is not None) & (rank_max is None):
                        min_value = np.nanpercentile(new_data, int(rank_min))
                        rankdata[(index_data >= min_value)] = rank
                        descriptdata[(index_data >= min_value)] = rank_descript
                    if (rank_min is None) & (rank_max is None):
                        pass
                    if (rank_min is not None) & (rank_max is not None):
                        max_value = np.nanpercentile(new_data, int(rank_max))
                        min_value = np.nanpercentile(new_data, int(rank_min))
                        if rank_max >= 100:
                            rankdata[(index_data >= min_value) & (index_data <= max_value)] = rank
                            descriptdata[(index_data >= min_value) & (index_data <= max_value)] = rank_descript
                        else:
                            rankdata[(index_data >= min_value) & (index_data < max_value)] = rank
                            descriptdata[(index_data >= min_value) & (index_data < max_value)] = rank_descript
            else:
                # 获取划分等级的百分位位置
                locations = []
                for i in range(1, rank_numbers):
                    locations.append(int(np.round(i/rank_numbers*100)))
                thresholds = []
                for location in locations:
                    thresholds.append(np.nanpercentile(new_data, location))
                thresholds.sort(reverse=True)
                rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
                descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size), dtype='<U10')
                for i in range(len(thresholds)):
                    if i == 0:
                        rankdata[(index_data >=thresholds[i])] = rank_content[rank_field][i]
                        descriptdata[(index_data >=thresholds[i])] = rank_content[rank_descript_field][i]
                    else:
                        rankdata[(index_data < thresholds[i - 1]) & (index_data >= thresholds[i])] = rank_content[rank_field][i]
                        descriptdata[(index_data < thresholds[i - 1]) & (index_data >= thresholds[i])] = rank_content[rank_descript_field][i]
                        if i == len(thresholds)-1:
                            rankdata[(index_data < thresholds[i])] = rank_content[rank_field][i+1]
                            descriptdata[(index_data < thresholds[i])] = rank_content[rank_descript_field][i+1]
        return rankdata, descriptdata

    @staticmethod
    def divideRankByNaturalBreakpoint(data, rank_content, rank_field, rank_descript_field, nodata=0):
        """
          自然断点等级划分
          :param data: list或tuple, 要分级的数据
          :param rank_content: dict,数据划分等级信息{rank_field:[1,2,3,4], rank_descript_field:["高", "较高",  "较低", "低"]}
          :param rank_field: str, 划分等级的字段名, 包含在rank_content中
          :param rank_descript_field: str, 等级描述字段名, 包含在rank_content中
          :param nodata：float, 无效值
          :return:
        """
        index_data = np.array(data)
        # 划分等级的个数
        rank_numbers = len(rank_content[rank_field])
        # 自然断点法输出分段阈值
        # newdata = index_data[(index_data!=-999)&(index_data!=0)]
        newdata = index_data[index_data != nodata]
        if newdata.size <=1:
            rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
            descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size), dtype='<U10')
        else:
            if newdata.size<=rank_numbers+2:
                newdata = np.array(list(newdata)+[np.mean(newdata)]*(rank_numbers+2-newdata.size))
            else:
                pass
            thresholds = list(jenkspy.jenks_breaks(newdata, rank_numbers))
            thresholds.sort(reverse=True)
            rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
            descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size), dtype='<U10')
            for i in range(len(thresholds)-1):
                if i == 0:
                    rankdata[(index_data >= thresholds[i + 1])] = rank_content[rank_field][i]
                    descriptdata[(index_data >= thresholds[i + 1])] = rank_content[rank_descript_field][i]
                else:
                    if i==len(thresholds)-2:
                        rankdata[(index_data < thresholds[i])] = rank_content[rank_field][i]
                        descriptdata[(index_data < thresholds[i])] = rank_content[rank_descript_field][i]
                    else:
                        rankdata[(index_data<thresholds[i]) & (index_data>=thresholds[i+1])] = rank_content[rank_field][i]
                        descriptdata[(index_data<thresholds[i]) & (index_data>=thresholds[i+1])] = rank_content[rank_descript_field][i]
        return rankdata, descriptdata

    @staticmethod
    def divideRankStd(data, rank_content, rank_field, rank_descript_field,
                         min_value_field, max_value_field, nodata=0):
        """
        标准差方法
        :param data: list或tuple, 要分级的数据
        :param rank_content: dict, 数据划分等级信息{rank_field:[1,2,3,4], rank_descript_field:["高", "较高",  "较低", "低"]，
                                                    max_value_field:[5,1,0,-1], min_value_field:[1,0,-1,-5]}
        :param rank_field: str, 划分等级的字段名, 包含在rank_content中
        :param rank_descript_field: str, 等级描述字段名, 包含在rank_content中
        :param min_value_field: str, 最小值字段, 包含在rank_content中
        :param max_value_field：str,最大值字段, 包含在rank_content中
        :param nodata：float, 无效值
        :return:
        """
        index_data = np.array(data)
        rank_numbers = len(rank_content[rank_field])
        newdata = index_data[index_data != nodata]
        if newdata.size <=1:
            rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
            descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size),dtype='<U10')
        else:
            # 均值
            mean_h = np.nanmean(newdata)
            # 标准差
            s = np.nanstd(newdata)
            rankdata = np.full(index_data.shape, rank_numbers, dtype=np.int)
            descriptdata = np.array([rank_content[rank_descript_field][rank_numbers - 1]] * (index_data.size),  dtype='<U10')
            for i in range(rank_numbers):
                rank = int(rank_content[rank_field][i])
                rank_descript = rank_content[rank_descript_field][i]
                rank_min = rank_content[min_value_field][i]
                rank_max = rank_content[max_value_field][i]
                if (rank_min is None) & (rank_max is not None):
                    rankdata[(index_data < (float(rank_max)*s + mean_h))] = rank
                    descriptdata[(index_data < (float(rank_max)*s + mean_h))] = rank_descript
                if (rank_min is not None) & (rank_max is None):
                    rankdata[(index_data >= (float(rank_min)*s + mean_h))] = rank
                    descriptdata[(index_data >= (float(rank_min)*s + mean_h))] = rank_descript
                if (rank_min is None) & (rank_max is None):
                    pass
                if (rank_min is not None) & (rank_max is not None):
                    rankdata[(index_data >= (float(rank_min) *s+ mean_h)) & (index_data < (float(rank_max) *s + mean_h))] = rank
                    descriptdata[(index_data >= (float(rank_min) *s+ mean_h)) & (index_data < (float(rank_max) *s + mean_h))] = rank_descript
        return rankdata, descriptdata


