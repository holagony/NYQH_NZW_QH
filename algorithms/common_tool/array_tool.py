"""
数组的一些处理工具
@Version<1> 2021-11-19 Created by lyb
"""

import numpy as np

class ArrayTool:

    @staticmethod
    def dataNormalProcess(data):
        """
        要素归一化处理
        :param data: array,数组，[[a1, b1],[a2, b2],[a3, b3],[a4, b4],[a5, b5],[a6, b6]]
        :return:
        """
        vals_min = np.nanmin(data, axis=0)
        vals_max = np.nanmax(data, axis=0)
        index_arr = (vals_max - vals_min)!=0
        vals_normed = np.zeros_like(data,dtype=np.float32)
        vals_normed[:, index_arr] = (data[:,index_arr] - vals_min[index_arr]) / (vals_max[index_arr] - vals_min[index_arr])
        return vals_normed

    @staticmethod
    def calcWeights(vals_normed):
        """
        信息熵赋权法
        :param vals_normed: array,归一化后的数组（0~1）,[[a1, b1],[a2, b2],[a3, b3],[a4, b4],[a5, b5],[a6, b6]]
        :return:
        """
        vals_normed = np.array(vals_normed)
        n = np.shape(vals_normed)[0]
        if n==1:
            wi = np.full(np.shape(vals_normed)[1], np.nan, dtype=np.float)
        else:
            pij = np.zeros_like(vals_normed, dtype=np.float32)
            sum_array = np.nansum(vals_normed, axis=0)
            pij[:, (sum_array!=0)] = vals_normed[:, (sum_array!=0)] / sum_array[(sum_array!=0)]
            k = -1 / np.log(n)
            pij_log = np.zeros_like(vals_normed, dtype=np.float32)
            pij_log[pij!=0] = np.log(pij[pij!=0])
            si = k*np.nansum(pij_log*pij,axis=0)
            wi = np.zeros_like(si, dtype=np.float32)
            wi[(sum_array!=0)] = (1-si[(sum_array!=0)]) / np.nansum(1-si[(sum_array!=0)])
        return wi

if __name__ == "__main__":
    import gdal
    import glob
    import os
    indir = r"F:\河北气象灾害普查\河北三县结果\平山、涞水和滦州危险性指数与等级图\平山县_20220104\综合\各灾害危险指数"
    files = glob.glob(os.path.join(indir, "*.tif"))
    order_data =[]
    list_data=[]
    for file in files:
        filename = os.path.basename(file)
        print(filename)

        ds = gdal.Open(file)
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()

        data = ds.ReadAsArray()
        order_data.append(data)

        data = data[data!=-999]
        list_data.append(data)
    list_data = np.array(list_data).T
    obj=ArrayTool()
    a = obj.dataNormalProcess(list_data)
    b = obj.calcWeights(a)
    print(b)
    s = 0
    for i in range(len(order_data)):
        data = order_data[i]
        min_v = np.min(data[data!=-999])
        max_v = np.max(data[data!=-999])
        data[data!=-999] = (data[data!=-999]-min_v)/(max_v-min_v)
        s=s+data*b[i]
    s[s<0]=-999
    print(1)
    outfile =r"F:\河北气象灾害普查\河北三县结果\平山、涞水和滦州危险性指数与等级图\平山县_20220104\综合\综合危险性\zh77.tif"
    driver = gdal.GetDriverByName("GTIFF")
    outds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
    outds.SetGeoTransform(geo)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(s)
    outband.SetNoDataValue(-999)
