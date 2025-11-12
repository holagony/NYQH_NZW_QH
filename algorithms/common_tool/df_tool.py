"""
pandas的DATAFrame的相关处理
@version<1.1> 2022-07-15 updated by WYX
@version<1> 2021-11-15 Created by LYB
"""

import numpy as np
import pandas as pd
import geopandas as gpd

class DataFrameTool:

    @staticmethod
    def rangeSelect(dataframe_ds, datefield, start_time, end_time):
        """
        根据时间截取dataFrame
        :param dataframe_ds: dataframe, pandas的dataframe数据，其中时间字段的数据格式为20200110
        :param datefield: str, 时间字段名称
        :param start_time: str, 开始时间，20200101
        :param end_time: str, 结束时间，20201231
        :return:
        """
        datestr = list(np.array(dataframe_ds[datefield]).astype(np.str))
        dt_ = pd.to_datetime(datestr, format='%Y%m%d')
        dt_strt = pd.to_datetime(start_time, format='%Y%m%d')
        dt_end = pd.to_datetime(end_time, format='%Y%m%d')
        dataframe_ds.index = dt_
        dataframe_ds = dataframe_ds[(dt_ >= dt_strt) & (dt_ <= dt_end)]
        return dataframe_ds

    @staticmethod
    def rangeSelectTwo(dataframe_ds, datefield, start_time, end_time):
        """
        根据时间截取dataFrame
        :param dataframe_ds: dataframe, pandas的dataframe数据，其中时间字段的数据格式为2020-01-10
        :param datefield: str, 时间字段
        :param start_time: str, 开始时间，20200101
        :param end_time: str, 结束时间，20201231
        :return:
        """
        dt_ = pd.to_datetime(dataframe_ds[datefield].values, format='%Y-%m-%d')
        dt_strt = pd.to_datetime(start_time, format='%Y%m%d')
        dt_end = pd.to_datetime(end_time, format='%Y%m%d')
        dataframe_ds.index = dt_
        dataframe_ds = dataframe_ds[(dt_ >= dt_strt) & (dt_ <= dt_end)]
        return dataframe_ds

    @staticmethod
    def stnSelectByShp(basestnfile,shpfile,lat_name='纬度',lon_name='经度',station_id_name='站号'):
        """ 
        筛选落在矢量中的站点信息
        basestnfile: 要进行筛选的csv文件或dataframe,包含经度、维度、站号等信息
        shpfile: 用于筛选站点的矢量文件
        lat_name: csv文件中的经度所在列名
        lon_name: csv文件中的纬度所在列名
        station_id_name: csv文件中的站号所在列名
        return: data_ds,返回经过筛选后的dataframe
        """
        try:
            stnds = pd.read_csv(basestnfile, dtype=str)
        except:
            stnds = basestnfile
        lon = np.array(stnds[lon_name]).astype(float)
        lat = np.array(stnds[lat_name]).astype(float)
        gdf = gpd.GeoDataFrame(stnds.copy(), geometry=gpd.points_from_xy(lon, lat))
        gdf.index = gdf[station_id_name]

        shp = gpd.read_file(shpfile)
        stnlist = []

        for key, geom in shp.geometry.items():
            pips = gdf.geometry.within(geom)
            if any(pips):
                stnID = list(pips[pips == True].index.values)
                stnlist = list(stnID) + stnlist
        stn_list = list(set(stnlist))
        data_ds = stnds.query("站号" + ' in ' + '[' + str(stn_list)[1:-1] + ']')
       
        return data_ds  