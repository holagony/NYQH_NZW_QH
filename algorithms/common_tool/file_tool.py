"""
文件处理工具，如文件检索等
@Version<1> 2021-11-15 Created by LYB
"""

# from distutils import filelist
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd

class FileTool:

    @staticmethod
    def searchFileBySuffix(indir=None, suffix=None):
        """
        通过文件后缀检索文件
        :param indir: str, 输入路径
        :param suffix: str, 文件后缀
        :return:
        """
        filelist = glob.glob(os.path.join(indir, "*."+suffix))
        return filelist

    @staticmethod
    def searchFileByStr(indir=None, suffix=None, restr=None):
        """
        通过文件后缀检索和文件中的字符串模糊匹配
        :param indir: str, 输入路径
        :param suffix: str, 文件后缀
        :param restr：str,文件名中的字符串
        :return:
        """
        filelist = glob.glob(os.path.join(indir, "*"+restr+"*."+suffix))
        return filelist

    @staticmethod
    def searchFileByType(inputdir,suffix=None):
        '''获取路径下所有特定格式文件的全路径（包括子目录下）'''
        filelist = []
        for root, dirs, files in os.walk(inputdir):
            for file in files:
                if os.path.splitext(file)[1]==suffix:
                    filelist.append(os.path.join(root, file))
        filelist.sort()
        return filelist

    @staticmethod
    def stnSelectByShp(basestnfile,filelist,shpfile,lat_name='纬度',lon_name='经度',station_id_name='站号'):
        """ 
        筛选落在矢量中的站点信息
        basestnfile: 包含经度、维度、站号等信息
        filelist: 要进行筛选的文件列表
        shpfile: 用于筛选站点的矢量文件
        lat_name: csv文件中的经度所在列名
        lon_name: csv文件中的纬度所在列名
        station_id_name: csv文件中的站号所在列名
        return: filelist,返回经过筛选后的filelist
        """
        try:
            stnds = pd.read_csv(basestnfile, dtype=str,encoding="gbk")
        except:
            stnds = basestnfile
        lon = np.array(stnds[lon_name].astype(float))
        lat = np.array(stnds[lat_name].astype(float))
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
        # data_ds = stnds.query("站号" + ' in ' + '[' + str(stn_list)[1:-1] + ']')
        filelist_cut = []
        for file in filelist:
            stn = (os.path.basename(file)).split('_')[1]
            if stn in str(stn_list):
                filelist_cut.append(file)

        return filelist_cut 
    

    @staticmethod
    def stn_select_by_shp1(shpfile,basestnfile):
        """
        根据shpfile筛选其中的站点号
        shpfile: 用于筛选站点的矢量文件
        basestnfile: 全国站点信息
        """
        try:
            stnds = pd.read_csv(basestnfile, dtype=str, encoding="utf-8", header=0)
        except:
            stnds = pd.read_csv(basestnfile, dtype=str, encoding="gbk", header=0)
        lon = np.array(stnds["经度"]).astype(float)
        lat = np.array(stnds["纬度"]).astype(float)
        gdf = gpd.GeoDataFrame(stnds.copy(), geometry=gpd.points_from_xy(lon, lat))
        gdf.index = gdf["站号"]

        try:
            shp = gpd.read_file(shpfile, encoding='utf-8')
        except:
            shp = gpd.read_file(shpfile, encoding='gbk')
        stnlist = []

        for key, geom in shp.geometry.items():
            pips = gdf.geometry.within(geom)
            if any(pips):
                stnID = list(pips[pips == True].index.values)
                stnlist = list(stnID) + stnlist
        stn_list = list(set(stnlist))
        # data_ds = stnds.query("站号" + ' in ' + '[' + str(stn_list)[1:-1] + ']')
        # data_ds.to_csv(outfile, index=None, header=True, encoding="gbk")
        return stn_list

    @staticmethod
    def stn_select_by_shp(self,shpfile,filelist,stationInfoPath):
        """
        根据shpfile筛选其中的站点号
        shpfile: 用于筛选站点的矢量文件
        basestnfile: 全国站点信息
        """
        if shpfile is None:
            shpfile = shpfile
        # shpfile_buffer = shpfile.replace('.shp','_buffer.shp')    
        # calculate_buffer(shpfile,shpfile_buffer,buffer_distance=100000)
            

        file_df = pd.DataFrame(filelist,columns=['filepath'])
        file_df['Station_Id_C'] = file_df['filepath'].map(lambda x: os.path.basename(x)[0:5])

        try:
            csv_ds = pd.read_csv(stationInfoPath, dtype=str, encoding="utf-8")
        except:
            csv_ds = pd.read_csv(stationInfoPath, dtype=str, encoding="gbk")
        lon = np.array(csv_ds['Lon']).astype(float)
        lat = np.array(csv_ds['Lat']).astype(float)
        gdf = gpd.GeoDataFrame(csv_ds[['Station_Id_C']], geometry=gpd.points_from_xy(lon, lat))
        gdf.index = gdf['Station_Id_C']

        try:
            shp = gpd.read_file(shpfile, encoding='utf-8')
        except:
            shp = gpd.read_file(shpfile, encoding='gbk')
        stnlist = []

        for key, geom in shp.geometry.items():
            pips = gdf.geometry.within(geom)
            if any(pips):
                stnID = list(pips[pips == True].index.values)
                stnlist = list(stnID) + stnlist
        station_list = list(set(stnlist))
        # data_ds = stnds.query("站号" + ' in ' + '[' + str(stn_list)[1:-1] + ']')
        # data_ds.to_csv(outfile, index=None, header=True, encoding="gbk")
        file_df = file_df[file_df['Station_Id_C'].isin(station_list)]
        return file_df['filepath'].to_list()    
    
    @staticmethod
    def stn_select_by_PAC(areaCode,filelist,stationInfoPath,PACs=None):
        """
        利用国家站csv信息文件进行筛选
        """
        if PACs is None:
            PACs = areaCode.split(",") 

        file_df = pd.DataFrame(filelist,columns=['filepath'])
        file_df['Station_Id_C'] = file_df['filepath'].map(lambda x: os.path.basename(x)[0:5])

        try:
            csv_ds = pd.read_csv(stationInfoPath, dtype=str, encoding="utf-8")
        except:
            csv_ds = pd.read_csv(stationInfoPath, dtype=str, encoding="gbk")
   

        PACs = [i[0:2]+'0000' for i in PACs]
        station_list = csv_ds['Station_Id_C'][csv_ds['PAC_prov'].isin(PACs)]

        file_df = file_df[file_df['Station_Id_C'].isin(station_list)]
        return file_df['filepath'].to_list()