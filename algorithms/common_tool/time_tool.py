"""
时间处理
@Version<1> 2025-09-17 Created by wyx
"""
import numpy as np
import pandas as pd

class Time_Tool:
    @staticmethod
    def timeSelect(dataframe_ds,start_time, end_time,dateUnit):
        """
        根据时间截取dataFrame
        :param dataframe_ds: dataframe, pandas的dataframe数据，其中时间字段的数据格式为20200110
        :param datefield: str, 时间字段名称
        :param start_time: str, 开始时间，例如0101
        :param end_time: str, 结束时间，例如1231
        :return:
        """
        datestr = list(np.array(dataframe_ds['Datetime']))
        dt_ = pd.to_datetime(datestr, format='%Y%m%d')
        dataframe_ds.index = dt_
        if end_time[4:8]=="0229":
            end_time = end_time[0:4]+"0228"
            
        if dateUnit in ["day","year"]:
            dt_strt = pd.to_datetime(start_time, format='%Y%m%d')
            dt_end = pd.to_datetime(end_time, format='%Y%m%d')
            dataframe_ds = dataframe_ds[(dt_ >= dt_strt) & (dt_ <= dt_end)]
        elif dateUnit in ["xun","month","hou"]:
            dataframe_ds['Date'] = pd.to_datetime(dataframe_ds['Datetime']) 
            dataframe_ds['year'] = dataframe_ds['Datetime'].apply(lambda x: int(x[0:4]))            
            dataframe_ds['start'] = pd.to_datetime(dataframe_ds['Datetime'].apply(lambda x: str(x[0:4])+start_time[4:8]))
            dataframe_ds['end'] = pd.to_datetime(dataframe_ds['Datetime'].apply(lambda x: str(x[0:4])+end_time[4:8]))  
            dataframe_ds=dataframe_ds[(dataframe_ds['year'] >= int(start_time[0:4])) & \
                                (dataframe_ds['year'] <= int(end_time[0:4])) & \
                                (dataframe_ds['Date']>=dataframe_ds['start']) & \
                                    (dataframe_ds['Date']<=dataframe_ds['end'])]
            dataframe_ds.drop(['Date','start','end',"year"],axis=1,inplace=True)  
        elif dateUnit=="season":
            if end_time[4:8] in ["0228","0229"]:
                dataframe_ds['year'] = dataframe_ds['Datetime'].apply(lambda x: int(x[0:4])) 
                dataframe_ds['Date'] = dataframe_ds['Datetime'].apply(lambda x: int(x[4:8])) 
                dataframe_ds=dataframe_ds[(dataframe_ds['year'] >= int(start_time[0:4])) & \
                                    (dataframe_ds['year'] <= (int(end_time[0:4])-1))&((dataframe_ds['Date']>=int(start_time[4:8]))|
                                        (dataframe_ds['Date']<=int(end_time[4:8])))]
                    
                dataframe_ds.drop(['Date',"year"],axis=1,inplace=True)            
            else:
                dataframe_ds['Date'] = pd.to_datetime(dataframe_ds['Datetime']) 
                dataframe_ds['year'] = dataframe_ds['Datetime'].apply(lambda x: int(x[0:4]))            
                dataframe_ds['start'] = pd.to_datetime(dataframe_ds['Datetime'].apply(lambda x: str(x[0:4])+start_time[4:8]))
                dataframe_ds['end'] = pd.to_datetime(dataframe_ds['Datetime'].apply(lambda x: str(x[0:4])+end_time[4:8]))  
                dataframe_ds=dataframe_ds[(dataframe_ds['year'] >= int(start_time[0:4])) & \
                                    (dataframe_ds['year'] <= int(end_time[0:4])) & \
                                    (dataframe_ds['Date']>=dataframe_ds['start']) & \
                                        (dataframe_ds['Date']<=dataframe_ds['end'])]
                dataframe_ds.drop(['Date','start','end',"year"],axis=1,inplace=True) 

        return dataframe_ds

