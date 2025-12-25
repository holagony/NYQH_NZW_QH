"""
栅格数据处理
@Version<1> 2021-11-20 Created by lyb
"""

import re
import os
try:
    import gdal, osr, ogr
except:
    from osgeo import gdal, osr, ogr
import shapefile
from tqdm import tqdm
import numpy as np
from pykrige.ok import OrdinaryKriging
import jenkspy
import scipy.ndimage as nd
import geopandas as gpd
import pandas as pd
from scipy import interpolate
from scipy.ndimage import sobel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as msee
from scipy.interpolate import Rbf
import statsmodels.api as sm
# Enable GDAL/OGR exceptions
gdal.UseExceptions()
from copy import deepcopy
from scipy import ndimage as nd

class RasterTool:

    def save_array_to_tif(array, out_fullpath, proj, gt,nodata=-32768):
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(str(out_fullpath), array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])     # GDT_Float32
        # spatial_ref = osr.SpatialReference()
        # spatial_ref.ImportFromEPSG(3857)
        out_ds.SetProjection(proj)     #spatial_ref.ExportToWkt()
        out_ds.SetGeoTransform(gt)
        # (-0.5 * ds.RasterXSize * res_in, res_in, 0, 0.5 * ds.RasterYSize * res_in, 0, -res_in)
        band = out_ds.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(nodata)
        band = None  # save, close
        out_ds = None  # save, close

    @staticmethod
    def ml_rf_(datafile,demfile,lulcfile,otherfiles,gridfile_cut,var_name,min_value,max_value,outfile):
        """
        机器学习-随机森林算法推算指标
        datafile   包含样本数据的csv文件
        demfile  剪切到计算区域的高程tif路径，也可使未剪切的
        lulcfile   计算区域对应的土地利用tif路径，
        otherfiles   其他产品文件list[file1,file2...],未经裁剪,可以按照需要增加
        gridfile_cut    最后要生成产品的格网数据
        var_name   要推算的指标
        min_value  指标最小值
        max_value  指标最大值
        outfile  推算指标结果文件

        """
        
        #数据预处理----高程数据算坡度--------------------------------------
        # tempfile=os.path.join(os.path.dirname(outfile),"temp.tif")
        # alignRePrjRmp(gridfile_cut, demfile, tempfile, srs_nodata=None, dst_nodata=-32768, resample_type=gdal.GRA_NearestNeighbour)
        in_ds_dem=gdal.Open(demfile)
        gtt= in_ds_dem.GetGeoTransform()
        alti_array= in_ds_dem.GetRasterBand(1).ReadAsArray() #读取波段数据
        Nodata=in_ds_dem.GetRasterBand(1).GetNoDataValue()
        alti_array = np.where(alti_array==Nodata, np.nan, alti_array)  
        #求纬度
        width = in_ds_dem.RasterXSize    #cols
        height = in_ds_dem.RasterYSize    #rows
        x = np.linspace(gtt[0], gtt[0] + gtt[1]*width, width)
        y = np.linspace(gtt[3], gtt[3] + gtt[5]*height, height)
        lon, lat = np.meshgrid(x, y)
        # 计算坡度
        # 使用Sobel算子计算水平和垂直方向的梯度
        dz_dx = sobel(alti_array, axis=1) / (gtt[1]*111320 * np.cos(np.radians(lat)))
        dz_dy = sobel(alti_array, axis=0) / (gtt[1]*111320 * np.cos(np.radians(lat)))

        
        # 计算坡度（以度为单位）和坡向
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
        aspect = np.arctan2(dz_dy , dz_dx)
        slope=np.where(alti_array==Nodata, np.nan, slope)  
        aspect==np.where(alti_array==Nodata, np.nan, aspect) 
        #训练模型---------------------------------------------
        data_ds=""
        if type(data_ds) == str:
            try:
                data_ds = pd.read_csv(datafile, dtype=str, encoding="gbk")
            except:
                data_ds = pd.read_csv(datafile, dtype=str, encoding="utf-8")
        else:
            data_ds = data_ds 
        data_ds["Lon"]=data_ds["Lon"].astype(float)
        data_ds["Lat"]=data_ds["Lat"].astype(float)
        #坡度数据获取
        data_ds["x"] = ((data_ds["Lon"] - gtt[0]) / gtt[1]).astype(int)
        data_ds["y"] = ((data_ds["Lat"] - gtt[3]) / gtt[5]).astype(int)
        data_ds["slope"]=0
        data_ds["aspect"]=0
        for i in range(len(data_ds)):
            # 获取x和y坐标
            x = int(data_ds.loc[i, "x"])  # 确保x是整数
            y = int(data_ds.loc[i, "y"])  # 确保y是整数
        
            # 检查x和y是否在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                raise ValueError(f"坐标({x}, {y})超出栅格数据范围")
        
            # 读取单个像素的值
            pixel_value = slope[y,x]   #易错
            aspect_value = aspect[y,x]   #易错
            # 将结果存储到data_ds中
            data_ds.loc[i, "slope"] = pixel_value
            data_ds.loc[i, "aspect"] = aspect_value
        #土地类型获取
        # alignRePrjRmp(gridfile_cut, lulcfile, tempfile, srs_nodata=None, dst_nodata=None, resample_type=gdal.GRA_NearestNeighbour)
        in_ds_lulc=gdal.Open(lulcfile)
        lulc_array= in_ds_lulc.GetRasterBand(1).ReadAsArray() #读取波段数据 
        lulc_array=np.where(lulc_array==0, np.nan, lulc_array)     #重要
        for i in range(len(data_ds)):
            # 获取x和y坐标
            x = int(data_ds.loc[i, "x"])  # 确保x是整数
            y = int(data_ds.loc[i, "y"])  # 确保y是整数
        
            # 检查x和y是否在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                raise ValueError(f"坐标({x}, {y})超出栅格数据范围")
        
            # 读取单个像素的值
            pixel_value = lulc_array[y,x]   #易错
        
            # 将结果存储到data_ds中
            data_ds.loc[i, "lulc"] = pixel_value
        if otherfiles!="":
            otherfiles_=otherfiles.split(',')
            #产品数据获取 
            for ii,file in enumerate(otherfiles_):
                
                dataset = gdal.Open(file)
                if dataset is None:
                    print("无法打开文件")
                    return None
                geotransform = dataset.GetGeoTransform()
                if geotransform is None:
                    print("无法获取地理参考信息")
                    return None
                data_ds["x"] = ((data_ds["Lon"] - geotransform[0]) / geotransform[1]).astype(int)
                data_ds["y"] = ((data_ds["Lat"] - geotransform[3]) / geotransform[5]).astype(int)
                band = dataset.GetRasterBand(1)
                for j in range(len(data_ds)):
                    # 获取x和y坐标
                    x = int(data_ds.loc[j, "x"])  # 确保x是整数
                    y = int(data_ds.loc[j, "y"])  # 确保y是整数
                
                    # 检查x和y是否在有效范围内
                    if x < 0 or x >= dataset.RasterXSize or y < 0 or y >= dataset.RasterYSize:
                        raise ValueError(f"坐标({x}, {y})超出栅格数据范围")                
                # 获取像素值
                    data_ds.loc[j,"other_"+str(ii)] = band.ReadAsArray(x, y, 1, 1)[0][0]
        else:
            pass

        Sample_DT_temp=pd.DataFrame() 
        #selected_columns = data_ds.columns[0:3].tolist() + [data_ds.columns[8]]+data_ds.columns[11:].tolist()
        selected_columns =[data_ds.columns[2]] + [data_ds.columns[8]]+data_ds.columns[11:].tolist()
        Sample_DT_temp=data_ds[selected_columns]
        Sample_DT_temp=Sample_DT_temp.dropna(axis=0,inplace=False)   
        train_rate = 0.8
        n_estimators = 150
        max_depth = 40    #40
        min_samples_split = 2
        min_samples_leaf = 1
        max_leaf_nodes = None
        x_select = [col for col in Sample_DT_temp.columns if col != var_name]
        X = Sample_DT_temp[x_select].copy()
        y = Sample_DT_temp[var_name].copy()
        X_train, X_val_rf, y_train, y_val = train_test_split(X, y, test_size=1-train_rate)
        # 训练
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
        rf.fit(X_train, y_train) 
        y_pred_val_ = rf.predict(X_val_rf)
        #均方误差MSE
        mse = msee(y_val, y_pred_val_)
        #均方根误差
        rmse = np.sqrt(mse)
        #平均绝对误差
        mae = mean_absolute_error(y_val, y_pred_val_)
        #决定系数
        r2 = r2_score(y_val, y_pred_val_)    
        
        with open(os.path.join(os.path.dirname(outfile),"random_forest_report.txt"), 'w') as f:
            f.write(f"均方误差: {mse:.4f}\n")
            f.write(f"均方根误差: {rmse:.4f}\n")
            f.write(f"平均绝对误差: {mae:.4f}\n")
            f.write(f"决定系数: {r2:.4f}\n")         
            
        #创建数据集---------------------------------------------------------
        #经纬度数据
        in_ds =  gdal.Open(gridfile_cut)
        gt = in_ds.GetGeoTransform()
        sf = in_ds.GetProjection()
        width = in_ds.RasterXSize    #cols
        height = in_ds.RasterYSize    #rows
        x = np.linspace(gt[0], gt[0] + gt[1]*width, width)
        y = np.linspace(gt[3], gt[3] + gt[5]*height, height)
        # longitude, latitude = np.meshgrid(x, y)

        # longitude= np.where(alti_array==Nodata, np.nan, longitude)  
        # latitude= np.where(alti_array==Nodata, np.nan, latitude) 
        block_data = {
            'Alti': alti_array.flatten(),
            'slope':slope.flatten(),
            'aspect':aspect.flatten(),
            'lulc':lulc_array.flatten()
        } 
        if otherfiles!="":
            #otherfiles的数据
            for jj,file in enumerate(otherfiles_):
                # alignRePrjRmp(gridfile_cut, file, tempfile, srs_nodata=None, dst_nodata=None, resample_type=gdal.GRA_NearestNeighbour)
                in_ds_temp=gdal.Open(file)
                temp_array= in_ds_temp.GetRasterBand(1).ReadAsArray() #读取波段数据
                Nodata=in_ds_temp.GetRasterBand(1).GetNoDataValue()
                temp_array = np.where(temp_array==Nodata, np.nan, temp_array) 
                temp_dict={str("other_"+str(jj)):temp_array.flatten()}  
                block_data.update(temp_dict)
        else:
            pass
        # 初始化预测结果数组
        block_df = pd.DataFrame(block_data)
        y_pred_all_array = np.full((height, width), np.nan)
        X_val=block_df.dropna(axis=0)
        X_val_y=block_df.copy()
        X_val_y['y']=-32768
        #预测结果生成-----------------------------------------------------------
        if not X_val.empty:
            # 模型预测
            y_pred_val = rf.predict(X_val)
            #y_pred_val_int = np.around(y_pred_val).astype(int)
        if min_value is None:
            pass
        else:
            y_pred_val[y_pred_val < min_value] =min_value
        if max_value is None:
            pass
        else:
            y_pred_val[y_pred_val > max_value] = max_value
        X_val_y.loc[X_val.index, "y"] = y_pred_val   #很重要,把y的预测值赋值给x_val_y,并根据X_val_not_nan的索引填入 
        y_pred_all_array = np.array(X_val_y["y"]).reshape(height, width)     #设置成行列
        save_array_to_tif(y_pred_all_array,outfile , sf, gt) 
        # delete_file(tempfile)

        return outfile

    @staticmethod
    def mlp_(datafile,demfile,lulcfile,otherfiles,gridfile_cut,var_name,min_value,max_value,outfile):
        """
        机器学习-神经网络算法推算指标
        datafile   包含样本数据的csv文件
        demfile  剪切到计算区域的高程tif路径，也可使未剪切的
        lulcfile   计算区域对应的土地利用tif路径，
        otherfiles   其他产品文件list[file1,file2...],未经裁剪,可以按照需要增加
        gridfile_cut    最后要生成产品的格网数据
        var_name   要推算的指标
        min_value  指标最小值
        max_value  指标最大值
        outfile  推算指标结果文件

        """
        #数据预处理----高程数据算坡度--------------------------------------
        tempfile=os.path.join(os.path.dirname(outfile),"temp.tif")
        alignRePrjRmp(gridfile_cut, demfile, tempfile, srs_nodata=None, dst_nodata=-32768, resample_type=gdal.GRA_NearestNeighbour)
        in_ds_dem=gdal.Open(tempfile)
        gtt= in_ds_dem.GetGeoTransform()
        alti_array= in_ds_dem.GetRasterBand(1).ReadAsArray() #读取波段数据
        Nodata=in_ds_dem.GetRasterBand(1).GetNoDataValue()
        alti_array = np.where(alti_array==Nodata, np.nan, alti_array)  
        #求纬度
        width = in_ds_dem.RasterXSize    #cols
        height = in_ds_dem.RasterYSize    #rows
        x = np.linspace(gtt[0], gtt[0] + gtt[1]*width, width)
        y = np.linspace(gtt[3], gtt[3] + gtt[5]*height, height)
        lon, lat = np.meshgrid(x, y)
        # 计算坡度
        # 使用Sobel算子计算水平和垂直方向的梯度
        dz_dx = sobel(alti_array, axis=1) / (gtt[1]*111320 * np.cos(np.radians(lat)))
        dz_dy = sobel(alti_array, axis=0) / (gtt[1]*111320 * np.cos(np.radians(lat)))

        
        # 计算坡度（以度为单位）和坡向
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
        aspect = np.arctan2(dz_dy , dz_dx)
        slope=np.where(alti_array==Nodata, np.nan, slope)  
        aspect==np.where(alti_array==Nodata, np.nan, aspect) 
        #训练模型---------------------------------------------
        data_ds=""
        if type(data_ds) == str:
            try:
                data_ds = pd.read_csv(datafile, dtype=str, encoding="gbk")
            except:
                data_ds = pd.read_csv(datafile, dtype=str, encoding="utf-8")
        else:
            data_ds = data_ds 
        data_ds["Lon"]=data_ds["Lon"].astype(float)
        data_ds["Lat"]=data_ds["Lat"].astype(float)
        #坡度数据获取
        data_ds["x"] = ((data_ds["Lon"] - gtt[0]) / gtt[1]).astype(int)
        data_ds["y"] = ((data_ds["Lat"] - gtt[3]) / gtt[5]).astype(int)
        data_ds["slope"]=0
        data_ds["aspect"]=0
        for i in range(len(data_ds)):
            # 获取x和y坐标
            x = int(data_ds.loc[i, "x"])  # 确保x是整数
            y = int(data_ds.loc[i, "y"])  # 确保y是整数
        
            # 检查x和y是否在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                raise ValueError(f"坐标({x}, {y})超出栅格数据范围")
        
            # 读取单个像素的值
            pixel_value = slope[y,x]   #易错
            aspect_value = aspect[y,x]   #易错
            # 将结果存储到data_ds中
            data_ds.loc[i, "slope"] = pixel_value
            data_ds.loc[i, "aspect"] = aspect_value
        #土地类型获取
        alignRePrjRmp(gridfile_cut, lulcfile, tempfile, srs_nodata=None, dst_nodata=None, resample_type=gdal.GRA_NearestNeighbour)
        in_ds_lulc=gdal.Open(tempfile)
        lulc_array= in_ds_lulc.GetRasterBand(1).ReadAsArray() #读取波段数据 
        lulc_array=np.where(lulc_array==0, np.nan, lulc_array)     #重要
        for i in range(len(data_ds)):
            # 获取x和y坐标
            x = int(data_ds.loc[i, "x"])  # 确保x是整数
            y = int(data_ds.loc[i, "y"])  # 确保y是整数
        
            # 检查x和y是否在有效范围内
            if x < 0 or x >= width or y < 0 or y >= height:
                raise ValueError(f"坐标({x}, {y})超出栅格数据范围")
        
            # 读取单个像素的值
            pixel_value = lulc_array[y,x]   #易错
        
            # 将结果存储到data_ds中
            data_ds.loc[i, "lulc"] = pixel_value
        if otherfiles!="":
            otherfiles_=otherfiles.split(',')
            #产品数据获取 
            for ii,file in enumerate(otherfiles_):
                
                dataset = gdal.Open(file)
                if dataset is None:
                    print("无法打开文件")
                    return None
                geotransform = dataset.GetGeoTransform()
                if geotransform is None:
                    print("无法获取地理参考信息")
                    return None
                data_ds["x"] = ((data_ds["Lon"] - geotransform[0]) / geotransform[1]).astype(int)
                data_ds["y"] = ((data_ds["Lat"] - geotransform[3]) / geotransform[5]).astype(int)
                band = dataset.GetRasterBand(1)
                for j in range(len(data_ds)):
                    # 获取x和y坐标
                    x = int(data_ds.loc[j, "x"])  # 确保x是整数
                    y = int(data_ds.loc[j, "y"])  # 确保y是整数
                
                    # 检查x和y是否在有效范围内
                    if x < 0 or x >= dataset.RasterXSize or y < 0 or y >= dataset.RasterYSize:
                        raise ValueError(f"坐标({x}, {y})超出栅格数据范围")                
                # 获取像素值
                    data_ds.loc[j,"other_"+str(ii)] = band.ReadAsArray(x, y, 1, 1)[0][0]
        else:
            pass

        Sample_DT_temp=pd.DataFrame() 
        #selected_columns = data_ds.columns[0:3].tolist() + [data_ds.columns[8]]+data_ds.columns[11:].tolist()
        selected_columns =[data_ds.columns[2]] + [data_ds.columns[8]]+data_ds.columns[11:].tolist()
        Sample_DT_temp=data_ds[selected_columns]
        Sample_DT_temp=Sample_DT_temp.dropna(axis=0,inplace=False)   
        train_rate = 0.8
        # n_estimators = 150
        # max_depth = 40    #40
        # min_samples_split = 2
        # min_samples_leaf = 1
        # max_leaf_nodes = None
        x_select = [col for col in Sample_DT_temp.columns if col != var_name]
        X = Sample_DT_temp[x_select].copy()
        y = Sample_DT_temp[var_name].copy()
        X_train, X_val_rf, y_train, y_val = train_test_split(X, y, test_size=1-train_rate,random_state=42)
        # 训练
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)

        mlp.fit(X_train, y_train) 
        y_pred_val_ = mlp.predict(X_val_rf)
        #均方误差MSE
        mse = msee(y_val, y_pred_val_)
        #均方根误差
        rmse = np.sqrt(mse)
        #平均绝对误差
        mae = mean_absolute_error(y_val, y_pred_val_)
        #决定系数
        r2 = r2_score(y_val, y_pred_val_)    
        
        with open(os.path.join(os.path.dirname(outfile),"mlp_report.txt"), 'w') as f:
            f.write(f"均方误差: {mse:.4f}\n")
            f.write(f"均方根误差: {rmse:.4f}\n")
            f.write(f"平均绝对误差: {mae:.4f}\n")
            f.write(f"决定系数: {r2:.4f}\n")         
            
        
        
        #创建数据集---------------------------------------------------------
        #经纬度数据
        in_ds =  gdal.Open(gridfile_cut)
        gt = in_ds.GetGeoTransform()
        sf = in_ds.GetProjection()
        width = in_ds.RasterXSize    #cols
        height = in_ds.RasterYSize    #rows
        x = np.linspace(gt[0], gt[0] + gt[1]*width, width)
        y = np.linspace(gt[3], gt[3] + gt[5]*height, height)
        # longitude, latitude = np.meshgrid(x, y)

        # longitude= np.where(alti_array==Nodata, np.nan, longitude)  
        # latitude= np.where(alti_array==Nodata, np.nan, latitude) 
        block_data = {
            'Alti': alti_array.flatten(),
            'slope':slope.flatten(),
            'aspect':aspect.flatten(),
            'lulc':lulc_array.flatten()
        } 
        if otherfiles!="":
            #otherfiles的数据
            for jj,file in enumerate(otherfiles_):
                alignRePrjRmp(gridfile_cut, file, tempfile, srs_nodata=None, dst_nodata=None, resample_type=gdal.GRA_NearestNeighbour)
                in_ds_temp=gdal.Open(tempfile)
                temp_array= in_ds_temp.GetRasterBand(1).ReadAsArray() #读取波段数据
                Nodata=in_ds_temp.GetRasterBand(1).GetNoDataValue()
                temp_array = np.where(temp_array==Nodata, np.nan, temp_array) 
                temp_dict={str("other_"+str(jj)):temp_array.flatten()}  
                block_data.update(temp_dict)
        else:
            pass
        # 初始化预测结果数组
        block_df = pd.DataFrame(block_data)
        y_pred_all_array = np.full((height, width), np.nan)
        X_val=block_df.dropna(axis=0)
        X_val_y=block_df.copy()
        X_val_y['y']=-32768
        #预测结果生成-----------------------------------------------------------
        if not X_val.empty:
            # 模型预测
            y_pred_val = mlp.predict(X_val)
            #y_pred_val_int = np.around(y_pred_val).astype(int)
        if min_value is None:
            pass
        else:
            y_pred_val[y_pred_val < min_value] =min_value
        if max_value is None:
            pass
        else:
            y_pred_val[y_pred_val > max_value] = max_value
        X_val_y.loc[X_val.index, "y"] = y_pred_val   #很重要,把y的预测值赋值给x_val_y,并根据X_val_not_nan的索引填入 
        y_pred_all_array = np.array(X_val_y["y"]).reshape(height, width)     #设置成行列
        save_array_to_tif(y_pred_all_array,outfile , sf, gt) 
        delete_file(tempfile)

        return outfile

    @staticmethod
    def LSM(datafile, demfile,gridfile_cut, var_name, min_value, max_value, outfile, block_size=256):
        """
        小网格推算插值方法，输入变量为经度、纬度、海拔，输出值为气候资源指标
        datafile   包含样本数据的csv文件
        demfile  剪切到计算区域的高程tif路径，
        gridfile_cut   剪切到计算区域的格点tif路径
        var_name   要推算的指标
        min_value  指标最小值
        max_value  指标最大值
        outfile  推算指标结果文件
        block_size  分块大小，默认为1000
        """
        #建模---------------------------------
        # 读取样本数据
        try:
            data_ds = pd.read_csv(datafile, dtype=str, encoding="gbk")
        except:
            data_ds = pd.read_csv(datafile, dtype=str, encoding="utf-8")
        data_ds=data_ds[["Lon","Lat","Alti",var_name]]
        data_ds=data_ds.dropna()
        data_ds["Lon"] = data_ds["Lon"].astype(float)
        data_ds["Lat"] = data_ds["Lat"].astype(float)
        data_ds["Alti"] = data_ds["Alti"].astype(float)
        data_ds[var_name]=data_ds[var_name].astype(float)
        X = data_ds[['Lon', 'Lat', 'Alti']]  # 自变量
        y = data_ds[var_name]  # 因变量

        # 使用最小二乘法拟合多元线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        #模型精度输出----------------------------------------
        y_pred = model.predict(X)
        
        #均方误差MSE
        mse = msee(y, y_pred)
        #均方根误差
        rmse = np.sqrt(mse)
        #平均绝对误差
        mae = mean_absolute_error(y, y_pred)
        #决定系数
        r2 = r2_score(y, y_pred)    
        
        with open(os.path.join(os.path.dirname(outfile),"LSM_report.txt"), 'w') as f:
            f.write(f"均方误差: {mse:.4f}\n")
            f.write(f"均方根误差: {rmse:.4f}\n")
            f.write(f"平均绝对误差: {mae:.4f}\n")
            f.write(f"决定系数: {r2:.4f}\n") 
            
            
        #预测结果---------------------------------------------------    
        # 打开高程文件
        #高程文件重采样
        tempfile=os.path.join(os.path.dirname(outfile),"temp.tif")
        alignRePrjRmp(gridfile_cut, demfile, tempfile, srs_nodata=None, dst_nodata=-32768, resample_type=gdal.GRA_NearestNeighbour)    
        dataset = gdal.Open(tempfile, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError("无法打开文件: " + tempfile)
        
        # 获取地理信息
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # 获取栅格的行列数
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize

        # 初始化结果数组
        grid_values = np.full((rows, cols), np.nan)

        # 分块处理
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                # 计算当前块的范围
                row_start = i
                row_end = min(i + block_size, rows)
                col_start = j
                col_end = min(j + block_size, cols)

                # 获取当前块的高程数据
                data_block = dataset.GetRasterBand(1).ReadAsArray(col_start, row_start, col_end - col_start, row_end - row_start)
            # --- 2. 空块直接跳过 ---
                if data_block is None:            # GDAL 读取失败
                    continue
                if np.all(data_block == dataset.GetRasterBand(1).GetNoDataValue()):
                    continue                      # 整块都是 NoData
                if data_block.size == 0:          # 极端情况：行列数为 0
                    continue
                # 计算当前块的经度和纬度
                x_coords_block = np.linspace(geotransform[0] + col_start * geotransform[1], 
                                            geotransform[0] + col_end * geotransform[1], col_end - col_start)
                y_coords_block = np.linspace(geotransform[3] + row_start * geotransform[5], 
                                            geotransform[3] + row_end * geotransform[5], row_end - row_start)
                lon_block, lat_block = np.meshgrid(x_coords_block, y_coords_block)


                block_data = {
                    'Lon': lon_block.flatten(),
                    'Lat':lat_block.flatten(),
                    'Alti':data_block.flatten()
                }             
        # 初始化预测结果数组
                block_df = pd.DataFrame(block_data)
                y_pred_all_array = np.zeros_like(lon_block)
                X_val=block_df.dropna(axis=0)
                X_val_y=block_df.copy()
                X_val_y['y']=-32768
                #预测结果生成-----------------------------------------------------------
                if not X_val.empty:
                    # 模型预测
                    y_pred_val = model.predict(X_val)
                    #y_pred_val_int = np.around(y_pred_val).astype(int)
                if min_value is None:
                    pass
                else:
                    y_pred_val[y_pred_val < int(min_value)] =int(min_value)
                if max_value is None:
                    pass
                else:
                    y_pred_val[y_pred_val > int(max_value)] = int(max_value)
                X_val_y.loc[X_val.index, "y"] = y_pred_val   #很重要,把y的预测值赋值给x_val_y,并根据X_val_not_nan的索引填入 
                y_pred_all_array = np.array(X_val_y["y"]).reshape(lon_block.shape)     #设置成行列

                grid_values[row_start:row_end, col_start:col_end] = y_pred_all_array
    
        grid_data=gdal.Open(gridfile_cut, gdal.GA_ReadOnly)  
        grid_array=grid_data.GetRasterBand(1).ReadAsArray()
        dem_array=dataset.GetRasterBand(1).ReadAsArray()
        grid_values=np.where(grid_array==0,np.nan,grid_values)
        grid_values=np.where(dem_array==-32768,np.nan,grid_values)
        # 保存结果为GeoTIFF文件
        save_array_to_tif(grid_values, outfile, projection, geotransform)
        dataset = None
        grid_data=None
        os.remove(tempfile)

    @staticmethod        
    def LSM_idw(datafile, demfile,gridfile_cut,dst_epsg, var_name, min_value, max_value, outfile, block_size=256,radius_dist=2.0, min_num=20, first_size=200):
        """
        小网格推算插值方法，输入变量为经度、纬度、海拔，输出值为气候资源指标
        datafile   包含样本数据的csv文件
        demfile  剪切到计算区域的高程tif路径，也可使未剪切的
        var_name   要推算的指标
        min_value  指标最小值
        max_value  指标最大值
        outfile  推算指标结果文件
        block_size  分块大小，默认为1000
        """
        #建模---------------------------------
        # 读取样本数据
        try:
            data_ds = pd.read_csv(datafile, dtype=str, encoding="gbk")
        except:
            data_ds = pd.read_csv(datafile, dtype=str, encoding="utf-8")
        data_ds=data_ds[["Lon","Lat","Alti",var_name]]
        data_ds=data_ds.dropna()
        data_ds["Lon"] = data_ds["Lon"].astype(float)
        data_ds["Lat"] = data_ds["Lat"].astype(float)
        data_ds["Alti"] = data_ds["Alti"].astype(float)
        data_ds[var_name]=data_ds[var_name].astype(float)
        X = data_ds[['Lon', 'Lat', 'Alti']]  # 自变量
        #print(type(X))
        y = data_ds[var_name]  # 因变量

        # # 使用最小二乘法拟合多元线性回归模型
        # model = LinearRegression()
        # model.fit(X, y)
        # 添加常数项
        #cols_order = ['const', 'Lon', 'Lat', 'Alti'] 
        X = sm.add_constant(X)
        #print("列名：", X.columns.tolist())
        #X=X[cols_order]
        #rint("预测时 X 的列数：", X.shape[1])
        # 拟合线性模型
        model = sm.OLS(y, X).fit()
        #模型残差反距离插值到全域----------------------------------------
        residuals = model.resid
        tempfile=os.path.join(os.path.dirname(outfile),"temp.tif")
        alignRePrjRmp(gridfile_cut, demfile, tempfile, srs_nodata=None, dst_nodata=-32768, resample_type=gdal.GRA_NearestNeighbour)    
        
        boundary_range, cols, rows = getColsRows(tempfile)
        outds = idwInterProcess(data=np.array(residuals.astype(float)),
                                latdata=np.array(data_ds['Lat'].astype(float)),
                                londata=np.array(data_ds['Lon'].astype(float)),
                                outfile=None, boundary_range=boundary_range,
                                dst_epsg=dst_epsg, dst_rows=rows, dst_cols=cols,
                                radius_dist=radius_dist, min_num=min_num,
                                min_value=min_value, max_value=max_value, first_size=first_size)  
        # residualsdata = outds.ReadAsArray()
        #预测结果---------------------------------------------------    
        # 打开高程文件
        dataset = gdal.Open(tempfile, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError("无法打开文件: " + tempfile)
        
        # 获取地理信息
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # 获取栅格的行列数
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize

        # 初始化结果数组
        grid_values = np.full((rows, cols), np.nan)

        # 分块处理
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block_df=pd.DataFrame()
                block_data={}
                # 计算当前块的范围
                row_start = i
                row_end = min(i + block_size, rows)
                col_start = j
                col_end = min(j + block_size, cols)

                # 获取当前块的高程数据
                data_block = dataset.GetRasterBand(1).ReadAsArray(col_start, row_start, col_end - col_start, row_end - row_start)
                residualsdata_block = outds.ReadAsArray(col_start, row_start, col_end - col_start, row_end - row_start)
                # --- 2. 空块直接跳过 ---
                if data_block is None:            # GDAL 读取失败
                    continue
                if np.all(data_block == dataset.GetRasterBand(1).GetNoDataValue()):
                    continue                      # 整块都是 NoData
                if data_block.size == 0:          # 极端情况：行列数为 0
                    continue
                # 计算当前块的经度和纬度
                x_coords_block = np.linspace(geotransform[0] + col_start * geotransform[1], 
                                            geotransform[0] + col_end * geotransform[1], col_end - col_start)
                y_coords_block = np.linspace(geotransform[3] + row_start * geotransform[5], 
                                            geotransform[3] + row_end * geotransform[5], row_end - row_start)
                lon_block, lat_block = np.meshgrid(x_coords_block, y_coords_block)


                block_data = {
                    'Lon': lon_block.flatten(),
                    'Lat':lat_block.flatten(),
                    'Alti':data_block.flatten()
                }             
                # 初始化预测结果数组
                block_df = pd.DataFrame(block_data)
                
                #block_df = sm.add_constant(block_df)
                #print("block_df列名：", block_df.columns.tolist())
                y_pred_all_array = np.zeros_like(lon_block)
                
                # 过滤空值
                X_vala = block_df.dropna(subset=['Alti'])
                if X_vala.shape[0] == 0:
                    continue
                
                # 关键：加常数列并按固定顺序重排
                
                X_val = sm.add_constant(X_vala,has_constant='add')
                #print("X_val列名：", X_val.columns.tolist())
                # cols_order = ['const', 'Lon', 'Lat', 'Alti']
                # X_val = X_val[cols_order]
                X_val_y=block_df.copy()
                X_val_y['y']=-32768            
                #预测结果生成-----------------------------------------------------------
                #print("X_val列名：", X_val.columns.tolist())
                #print("预测时 X 的列数：", X_val.shape[1])
                y_pred_val = model.predict(X_val)
                

                if min_value is None:
                    pass
                else:
                    y_pred_val[y_pred_val < int(min_value)] =int(min_value)
                if max_value is None:
                    pass
                else:
                    y_pred_val[y_pred_val > int(max_value)] = int(max_value)
                X_val_y.loc[X_vala.index, "y"] = y_pred_val   #很重要,把y的预测值赋值给x_val_y,并根据X_val_not_nan的索引填入 
                y_pred_all_array = np.array(X_val_y["y"]).reshape(lon_block.shape)     #设置成行列
                #residualsdata_block=np.where(y_pred_all_array==-32768,np.nan,residualsdata_block)
                grid_values[row_start:row_end, col_start:col_end] = y_pred_all_array+residualsdata_block       
        grid_data=gdal.Open(gridfile_cut, gdal.GA_ReadOnly)
        grid_array=grid_data.GetRasterBand(1).ReadAsArray() 
        dem_array=dataset.GetRasterBand(1).ReadAsArray() 
        grid_values=np.where(grid_array==0,np.nan,grid_values)
        
        grid_values=np.where(dem_array==-32768,np.nan,grid_values)
        #种植区划适宜度分级
        # y_class=np.clip(grid_values,0,1)
        # y_class=np.where(grid_values<0.3,4,y_class)
        # y_class=np.where((grid_values>=0.3)&(grid_values<0.5),3,y_class)
        # y_class=np.where((grid_values>=0.5)&(grid_values<0.7),2,y_class)
        # y_class=np.where((grid_values>=0.7)&(grid_values<=1),1,y_class)
        # y_class=np.where(np.isnan(grid_values),-32768,y_class)
        # y_class = y_class.astype(np.int32)  
        # save_array_to_tif(y_class, outfile, projection, geotransform)
        # 保存结果为GeoTIFF文件
        RasterTool.save_array_to_tif(grid_values, outfile, projection, geotransform)
        dataset = None
        grid_data=None
        # os.remove(tempfile)


    @staticmethod
    def rbfInterProcess(data=None, latdata=None, londata=None,
                        outfile=None,
                        boundary_range=None, dst_epsg=None,
                        dst_rows=None, dst_cols=None,
                        radius_dist=1.0, min_num=10,
                        min_value=None, max_value=None):
        """
        径向基插值
        :param data: array，数据数组
        :param latdata: array, 纬度数组
        :param londata: array， 经度数据
        :param outfile: str,结果路径
        :param boundary_range: list, 插值范围
        :param dst_rows: float, 行数
        :param dst_cols: float, 列数
        :param dst_epsg: int, 输出坐标系,epsg编码
        :param radius_dist: float, 插值时的检索半径
        :param min_num: int, 最小点的个数
        :param max_value: float, 插值结果上限值
        :param min_value: float, 插值结果下限值
        :return:
        """
        try:
            if min_num > data.size:
                min_num = data.size
            cols = int(dst_cols)
            rows = int(dst_rows)
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
            outdata = np.full((rows, cols), 0, dtype=np.float32)
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            else:
                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())
            # fuc = interpolate.Rbf(londata, latdata, data, function="thin_plate")
            # xgrid, ygrid = np.meshgrid(
            #     np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, num=cols),
            #     np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, num=rows))
            # outdata = fuc(xgrid, ygrid)
            xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
            ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)
            dist = radius_dist ** 2  # 检索范围，经纬度
            for i in range(cols):
                print(str(i))
                lon = xmaps[i]
                dist_x = (londata - lon) ** 2
                for j in range(rows):
                    lat = ymaps[j]
                    dist_y = (latdata - lat) ** 2
                    dist_point = dist_x + dist_y
                    array_index = dist_point < dist
                    if np.sum(array_index) < min_num:
                        dist_copy = dist_point.copy()
                        dist_copy.sort()
                        array_index = dist_point <= dist_copy[min_num-1]
                    lons = londata[array_index]
                    lats = latdata[array_index]
                    values = data[array_index]
                    if np.sum(values==values[0]) == values.size:
                        outdata[j, i] = values[0]
                    else:
                        fuc = interpolate.Rbf(lons, lats, values, function="thin_plate")
                        rbf_v = fuc(lon, lat)
                        # fuc = interpolate.interp2d(lons, lats, values, kind='linear')
                        # rbf_v = fuc(lon, lat)[0]
                        outdata[j, i] = rbf_v
            if min_value is None:
                pass
            else:
                outdata[outdata < min_value] =min_value
            if max_value is None:
                pass
            else:
                outdata[outdata > max_value] = max_value
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outdata)
            return out_ds
        finally:
            ds = None

    # @staticmethod
    # def rbfInterProcess(data=None, latdata=None, londata=None,
    #                     outfile=None,
    #                      boundary_range=None, dst_epsg=None,
    #                      dst_rows=None, dst_cols=None,
    #                      radius_dist=1.0, min_num=10,
    #                      min_value=None, max_value=None):
    #     """
    #      径向基插值
    #      :param data: array，数据数组
    #      :param latdata: array, 纬度数组
    #      :param londata: array， 经度数据
    #      :param outfile: str,结果路径
    #      :param boundary_range: list, 插值范围
    #      :param dst_rows: float, 行数
    #      :param dst_cols: float, 列数
    #      :param dst_epsg: int, 输出坐标系,epsg编码
    #      :param radius_dist: float, 插值时的检索半径
    #      :param min_num: int, 最小点的个数
    #      :param max_value: float, 插值结果上限值
    #      :param min_value: float, 插值结果下限值
    #      :return:
    #      """
    #     try:
    #         if min_num > data.size:
    #             min_num = data.size
    #         cols = int(dst_cols)
    #         rows = int(dst_rows)
    #         x_min = boundary_range[0]
    #         y_max = boundary_range[3]
    #         x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
    #         y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
    #         outdata = np.full((rows, cols), 0, dtype=np.float32)
    #         if outfile is None:
    #             driver = gdal.GetDriverByName("MEM")
    #             out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    #         else:
    #             driver = gdal.GetDriverByName("GTiff")
    #             out_ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
    #         out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
    #         out_srs = osr.SpatialReference()
    #         out_srs.ImportFromEPSG(dst_epsg)
    #         out_ds.SetProjection(out_srs.ExportToWkt())
    #         # fuc = interpolate.Rbf(londata, latdata, data, function="thin_plate")
    #         # xgrid, ygrid = np.meshgrid(
    #         #     np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, num=cols),
    #         #     np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, num=rows))
    #         # outdata = fuc(xgrid, ygrid)
    #         xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
    #         ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)
    #         dist = radius_dist ** 2  # 检索范围，经纬度
    #         for i in range(cols):
    #             lon = xmaps[i]
    #             dist_x = (londata - lon) ** 2
    #             for j in range(rows):
    #                 lat = ymaps[j]
    #                 dist_y = (latdata - lat) ** 2
    #                 dist_point = dist_x + dist_y
    #                 array_index = dist_point < dist
    #                 if np.sum(array_index) < min_num:
    #                     dist_copy = dist_point.copy()
    #                     dist_copy.sort()
    #                     array_index = dist_point <= dist_copy[min_num-1]
    #                 lons = londata[array_index]
    #                 lats = latdata[array_index]
    #                 values = data[array_index]
    #                 if np.sum(values==values[0]) == values.size:
    #                     outdata[j, i] = values[0]
    #                 else:
    #                     fuc = interpolate.Rbf(lons, lats, values, function="thin_plate")
    #                     rbf_v = fuc(lon, lat)
    #                     # fuc = interpolate.interp2d(lons, lats, values, kind='linear')
    #                     # rbf_v = fuc(lon, lat)[0]
    #                     outdata[j, i] = rbf_v
    #         if min_value is None:
    #             pass
    #         else:
    #             outdata[outdata < min_value] =min_value
    #         if max_value is None:
    #             pass
    #         else:
    #             outdata[outdata > max_value] = max_value
    #         out_band = out_ds.GetRasterBand(1)
    #         out_band.WriteArray(outdata)
    #         return out_ds
    #     finally:
    #         ds = None

    # @staticmethod
    # def idwInterProcess(data=None, latdata=None, londata=None,
    #                       outfile=None,
    #                      boundary_range=None, dst_epsg=None,
    #                      dst_rows=None, dst_cols=None,
    #                      radius_dist=1.0, min_num=10,
    #                     min_value=None, max_value=None,
    #                     first_size=1500):
    #     """
    #     反距离插值
    #     :param data: array，数据数组
    #     :param latdata: array, 纬度数组
    #     :param londata: array， 经度数据
    #     :param outfile: str,结果路径
    #     :param boundary_range: list, 插值范围
    #     :param dst_rows: float, 行数
    #     :param dst_cols: float, 列数
    #     :param dst_epsg: int, 输出坐标系,epsg编码
    #     :param radius_dist: float, 插值时的检索半径
    #     :param min_num: int, 最小点的个数
    #     :param max_value: float, 插值结果上限值
    #     :param min_value: float, 插值结果下限值
    #     :return:
    #     """
    #     try:
    #         if min_num > data.size:
    #             min_num = data.size
    #         if (dst_cols>first_size) | (dst_rows>first_size):
    #             if dst_cols>dst_rows:
    #                 cols = first_size
    #                 rows = int(np.ceil(dst_rows/(dst_cols/first_size)))
    #             else:
    #                 rows = first_size
    #                 cols = int(np.ceil(dst_cols/(dst_rows/first_size)))
    #         else:
    #             cols = int(dst_cols)
    #             rows = int(dst_rows)
    #         # cols = int(dst_cols)
    #         # rows = int(dst_rows)
    #         x_min = boundary_range[0]
    #         y_max = boundary_range[3]
    #         x_cell = abs(boundary_range[2] - boundary_range[0]) /cols
    #         y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
    #         outdata = np.full((rows, cols), 0, dtype=np.float32)
    #         driver = gdal.GetDriverByName("MEM")
    #         out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    #         # if outfile is None:
    #         #     driver = gdal.GetDriverByName("MEM")
    #         #     out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
    #         # else:
    #         #     driver = gdal.GetDriverByName("GTiff")
    #         #     out_ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
    #         out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
    #         out_srs = osr.SpatialReference()
    #         out_srs.ImportFromEPSG(dst_epsg)
    #         out_ds.SetProjection(out_srs.ExportToWkt())
    #         xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
    #         ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)
    #         dist = radius_dist ** 2  # 检索范围，经纬度
    #         for i in range(cols):
    #             lon = xmaps[i]
    #             dist_x = (londata - lon) ** 2
    #             for j in range(rows):
    #                 lat = ymaps[j]
    #                 dist_y = (latdata - lat) ** 2
    #                 dist_point = dist_x + dist_y
    #                 array_index = dist_point < dist
    #                 if np.sum(array_index) < min_num:
    #                     dist_copy = dist_point.copy()
    #                     dist_copy.sort()
    #                     array_index = dist_point <= dist_copy[min_num-1]
    #                 lons = londata[array_index]
    #                 lats = latdata[array_index]
    #                 values = data[array_index]
    #                 if np.sum(values==values[0]) == values.size:
    #                     outdata[j, i] = values[0]
    #                 # if np.sum(values) == values[0]*values.size:
    #                 #     outdata[j, i] = values[0]
    #                 else:
    #                     dist_all = 1.0 / np.sqrt((lat - lats) ** 2 + (lon - lons) ** 2)
    #                     dist_sum = np.sum(dist_all)
    #                     idw_v = np.sum((dist_all / dist_sum) * values)
    #                     outdata[j, i] = idw_v
    #         if min_value is None:
    #             pass
    #         else:
    #             outdata[outdata < min_value] =min_value
    #         if max_value is None:
    #             pass
    #         else:
    #             outdata[outdata > max_value] = max_value
    #         out_band = out_ds.GetRasterBand(1)
    #         out_band.WriteArray(outdata)
    #         if outfile is not None:
    #             outds = gdal.Warp(outfile, out_ds, format="GTiff", width=dst_cols, height=dst_rows,
    #                                   resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
    #         else:
    #             outds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
    #                                   resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
    #         return outds
    #     finally:
    #         ds = None


    @staticmethod
    def idwInterProcess(data=None, latdata=None, londata=None,
                        outfile=None,
                        boundary_range=None, dst_epsg=None,
                        dst_rows=None, dst_cols=None,
                        radius_dist=1.0, min_num=10,
                        min_value=None, max_value=None,
                        first_size=1500):
        """
        反距离插值
        :param data: array，数据数组
        :param latdata: array, 纬度数组
        :param londata: array， 经度数据
        :param outfile: str,结果路径
        :param boundary_range: list, 插值范围
        :param dst_rows: float, 行数
        :param dst_cols: float, 列数
        :param dst_epsg: int, 输出坐标系,epsg编码
        :param radius_dist: float, 插值时的检索半径
        :param min_num: int, 最小点的个数
        :param max_value: float, 插值结果上限值
        :param min_value: float, 插值结果下限值
            first_size：用于控制输出栅格的初始大小，避免内存溢出
        :return:
        """
        try:
            if min_num > data.size:
                min_num = data.size
            if (dst_cols > first_size) | (dst_rows > first_size):
                if dst_cols > dst_rows:
                    cols = first_size
                    rows = int(np.ceil(dst_rows / (dst_cols / first_size)))
                else:
                    rows = first_size
                    cols = int(np.ceil(dst_cols / (dst_rows / first_size)))
            else:
                cols = int(dst_cols)
                rows = int(dst_rows)
            # cols = int(dst_cols)
            # rows = int(dst_rows)
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
            outdata = np.full((rows, cols), 0, dtype=np.float32)
            driver = gdal.GetDriverByName("MEM")
            out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            # if outfile is None:
            #     driver = gdal.GetDriverByName("MEM")
            #     out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            # else:
            #     driver = gdal.GetDriverByName("GTiff")
            #     out_ds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())
            xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
            ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)
            dist = radius_dist ** 2  # 检索范围，经纬度
            for i in range(cols):
                lon = xmaps[i]
                dist_x = (londata - lon) ** 2
                for j in range(rows):
                    lat = ymaps[j]
                    dist_y = (latdata - lat) ** 2
                    dist_point = dist_x + dist_y
                    array_index = dist_point < dist
                    if np.sum(array_index) < min_num:
                        dist_copy = dist_point.copy()
                        dist_copy.sort()
                        array_index = dist_point <= dist_copy[min_num - 1]
                    lons = londata[array_index]
                    lats = latdata[array_index]
                    values = data[array_index]
                    if np.sum(values == values[0]) == values.size:
                        outdata[j, i] = values[0]
                    # if np.sum(values) == values[0]*values.size:
                    #     outdata[j, i] = values[0]
                    else:
                        dist_all = 1.0 / np.sqrt((lat - lats) ** 2 + (lon - lons) ** 2)
                        dist_sum = np.sum(dist_all)
                        idw_v = np.sum((dist_all / dist_sum) * values)
                        outdata[j, i] = idw_v
            if min_value is None:
                pass
            else:
                outdata[outdata < min_value] = min_value
            if max_value is None:
                pass
            else:
                outdata[outdata > max_value] = max_value
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outdata)
            if outfile is not None:
                outds = gdal.Warp(outfile, out_ds, format="GTiff", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            else:
                outds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            return outds
        finally:
            ds = None


    @staticmethod
    def krigingInterProcess(data=None, latdata=None, londata=None,
                            outfile=None,
                            boundary_range=None, dst_epsg=None,
                            dst_rows=None, dst_cols=None,
                            radius_dist=1.0, min_num=10,
                            min_value=None, max_value=None,first_size=1500):
        """
        克里金插值
        :param data: array，数据数组
        :param latdata: array, 纬度数组
        :param londata: array，经度数据
        :param outfile: str, 结果路径
        :param boundary_range: list, 插值范围
        :param dst_rows: float, 行数
        :param dst_cols: float, 列数
        :param dst_epsg: int, 输出坐标系, epsg编码
        :param radius_dist: float, 插值时的检索半径
        :param min_num: int, 最小点的个数
        :param max_value: float, 插值结果上限值
        :param min_value: float, 插值结果下限值
        :return:
        """
        try:
            if min_num > data.size:
                min_num = data.size
            if (dst_cols > first_size) | (dst_rows > first_size):
                if dst_cols > dst_rows:
                    cols = first_size
                    rows = int(np.ceil(dst_rows / (dst_cols / first_size)))
                else:
                    rows = first_size
                    cols = int(np.ceil(dst_cols / (dst_rows / first_size)))
            else:
                cols = int(dst_cols)
                rows = int(dst_rows)
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows

            outdata = np.full((rows, cols), np.nan, dtype=np.float32)

            # 创建克里金插值模型
            OK = OrdinaryKriging(
                londata,
                latdata,
                data,
                variogram_model='spherical',  # 可选模型：'linear', 'power', 'gaussian', 'spherical'
                verbose=False,
                enable_plotting=False
            )

            # 生成网格点
            xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
            ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)

            # 执行克里金插值
            outdata, ss = OK.execute('grid', xmaps, ymaps)

            # 应用上下限值
            if min_value is not None:
                outdata[outdata < min_value] = min_value
            if max_value is not None:
                outdata[outdata > max_value] = max_value

            # 创建输出数据集
            driver = gdal.GetDriverByName("MEM")
            out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())

            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outdata)

            # 保存到文件
            if outfile is not None:
                outds = gdal.Warp(outfile, out_ds, format="GTiff", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            else:
                outds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)

            return outds
        finally:
            out_ds = None


    @staticmethod
    def krigeInterWay(data=None, latdata=None, londata=None,
                      outfile=None, mask_ds=None,
                     boundary_range=None, dst_epsg=None,
                     dst_rows=None, dst_cols=None,
                     radius_dist=1.0, min_num=5,
                     min_value=None, max_value=None,
                     dst_nodata=-999, mask_nodata=0):
        """
        克里金插值
        :param data: array，数据数组
        :param latdata: array, 纬度数组
        :param londata: array， 经度数据
        :param outfile: str,结果路径
        :param mask_ds: 文件或文件对象
        :param boundary_range: list, 插值范围
        :param dst_rows: float, 行数
        :param dst_cols: float, 列数
        :param dst_epsg: int, 输出坐标系,epsg编码
        :param radius_dist: float, 插值时的检索半径
        :param min_num: int, 最小点的个数
        :param max_value: float, 插值结果上限值
        :param min_value: float, 插值结果下限值
        :return:
        """
        try:
            if min_num > data.size:
                min_num = data.size
            cols = int(dst_cols)
            rows = int(dst_rows)
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
            driver = gdal.GetDriverByName("MEM")
            out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())
            dist = radius_dist ** 2  # 检索范围，经纬度
            out_band = out_ds.GetRasterBand(1)
            try:
                mask_data = mask_ds.ReadAsArray()
            except:
                mask_ds = gdal.Open(mask_ds)
                mask_data = mask_ds.ReadAsArray()
            # X方向分块
            xblocksize = 3
            # Y方向分块
            yblocksize = 3
            for i in tqdm(range(0, rows, yblocksize)):
                if i + yblocksize < rows:
                    numrows = yblocksize
                else:
                    numrows = rows - i
                ymaps = np.linspace(y_max + y_cell / 2.0 + i * y_cell,y_max + y_cell / 2.0 + (i + numrows - 1) * y_cell, numrows,endpoint=True)
                ymaps = ymaps.reshape(ymaps.size, 1)
                dist_y = (latdata - ymaps) ** 2
                for j in range(0, cols, xblocksize):
                    if j + xblocksize < cols:
                        numcols = xblocksize
                    else:
                        numcols = cols - j
                    m_data = mask_data[i:i+yblocksize, j:j+xblocksize]
                    if np.sum(m_data==mask_nodata)==m_data.size:
                        z = np.full((yblocksize, xblocksize), dst_nodata)
                    # if np.sum(m_data)==0:
                    #     z = np.full((yblocksize, xblocksize), nodata)
                    else:
                        xmaps = np.linspace(x_min + x_cell / 2.0 + j * x_cell,x_min + x_cell / 2.0 + (j + numcols - 1) * x_cell, numcols,endpoint=True)
                        xmaps = xmaps.reshape(xmaps.size, 1)
                        dist_x = (londata - xmaps) ** 2
                        dist_point = dist_x + dist_y  # 两点之间距离平方
                        dist_point = np.max(dist_point, axis=0)
                        array_index = dist_point < dist
                        if np.sum(array_index) < min_num:
                            dist_copy = dist_point.copy()
                            dist_copy.sort()
                            array_index = dist_point <= dist_copy[min_num-1]
                        lons = londata[array_index]
                        lats = latdata[array_index]
                        values = data[array_index]
                        if np.sum(values == values[0]) == values.size:
                            z = np.full((yblocksize, xblocksize), values[0])
                        # if np.sum(values) == 0:
                        #     z = np.full((yblocksize, xblocksize), 0)
                        else:
                            ok_obj = OrdinaryKriging(lons, lats, values,
                                                     variogram_model="spherical",  # spherical
                                                     variogram_parameters=None, weight=True,
                                                     verbose=False, enable_plotting=False,
                                                     coordinates_type="geographic", nlags=6)
                            z, ss = ok_obj.execute("grid", xmaps, ymaps, backend="vectorized")
                            if min_value is None:
                                pass
                            else:
                                z[z < min_value] = min_value
                            if max_value is None:
                                pass
                            else:
                                z[z > max_value] = max_value
                    z[m_data==mask_nodata] = dst_nodata
                    out_band.WriteArray(z, j, i)
            out_band.SetNoDataValue(dst_nodata)
            if outfile is not None:
                outds = gdal.Warp(outfile, out_ds, format="GTiff", width=dst_cols, height=dst_rows,
                                      resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            else:
                outds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                      resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            return outds
        finally:
            ds = None
            out_ds = None

    @staticmethod
    def maskRasterByRaster(infile, maskfile, outfile, mask_nodata=None, dst_nodata=None, srs_nodata=None):
        """
        数据掩膜
        :param infile: 文件或文件对象
        :param maskfile: 文件或文件对象
        :param outfile: str,结果文件
        :param mask_nodata: float， 掩膜的无效值
        :param dst_nodata: float, 输出的无效值
        :return:
        """
        try:
            try:
                inds = gdal.Open(infile)
            except:
                inds = infile
            indata = inds.ReadAsArray()
            cols = inds.RasterXSize
            rows = inds.RasterYSize
            geo = inds.GetGeoTransform()
            proj = inds.GetProjection()
            band = inds.GetRasterBand(1)
            if srs_nodata is not None:
                pass
            else:
                srs_nodata = band.GetNoDataValue()
            indata[indata == srs_nodata] = dst_nodata
            try:
                maskds = gdal.Open(maskfile)
            except:
                maskds = maskfile
            maskdata = maskds.ReadAsArray()
            band_mask = maskds.GetRasterBand(1)
            if mask_nodata is not None:
                pass
            else:
                mask_nodata = band_mask.GetNoDataValue()
            indata[maskdata==mask_nodata] = dst_nodata
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                out_ds = driver.Create("", cols, rows, 1, band.DataType)
            else:
                driver = gdal.GetDriverByName("Gtiff")
                out_ds = driver.Create(outfile, cols, rows, 1, band.DataType)
            out_ds.SetGeoTransform(geo)
            out_ds.SetProjection(proj)
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(indata)
            out_band.SetNoDataValue(dst_nodata)
            return out_ds
        finally:
            inds =None
            maskds = None

    @staticmethod
    def rasterDivideRank(infile=None, outfile=None, method=None, rank_field=None,
                         rank_content=None, min_value_field=None,max_value_field=None,
                         nodata=None, dst_nodata=0, data_min=False):
        """

        栅格数据等级划分
        :param infile: str, 分级前文件
        :param outfile: str, 分级后文件
        :param method: str, 分级方法
        :param rank_field: str，分级字段
        :param rank_content: dict，分级的字典
        :param min_value_field: list，分级时的最小值列表
        :param max_value_field: list，分级时的最大值列表
        :param nodata: float, infile的填充值
        :param dst_nodata:结果数据填充值
        :return:
        """
        try:
            ds = gdal.Open(infile)
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            data = ds.ReadAsArray()
            rank_numbers = len(rank_content[rank_field])
            rankdata = np.full(data.shape, rank_numbers, dtype=np.int)
            if data_min:
                data_min = np.min(data[data != nodata])
                newdata = data[(data != nodata) & (data != data_min)]
            else:
                newdata = data[(data != nodata)]
            # newdata = data[(data!=nodata)&(data!=data_min)]
            if newdata.size==0:
                pass
            else:
                if method=="PRM":
                    for i in range(rank_numbers):
                        rank = int(rank_content[rank_field][i])
                        rank_min = rank_content[min_value_field][i]
                        rank_max = rank_content[max_value_field][i]
                        if (rank_min is None) & (rank_max is not None):
                            max_value = np.nanpercentile(newdata, int(rank_max))
                            rankdata[(data < max_value)] = rank
                        if (rank_min is not None) & (rank_max is None):
                            min_value = np.nanpercentile(newdata, int(rank_min))
                            rankdata[(data >= min_value)] = rank
                        if (rank_min is None) & (rank_max is None):
                            pass
                        if (rank_min is not None) & (rank_max is not None):
                            max_value = np.nanpercentile(newdata, int(rank_max))
                            min_value = np.nanpercentile(newdata, int(rank_min))
                            if rank_max >= 100:
                                rankdata[(data >= min_value) & (data <= max_value)] = rank
                            else:
                                rankdata[(data >= min_value) & (data < max_value)] = rank
                elif method=="SDM":
                    mean_h = np.nanmean(newdata)
                    s = np.nanstd(newdata)
                    for i in range(rank_numbers):
                        rank = int(rank_content[rank_field][i])
                        rank_min = rank_content[min_value_field][i]
                        rank_max = rank_content[max_value_field][i]
                        if (rank_min is None) & (rank_max is not None):
                            rankdata[(data < (float(rank_max) * s + mean_h))] = rank
                        if (rank_min is not None) & (rank_max is None):
                            rankdata[(data >= (float(rank_min) * s + mean_h))] = rank
                        if (rank_min is None) & (rank_max is None):
                            pass
                        if (rank_min is not None) & (rank_max is not None):
                            rankdata[(data >= (float(rank_min) * s + mean_h)) & (data < (float(rank_max) * s + mean_h))] = rank
                else:
                    rank_numbers = len(rank_content[rank_field])
                    # 自然断点法输出分段阈值
                    if newdata.size<100000:
                        pass
                    else:
                        newdata = nd.interpolation.zoom(newdata, 100000/newdata.size, order=0)
                    thresholds = list(jenkspy.jenks_breaks(newdata, rank_numbers))
                    thresholds.sort(reverse=True)
                    rankdata = np.zeros(data.shape, dtype=np.int)
                    for i in range(len(thresholds) - 1):
                        if i == 0:
                            rankdata[(data >= thresholds[i + 1])] = rank_content[rank_field][i]
                        else:
                            if i == len(thresholds) - 2:
                                rankdata[(data < thresholds[i])] = rank_content[rank_field][i]
                            else:
                                rankdata[(data < thresholds[i]) & (data >= thresholds[i + 1])] = rank_content[rank_field][i]
            rankdata[data == nodata] = dst_nodata
            driver = gdal.GetDriverByName("Gtiff")
            outds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Byte)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(rankdata)
            outband.SetNoDataValue(dst_nodata)
        finally:
            ds = None
            outds = None

    @staticmethod
    def statisFuction(regionfile=None, datafile=None, shpfile=None, field=None, srs_nodata=-999):
        """
        栅格分区统计
        :param regionfile: 文件或文件对象
        :param datafile: str,数据文件
        :param shpfile: str, 矢量文件
        :param field: str,统计后字段
        :param srs_nodata:float,数据中的无效值
        :return:
        """
        try:
            shpds = gpd.read_file(shpfile, encoding="utf-8")
            stat = np.full(shpds.shape[0], 0, dtype=np.float32)
            try:
                regionds = gdal.Open(regionfile)
            except:
                regionds = regionfile
            r_data = regionds.ReadAsArray()
            ds = gdal.Open(datafile)
            data = ds.ReadAsArray()
            for i in range(shpds.shape[0]):
                s_r_data = data[r_data == i + 1]
                stat[i] = np.nanmean(s_r_data[s_r_data != srs_nodata])
            mask = np.isnan(stat)
            if np.sum(mask)>0:
                stat[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), stat[~mask])
            stat[np.isnan(stat)] = 0
            shpds[field] = stat
            shpds.to_file(shpfile,  driver="ESRI Shapefile", encoding="utf-8")
            return stat
        finally:
            shpds = None
            ds = None
            regionds = None

    @staticmethod
    def readDayFile(filelist=None, outfile=None, start_time=None, end_time=None, dts_nodata=-999):
        """
        波段数据合成
        :param filelist: list, 文件列表
        :param outfile: str, 输出文件路径
        :param start_time: str, 开始时间
        :param end_time: str, 结束时间
        :param dts_nodata: float, 目标无效值
        :return:
        """
        try: 
            ds = gdal.Open(filelist[0])
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            # band = ds.GetRasterBand(1)   #会报错，有些文件band可能没有定义
            year = int(start_time[0:4])
            month = int(start_time[4:6])
            day = int(start_time[6:8])
            starttime = pd.PeriodIndex(year=[year], month=[month], day=[day], freq="D")[0]
            year = int(end_time[0:4])
            month = int(end_time[4:6])
            day = int(end_time[6:8])
            endtime = pd.PeriodIndex(year=[year], month=[month], day=[day], freq="D")[0]
            days = (endtime - starttime).delta.days+1
            datelist = []
            for file in filelist:
                filename = os.path.basename(file)
                datelist.append(re.search(r"\d{8}", filename).group())
            outdata = []
            outdatelist = []
            for day in range(days):
                time_str = (starttime + day).strftime('%Y%m%d')
                if time_str in datelist:
                    index_ = datelist.index(time_str)
                    file = filelist[index_]
                    ds = gdal.Open(file)
                    data = ds.ReadAsArray()
                    outdata.append(data)
                    outdatelist.append(datelist[index_])
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                outds = driver.Create("", cols, rows, len(outdata), gdal.GDT_Float32)
            else:
                driver = gdal.GetDriverByName("gtiff")
                outds = driver.Create(outfile, cols, rows, len(outdata), gdal.GDT_Float32)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            for i in range(len(outdata)):
                outband = outds.GetRasterBand(i + 1)
                outband.WriteArray(outdata[i])
                outband.SetNoDataValue(dts_nodata)
            return outds, outdatelist
        finally:
            ds = None

    @staticmethod
    def get_geoinfo_tif(filename):
        '''
        提取经纬度信息
        :param filename: 文件名
        :return: 经度/维度 二维数组
        '''
        try:
            dataset = gdal.Open(filename)
            im_width = dataset.RasterXSize  # 栅格矩阵的列数
            im_height = dataset.RasterYSize  # 栅格矩阵的行数
            im_bands = dataset.RasterCount  # 波段数
            im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
            im_proj = dataset.GetProjection()  # 获取投影信息
            x, y = np.meshgrid(np.arange(im_width), np.arange(im_height))
            lon = im_geotrans[0] + x * im_geotrans[1] + y * im_geotrans[2]
            lat = im_geotrans[3] + x * im_geotrans[4] + y * im_geotrans[5]
            return lon, lat
        finally:
            ds = None



    @staticmethod
    def resamplingMethod(infile=None, outfile=None,
                         dst_cols=None, dst_rows=None,
                         boundary_range=None, nodata=-999):
        """

        :param infile: 文件路径或文件对象
        :param outfile: 结果文件
        :param dst_cols: int, 列
        :param dst_rows: int, 行
        :param boundary_range: truple, 四至范围
        :param nodata: 数据中的无效值
        :return:
        """
        try:
            if outfile is not None:
                outds = gdal.Warp(outfile,infile, format="GTiff", width=dst_cols, height=dst_rows,
                                  srcNodata=nodata, dstNodata=nodata,
                                  resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            else:
                outds = gdal.Warp("", infile, format="MEM", width=dst_cols, height=dst_rows,
                                  srcNodata=nodata, dstNodata=nodata,
                                  resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            return outds
        finally:
            infile = None

    @staticmethod
    def clipRasterFromShp(shpfile, inraster, outraster, nodata):
        """
        :param shpfile: str, 矢量文件路径
        :param inraster: str, 裁剪前的栅格数据
        :param outraster: str, 裁剪后的栅格数据
        :param nodata: float, 无效值
        :return:
        @Version<1> LYB 2020-04-23：Created
        """
        try:
            warp_parameters = gdal.WarpOptions(format='GTiff',
                                               cutlineDSName=shpfile,
                                               cropToCutline=True,
                                               dstNodata=nodata)
            gdal.Warp(outraster, inraster, options=warp_parameters)

        finally:
            pass

    @staticmethod
    def polygonizeTheRaster(infile, outfile, dst_nodata=0, dst_fieldname='class'):
        """
        栅格转矢量
        :param infile: str, 栅格文件
        :param outfile: str, 结果文件
        :param dst_nodata: int,数据中的无效值
        :param dst_fieldname: str, 转换字段
        :return:
        """
        try:
            ds = gdal.Open(infile)
            srcband = ds.GetRasterBand(1)
            maskband = srcband.GetMaskBand()
            drv = ogr.GetDriverByName('ESRI Shapefile')
            dst_ds = drv.CreateDataSource(outfile)
            prj = osr.SpatialReference()
            prj.ImportFromWkt(ds.GetProjection())
            dst_layername = os.path.splitext(os.path.basename(infile))[0]
            dst_layer = dst_ds.CreateLayer(dst_layername, srs=prj, geom_type=ogr.wkbPolygon)
            fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
            dst_layer.CreateField(fd)
            options = []
            # 参数  输入栅格图像波段\掩码图像波段、矢量化后的矢量图层、需要将DN值写入矢量字段的索引、算法选项、进度条回调函数、进度条参数
            gdal.FPolygonize(srcband, maskband, dst_layer,  dst_nodata, options)
        finally:
            ds = None

    @staticmethod
    def rasterReProject(infile=None, outfile=None, dst_epsg=None, nodata=None):
        """
        栅格数据投影转换
        :param infile: str, 输入数据
        :param outfile: str, 输出数据
        :param dst_epsg: int epsg
        :param nodata: float, 无效值
        :return:
        """
        try:
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            dstsrs = out_srs.ExportToWkt()
            if outfile is None:
                warp_parameters = gdal.WarpOptions(format='MEM',
                                                   dstSRS=dstsrs,
                                                   srcNodata=nodata,
                                                   dstNodata=nodata)
                outds = gdal.Warp("", infile, options=warp_parameters)
            else:
                warp_parameters = gdal.WarpOptions(format='GTIFF',
                                                   dstSRS=dstsrs,
                                                   srcNodata=nodata,
                                                   dstNodata=nodata)
                outds = gdal.Warp(outfile, infile, options=warp_parameters)
            return outds
        finally:
            pass

    @staticmethod
    def alignRePrjRmp(file1, file2, outfile, srs_nodata=None, dst_nodata=None,
                      resample_type=gdal.GRA_NearestNeighbour):
        '''
        根据参考影像，进行重采样，包含重投影，行列一致
        :param file1: str, 基准参考影像
        :param file2: str, 需要修改的栅格影像
        :param outfile: str, 结果输出文件
        :param nodata:float,无效值
        :param resample_type:ds，重采样方法
        :return:
        @version <1> LiYaobin 2019-10-08  : created
        '''
        try:
            try:
                refnds = gdal.Open(file1)
            except:
                refnds = file1
            base_xsize = refnds.RasterXSize
            base_ysize = refnds.RasterYSize
            base_proj = refnds.GetProjection()
            base_gt = refnds.GetGeoTransform()
            xmin = base_gt[0]
            ymax = base_gt[3]
            xmax = xmin + base_xsize * base_gt[1]
            ymin = ymax + base_ysize * base_gt[5]
            try:
                warpds = gdal.Open(file2)
            except:
                warpds = file2
            if srs_nodata is None:
                band = warpds.GetRasterBand(1)
                srs_nodata = band.GetNoDataValue()
            if outfile is None:
                dataset = gdal.Warp("",
                                    warpds,
                                    width=base_xsize,
                                    height=base_ysize,
                                    srcNodata=srs_nodata,
                                    dstNodata=dst_nodata,
                                    dstSRS=str(base_proj),
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    format="MEM",
                                    resampleAlg=resample_type)
            else:
                dataset = gdal.Warp(outfile,
                                    warpds,
                                    width=base_xsize,
                                    height=base_ysize,
                                    srcNodata=srs_nodata ,
                                    dstNodata=dst_nodata,
                                    dstSRS=str(base_proj),
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    format="GTiff",
                                    resampleAlg=resample_type)
            return dataset
        finally:
            warpds = None
            refnds = None

    @staticmethod
    def bandLayerStacking(outfile=None, filelist=None):
        """
        对多个单波段数据进行波段合成
        :param outfile: str, 波段合成数据
        :param filelist: list, 文件列表
        :return:
        """
        try:
            try:
                ds = gdal.Open(filelist[0])
            except:
                ds = filelist[0]
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            band = ds.GetRasterBand(1)
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                outds = driver.Create("", cols, rows, len(filelist), band.DataType)
            else:
                driver = gdal.GetDriverByName("gtiff")
                outds = driver.Create(outfile, cols, rows, len(filelist), band.DataType)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            i = 1
            for file in filelist:
                outband = outds.GetRasterBand(i)
                try:
                    ds = gdal.Open(file)
                except:
                    ds = file
                data = ds.ReadAsArray()
                outband.WriteArray(data)
                i = i+1
            return outds
        finally:
            ds = None

    @staticmethod
    def bandMean(outfile=None, infile=None, srs_nodata=None, dst_nodata=None, log_flag=None):
        """
        多波段数据求均值
        :param outfile:str, 结果文件
        :param infile:str,输入文件
        :param nodata：float,无效值
        :return:
        """
        try:
            try:
                ds = gdal.Open(infile)
            except:
                ds = infile
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            bands = ds.RasterCount
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                outds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            else:
                driver = gdal.GetDriverByName("gtiff")
                outds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            data = ds.ReadAsArray()
            band = ds.GetRasterBand(1)
            if srs_nodata is None:
                srs_nodata = band.GetNoDataValue()
            if bands==1:
                pass
            else:
                copy_data = deepcopy(data[:,0, 0])
                num_nodata = data!= srs_nodata
                num_sum = np.sum(num_nodata, axis=0)
                data[data==srs_nodata] = 0
                data = np.sum(data, axis=0)
                data = data/num_sum
                data[np.isnan(data)] = 0
                data[data==np.inf] = 0
                data[copy_data==-999] = -999
                # data[data==srs_nodata] = 0
            if log_flag:
                if (data<10).all():
                    outdata = data
                else:
                    avg_data = np.mean(data[(data != -999)&(data != 0)])
                    std_data = np.nanstd(data[(data != -999)&(data != 0)])
                    a = avg_data + 0.5 * std_data
                    outdata = deepcopy(data)
                    outdata[data >= a] = np.log(data[data >= a] / a) + 1
                    outdata[data < a] = data[data < a] / a
                    outdata[np.isnan(outdata)] =0
                    outdata[outdata==np.inf] = 0
                    outdata[outdata <= 0] = 0
                    outdata[data == -999] = -999
            else:
                outdata = data
            # outdata[outdata<=0] =0
            # outdata[outdata == srs_nodata] = 0
            outband = outds.GetRasterBand(1)
            outband.WriteArray(outdata)
            outband.SetNoDataValue(dst_nodata)
            return outds
        finally:
            ds = None

    @staticmethod
    def dataNormal(infile=None, outfile=None, nodata=-999):
        """
        栅格数据归一化
        :param infile:str,输入文件
        :param outfile：str,结果文件
        :param nodata: float,无效值
        :return:
        """
        try:
            try:
                ds = gdal.Open(infile)
            except:
                ds = infile
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            data = ds.ReadAsArray()
            band = ds.GetRasterBand(1)
            srs_nodata = band.GetNoDataValue()
            outdata = np.full((rows, cols), nodata, dtype=np.float)
            max_value = np.max(data[(data!=srs_nodata)&(data!=0)])
            min_value = np.min(data[(data!=srs_nodata)&(data!=0)])
            # outdata = data
            outdata[(data!=srs_nodata)&(data!=0)] = (data[(data!=srs_nodata)&(data!=0)]-min_value)/(max_value-min_value)*0.95+0.05
            outdata[(data == 0)] =0
            outdata[np.isnan(outdata)] = 0
            outdata[outdata == np.inf] = 0
            # outdata[data != srs_nodata] = (data[data != srs_nodata] - min_value) / (max_value - min_value)
            # outdata[np.isnan(outdata)] = 0
            # outdata[outdata == np.inf] = 0
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                outds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            else:
                driver = gdal.GetDriverByName("GTIFF")
                outds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(outdata)
            outband.SetNoDataValue(nodata)
            return outds
        finally:
            ds = None

    @staticmethod
    def dataNormal2(infile=None, outfile=None, nodata=-999, a_p=0.1):
        """
        栅格数据归一化
        :param infile:str,输入文件
        :param outfile：str,结果文件
        :param nodata: float,无效值
        :return:
        """
        try:
            try:
                ds = gdal.Open(infile)
            except:
                ds = infile
            cols = ds.RasterXSize
            rows = ds.RasterYSize
            geo = ds.GetGeoTransform()
            proj = ds.GetProjection()
            data = ds.ReadAsArray()
            band = ds.GetRasterBand(1)
            srs_nodata = band.GetNoDataValue()
            outdata = np.full((rows, cols), nodata, dtype=np.float)
            if np.sum(data[(data != srs_nodata) & (data != 0)]) == 0:
                outdata[data == 0] = a_p
            else:
                max_value = np.max(data[(data != srs_nodata) & (data != 0)])
                min_value = np.min(data[(data != srs_nodata) & (data != 0)])
                outdata[data != srs_nodata] = (data[data != srs_nodata] - min_value) / (max_value - min_value) * (1 - a_p) + a_p
                outdata[np.isnan(outdata)] = a_p
                outdata[outdata == np.inf] = a_p
                outdata[data == 0] = a_p

            # outdata[data != srs_nodata] = (data[data != srs_nodata] - min_value) / (max_value - min_value)
            # outdata[np.isnan(outdata)] = 0
            # outdata[outdata == np.inf] = 0
            if outfile is None:
                driver = gdal.GetDriverByName("MEM")
                outds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            else:
                driver = gdal.GetDriverByName("GTIFF")
                outds = driver.Create(outfile, cols, rows, 1, gdal.GDT_Float32)
            outds.SetGeoTransform(geo)
            outds.SetProjection(proj)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(outdata)
            outband.SetNoDataValue(nodata)
            return outds
        finally:
            ds = None

    @staticmethod
    def getRasterProject(infile):
        """
        获取栅格数据的坐标信息
        :param infile:
        :return:
        """
        try:
            ds = gdal.Open(infile)
            prj = osr.SpatialReference(wkt=ds.GetProjection())
            epsg_prj = prj.GetAttrValue('AUTHORITY', 1)
            return epsg_prj
        finally:
            ds = None

    @staticmethod
    def DataFilling(infile, outfile):

        try:
            ds = gdal.Open(infile)
        except:
            ds = infile
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()
        band = ds.GetRasterBand(1)
        d_type = gdal.GDT_Float32
        nodata = band.GetNoDataValue()
        data = (ds.ReadAsArray()).astype(np.float32)
        data[data == nodata] = np.nan
        mask = np.isnan(data)
        ind = nd.distance_transform_edt(mask,
                                        return_distances=False,
                                        return_indices=True)
        data = data[tuple(ind)]
        if outfile is None:
            driver = gdal.GetDriverByName("MEM")
            outds = driver.Create("", cols, rows, 1, d_type)
        else:
            driver = gdal.GetDriverByName("GTIFF")
            outds = driver.Create(outfile, cols, rows, 1, d_type)
        outds.SetGeoTransform(geo)
        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(data)
        outband.SetNoDataValue(nodata)
        return outds


    @staticmethod
    def tpsInterProcess(data=None, latdata=None, londata=None,
                        outfile=None,
                        boundary_range=None, dst_epsg=None,
                        dst_rows=None, dst_cols=None,
                        min_num=10,min_value=None, max_value=None,
                        first_size=1500):
        """
        薄板样条插值
        :param data: array，数据数组
        :param latdata: array, 纬度数组
        :param londata: array，经度数据
        :param outfile: str, 结果路径
        :param boundary_range: list, 插值范围
        :param dst_rows: float, 行数
        :param dst_cols: float, 列数
        :param min_num: int, 最小点的个数
        first_size：用于控制输出栅格的初始大小，避免内存溢出。
        :param dst_epsg: int, 输出坐标系, epsg编码
        :return:
        """
        try:
            if min_num > data.size:
                min_num = data.size
            if (dst_cols > first_size) | (dst_rows > first_size):
                if dst_cols > dst_rows:
                    cols = first_size
                    rows = int(np.ceil(dst_rows / (dst_cols / first_size)))
                else:
                    rows = first_size
                    cols = int(np.ceil(dst_cols / (dst_rows / first_size)))
            else:
                cols = int(dst_cols)
                rows = int(dst_rows)
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / rows
            outdata = np.full((rows, cols), np.nan, dtype=np.float32)
            
            # 创建薄板样条插值函数
            tps = Rbf(londata, latdata, data, function='thin_plate', smooth=0)
            
            # 生成网格点
            xmaps = np.linspace(x_min + x_cell / 2.0, (x_min + cols * x_cell) - x_cell / 2.0, cols, endpoint=True)
            ymaps = np.linspace(y_max + (y_cell) / 2.0, (y_max + rows * (y_cell)) - (y_cell) / 2.0, rows, endpoint=True)
            
            # 进行插值
            for i in range(cols):
                for j in range(rows):
                    lon = xmaps[i]
                    lat = ymaps[j]
                    outdata[j, i] = tps(lon, lat)
            
            # 创建输出数据集
            driver = gdal.GetDriverByName("MEM")
            out_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())
            if min_value is None:
                pass
            else:
                outdata[outdata < min_value] =min_value
            if max_value is None:
                pass
            else:
                outdata[outdata > max_value] = max_value
            
            # 写入数据
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(outdata)
            
            # 保存到文件或内存
            if outfile is not None:
                outds = gdal.Warp(outfile, out_ds, format="GTiff", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            else:
                outds = gdal.Warp("", out_ds, format="MEM", width=dst_cols, height=dst_rows,
                                resampleAlg=gdal.GRA_Bilinear, outputBounds=boundary_range)
            
            return outds
        finally:
            ds = None   
                 
    # @staticmethod
    # def maskRasterByRaster(infile, maskfile, outfile, mask_nodata=None, dst_nodata=None, srs_nodata=None):
    #     """
    #     数据掩膜
    #     :param infile: 文件或文件对象
    #     :param maskfile: 文件或文件对象
    #     :param outfile: str,结果文件
    #     :param mask_nodata: float， 掩膜的无效值
    #     :param dst_nodata: float, 输出的无效值
    #     :return:
    #     """
    #     try:
    #         try:
    #             inds = gdal.Open(infile)
    #         except:
    #             inds = infile
    #         indata = inds.ReadAsArray()
    #         cols = inds.RasterXSize
    #         rows = inds.RasterYSize
    #         geo = inds.GetGeoTransform()
    #         proj = inds.GetProjection()
    #         band = inds.GetRasterBand(1)
    #         if srs_nodata is not None:
    #             pass
    #         else:
    #             srs_nodata = band.GetNoDataValue()
    #         indata[indata == srs_nodata] = dst_nodata
    #         try:
    #             maskds = gdal.Open(maskfile)
    #         except:
    #             maskds = maskfile
    #         maskdata = maskds.ReadAsArray()
    #         band_mask = maskds.GetRasterBand(1)
    #         if mask_nodata is not None:
    #             pass
    #         else:
    #             mask_nodata = band_mask.GetNoDataValue()
    #         indata[maskdata == mask_nodata] = dst_nodata
    #         if outfile is None:
    #             driver = gdal.GetDriverByName("MEM")
    #             out_ds = driver.Create("", cols, rows, 1, band.DataType)
    #         else:
    #             driver = gdal.GetDriverByName("Gtiff")
    #             out_ds = driver.Create(outfile, cols, rows, 1, band.DataType)
    #         out_ds.SetGeoTransform(geo)
    #         out_ds.SetProjection(proj)
    #         out_band = out_ds.GetRasterBand(1)
    #         out_band.WriteArray(indata)
    #         out_band.SetNoDataValue(dst_nodata)
    #         return out_ds
    #     finally:
    #         inds = None
    #         maskds = None
  

    @staticmethod
    def raster_to_vector(input_raster, output_vector):
        raster_ds = gdal.Open(input_raster)
        # logging.info(input_raster)
        if raster_ds is None:
            raise ValueError("无法打开输入栅格文件。")

        driver = ogr.GetDriverByName("ESRI Shapefile")
        if driver is None:
            raise ValueError("ESRI Shapefile驱动程序不可用。")

        vector_ds = driver.CreateDataSource(output_vector)
        if vector_ds is None:
            raise ValueError("无法创建输出矢量文件。")

        band = raster_ds.GetRasterBand(1)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(raster_ds.GetProjectionRef())
        layer = vector_ds.CreateLayer("polygon", geom_type=ogr.wkbPolygon, srs=srs)
        field_def = ogr.FieldDefn("value", ogr.OFTInteger)
        layer.CreateField(field_def)

        gdal.Polygonize(band, None, layer, 0)

        ignore_values = [0]
        # 删除ignore_value链表中的类别要素
        if ignore_values is not None:
            for feature in layer:
                class_value = feature.GetField('value')
                for ignore_value in ignore_values:
                    if class_value == ignore_value:
                        # 通过FID删除要素
                        layer.DeleteFeature(feature.GetFID())
                        break

        # 清理并关闭数据集
        raster_ds = None
        vector_ds = None   
        
         
    @staticmethod
    def raster_clip_raster(inraster, inshape, outraster):
        """
        栅格裁剪栅格文件
        :param inraster: 待裁剪栅格文件
        :param inshape: 裁剪范围由指定的栅格文件决定
        :param outraster: 裁剪后的栅格文件
        :return:
        """
        def GetExtent(gt, cols, rows):
            """
            Return list of corner coordinates from a geotransform
            """
            ext = []
            xarr = [0, cols]
            yarr = [0, rows]

            for px in xarr:
                for py in yarr:
                    x = gt[0] + (px * gt[1]) + (py * gt[2])
                    y = gt[3] + (px * gt[4]) + (py * gt[5])
                    ext.append([x, y])
                yarr.reverse()
            return ext
        ds = gdal.Open(inshape)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ext = GetExtent(gt, cols, rows)
        lat = []
        lon = []
        for latlon in ext:
            lon.append(latlon[0])
            lat.append(latlon[1])
        process_cmd = ('gdalwarp -overwrite -t_srs "WGS84" -te ' +
                    str(min(lon)) + ' ' + str(min(lat)) + ' ' + str(max(lon)) + ' ' + str(max(lat)) +
                    ' ' + inraster + ' ' + outraster)
        # print(process_cmd)
        os.system(process_cmd)

    @staticmethod
    def reproject_image_to_master(master, slave, dst_filename=None,
                              res=None, resampleAlg=gdal.GRA_NearestNeighbour):
        slave_ds = gdal.Open(slave)
        # if slave_ds is None:
        #     raise IOError, "GDAL could not open slave file %s " \
        #                    % slave
        slave_proj = slave_ds.GetProjection()
        slave_geotrans = slave_ds.GetGeoTransform()
        data_type = slave_ds.GetRasterBand(1).DataType
        n_bands = slave_ds.RasterCount
        nodata = slave_ds.GetRasterBand(1).GetNoDataValue()

        master_ds = gdal.Open(master)
        # if master_ds is None:
        #     raise IOError, "GDAL could not open master file %s " \
        #                    % master
        master_proj = master_ds.GetProjection()
        master_geotrans = master_ds.GetGeoTransform()
        w = master_ds.RasterXSize
        h = master_ds.RasterYSize
        if res is not None:
            master_geotrans[1] = float(res)
            master_geotrans[-1] = - float(res)
        if dst_filename is None:
            dst_ds = gdal.GetDriverByName('MEM').Create("", w, h, n_bands, data_type)
        else:
            dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, w, h, n_bands, data_type)
        dst_ds.SetGeoTransform(master_geotrans)
        dst_ds.SetProjection(master_proj)
        if nodata is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)

        gdal.ReprojectImage(slave_ds, dst_ds, slave_proj,
                            master_proj, resampleAlg)
        if dst_filename is None:
            return dst_ds
        else:
            dst_ds = None  # Flush to disk
            return dst_filename

    @staticmethod
    def tif_clip_use_shpfile(infile=None,outfile=None,shpfile=None,outputbounds=None,nodata = 0):
        '''
        利用shp文件对tif文件进行裁剪
        '''
        
        if outfile is None:
            format_ = 'MEM'
        else:
            format_ = 'GTiff'

        if shpfile is not None:
            shp = shapefile.Reader(shpfile)
            outputbounds=shp.bbox

            ds = gdal.Warp(outfile,infile, format=format_,
                        outputBounds=outputbounds,
                        cutlineDSName = shpfile,    # 作为数据分割线的矢量
                        cropToCutline=True,         # 按照分割线裁剪数据集
                        # cutlineWhere="FIELD = 'whatever'",
                        dstNodata = nodata )
        elif outputbounds is not None:
            ds = gdal.Warp(outfile,infile, format = format_,
                    outputBounds=outputbounds, dstNodata = nodata)
        else:
            try:
                dataset = gdal.Open(infile)
            except:
                dataset = infile
            #获取地理信息
            img_geotrans = dataset.GetGeoTransform()
            rows = dataset.RasterXSize
            cols = dataset.RasterYSize

            lonmax = img_geotrans[0] + (cols-1)*img_geotrans[1] + (rows-1)*img_geotrans[2]
            lonmin = img_geotrans[0]
            latmax = img_geotrans[3]
            latmin = img_geotrans[3] + (cols-1)*img_geotrans[4] + (rows-1)*img_geotrans[5]

            outputbounds=(lonmin, latmin,lonmax, latmax)
            ds = gdal.Warp(outfile,infile,format = format_,\
                    outputBounds=outputbounds, dstNodata = nodata)

        return ds

    @staticmethod
    def mask_raster_by_grid(infile: str, outfile: str, gridfile: str, resample_alg='bilinear'):
        """
        使用格网文件对栅格数据进行掩膜，确保对齐经纬度和网格数
        
        Parameters:
        -----------
        infile : str
            输入栅格文件路径
        outfile : str  
            输出栅格文件路径
        gridfile : str
            参考格网文件路径
        resample_alg : str
            重采样算法，可选 'nearest', 'bilinear', 'cubic', 等
        """
        try:
            # 打开格网文件
            grid_ds = gdal.Open(gridfile)
            if grid_ds is None:
                raise ValueError(f"无法打开格网文件: {gridfile}")
            
            # 获取格网文件信息
            grid_geo = grid_ds.GetGeoTransform()
            grid_proj = grid_ds.GetProjection()
            grid_xsize = grid_ds.RasterXSize
            grid_ysize = grid_ds.RasterYSize
            
            # 计算范围
            ulx = grid_geo[0]
            uly = grid_geo[3]
            lrx = ulx + grid_geo[1] * grid_xsize
            lry = uly + grid_geo[5] * grid_ysize
            
            # 重采样算法映射
            resample_alg_map = {
                'nearest': gdal.GRA_NearestNeighbour,
                'bilinear': gdal.GRA_Bilinear,
                'cubic': gdal.GRA_Cubic,
                'cubicspline': gdal.GRA_CubicSpline,
                'lanczos': gdal.GRA_Lanczos,
                'average': gdal.GRA_Average,
                'mode': gdal.GRA_Mode
            }
            
            resample_method = resample_alg_map.get(resample_alg, gdal.GRA_Bilinear)
            
            # Warp选项 - 确保与格网文件完全对齐
            warp_options = gdal.WarpOptions(
                format='GTiff',
                outputBounds=[ulx, lry, lrx, uly],
                width=grid_xsize,
                height=grid_ysize,
                dstSRS=grid_proj,
                resampleAlg=resample_method,
                outputType=gdal.GDT_Float32,  # 统一输出类型
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            # 执行Warp
            result_ds = gdal.Warp(outfile, infile, options=warp_options)
            if result_ds is None:
                raise ValueError(f"栅格掩膜失败: {infile} -> {outfile}")
            
            result_ds = None
            grid_ds = None
            
            print(f"成功掩膜栅格: {infile} -> {outfile}")
            return True
            
        except Exception as e:
            print(f"栅格掩膜异常: {str(e)}")
            return False


if __name__ == "__main__":
    pass