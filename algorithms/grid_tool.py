# -*- coding: utf-8 -*-
from osgeo import gdal, ogr, osr
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import shapefile
import rasterio
import time
from .pykrige.ok import OrdinaryKriging
from scipy import interpolate
from scipy.ndimage import sobel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as msee
# from sklearn.metrics import recall_score, accuracy_score, f1_score, roc_auc_score
# from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.interpolate import Rbf
# from common_tool.raster_tool import RasterTool
def clipRasterFromShp(shpfile, inraster, outraster, nodata):#用shp裁剪栅格数据
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
    #print("alignRrprj ======resample")
def save_array_to_tif(array, out_fullpath, proj, gt):
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(out_fullpath), array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])     # GDT_Float32
    # spatial_ref = osr.SpatialReference()
    # spatial_ref.ImportFromEPSG(3857)
    out_ds.SetProjection(proj)     #spatial_ref.ExportToWkt()
    out_ds.SetGeoTransform(gt)
    # (-0.5 * ds.RasterXSize * res_in, res_in, 0, 0.5 * ds.RasterYSize * res_in, 0, -res_in)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(-32768)
    band = None  # save, close
    out_ds = None  # save, close

def calculate_slope(dem_path, slope_path, z_factor=1.0):
    """
    计算坡度并保存为TIFF文件。

    参数:
    dem_path: 输入的DEM文件路径（TIFF格式）
    slope_path: 输出的坡度文件路径（TIFF格式）
    z_factor: 高程单位与水平单位的比例因子，默认为1.0
    """
    # 打开DEM文件
    with rasterio.open(dem_path) as src:
        # 读取高程数据
        elevation = src.read(1)
        # 获取元数据
        meta = src.meta.copy()

    # 计算水平分辨率（假设x和y分辨率相同）
    cell_size = src.res[0]

    # 计算坡度
    # 使用Sobel算子计算水平和垂直方向的梯度
    dz_dx = sobel(elevation, axis=1) / cell_size
    dz_dy = sobel(elevation, axis=0) / cell_size

    # 计算坡度（以度为单位）
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2) * z_factor) * (180 / np.pi)

    # 更新元数据
    meta.update({
        'dtype': slope.dtype,
        'nodata': np.nan
    })

    # 保存坡度数据为TIFF文件
    with rasterio.open(slope_path, 'w', **meta) as dst:
        dst.write(slope, 1) 
def delete_file(file_path):
    try:
        os.remove(file_path)
        #print(f"文件 {file_path} 已被成功删除。")
    except PermissionError as e:
        #print(f"无法删除文件 {file_path}，因为它正在被使用或锁定。错误信息：{e}")
        # 等待一段时间后重试
        time.sleep(8)  # 等待5秒
        # 再次尝试删除
        try:
            os.remove(file_path)
            #print(f"文件 {file_path} 在等待后已被成功删除。")
        except PermissionError as e:
            print(f"文件仍然无法删除。请检查是否有其他程序正在使用该文件。错误信息：{e}")
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
    delete_file(tempfile)

    return outfile
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
    save_array_to_tif(grid_values, outfile, projection, geotransform)
    dataset = None
    grid_data=None
    # os.remove(tempfile)
    
def getColsRows(infile):
    """
    基于格网文件获取数据四至范围和行列
    :param infile: str, 格网文件
    :return:
    """
    print(infile)
    try:
        ds = gdal.Open(infile)
    except:
        ds = infile
    geo = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    x_min = geo[0]
    x_max = geo[0] + cols * geo[1]
    y_max = geo[3]
    y_min = geo[3] + rows * geo[5]
    boundary_range = (x_min, y_min, x_max, y_max)
    # ds = None
    return boundary_range, cols, rows


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
        indata[maskdata == mask_nodata] = dst_nodata
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
        inds = None
        maskds = None


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
    # ok_obj = OrdinaryKriging(lon, lat,
    #                          data,
    #                          variogram_model="spherical",#gaussian  spherical
    #                          variogram_parameters=None,
    #                          weight=True, verbose=False,
    #                          enable_plotting=False,
    #                          coordinates_type="geographic",
    #                          nlags=6)

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
def tif_clip(infile=None,outfile=None,shpfile=None,outputbounds=None,nodata = 0):
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

def rePairIntersection(shpfile):
    """
    修复自相交
    :param shpfile:
    :return:
    """
    ds = gpd.read_file(shpfile)
    for key, geom in ds.geometry.items():
        # geom_ = geom.union(geom)
        geom_ = geom.buffer(0)
        ds.geometry[key] = geom_
    ds.to_file(shpfile, driver="ESRI Shapefile", encoding="utf-8")


def shp_region_select_PAC(in_shp,outfile,filed_name,region_list): 
    #in_shp为输入shp文件； outfile输出文件；filed_name提取字段的名称；file_name输入文件的名称
    gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
    shp = ogr.Open(in_shp,1)   #打开shp文件
    lyr = shp.GetLayer()
    lydefn = lyr.GetLayerDefn()
    spatialref = lyr.GetSpatialRef()  #获取空间坐标系
    geomtype = lydefn.GetGeomType()   #文件类型（point，polyline，polygon等）
    a=[]  #初始化列表
    # b=[]  #初始化列表
    for i,fea in enumerate(lyr):

        feat = lyr.GetFeature(i)
        # x_min, x_max, y_min, y_max=feat.GetGeometryRef().Boundary().GetEnvelope()
        # print(feat.GetGeometryRef().Boundary().GetEnvelope())
        # x_min, x_max, y_min, y_max
        fid = feat.GetField(filed_name) 
        if ((str(fid)[0:2]+"0000") in region_list) | ((str(fid)[0:4]+"00") in region_list)| (str(fid) in region_list):
        # if (x_max>=102) and (y_min<=35):
            a.append(fid)                #获取字段的属性值
    # a = list(set(a))                 #剔除重复的属性值，得到属性值列表
    # print(b[1])

    driver = ogr.GetDriverByName("ESRI Shapefile")   #创建shp驱动
    out_shp = driver.CreateDataSource(outfile)    #创建文件，文件命名为字段属性值+输入的文件名。
    
    outlayer = out_shp.CreateLayer('0', srs=spatialref, geom_type=geomtype ,options=("ENCODING=GBK", ))   
    # outlayer = out_shp.CreateLayer('0', srs=spatialref, geom_type=geomtype ,options=("ENCODING=UTF-8", ))
    # print("GetFieldCount ", lydefn.GetFieldCount())
    # print("GetFeatureCount ", lyr.GetFeatureCount())
    for k in range(0,lydefn.GetFieldCount()):
        fieldDefn = lydefn.GetFieldDefn(k)
        fieldType = fieldDefn.GetType()
        # print("fieldType ", fieldType)
        ret = outlayer.CreateField(fieldDefn, fieldType)
        # print("ret  ", ret)
    outlayerDefn = outlayer.GetLayerDefn()
    # print("outlayerDefn.GetFieldCount() ", outlayerDefn.GetFieldCount())
    
    for i in range(0,lyr.GetFeatureCount()):
        feat = lyr.GetFeature(i)
        fid = feat.GetField(filed_name)
        
        for j in range(len(a)): 
            if fid == a[j]:    #判断属性值等于其中某一个值，提取相应的图层
                outFeature = ogr.Feature(outlayerDefn)
                geom = feat.GetGeometryRef()
                outFeature.SetGeometry(geom)
                
                for k in range(0, outlayerDefn.GetFieldCount()):
                    fieldDefn = outlayerDefn.GetFieldDefn(k)
                    # print(outlayerDefn.GetFieldDefn(k).GetName().encode("gbk"), "  ", feat.GetField(k))
                    outFeature.SetField(outlayerDefn.GetFieldDefn(k).GetName(), feat.GetField(k))
                
                outlayer.CreateFeature(outFeature)
                outFeature = None
    out_shp=None

    
def get_grid_and_shp(region_name,region_code,shpfile,gridfile,demfile,outpath):
    # 准备画图时用于掩膜的栅格数据      
    # 若没有栅格数据则从全国栅格中裁剪 
    
    if region_code[0]=='000000': 
        # mask_grid = gdal.Warp('',gridfile,format='MEM',outputBounds=[102,boundary_range[1],boundary_range[2],25])
        # boundary_range, cols, rows = getColsRows(mask_grid)
        shpfile_cut = shpfile
        gridfile_cut = gridfile
        demfile_cut = os.path.join(outpath,'000000'+'_'+os.path.basename(demfile)) 
        if not os.path.exists(demfile_cut):
            tif_clip(infile=demfile,outfile=demfile_cut,shpfile=shpfile_cut,outputbounds=None,nodata = -32768)        
        
        
    elif (len(region_code)==1) & (int(region_code[0])>820000):
        shpfile_cut = shpfile
        rePairIntersection(shpfile)  
        gridfile_cut = os.path.join(outpath,os.path.basename(gridfile).replace('000000',region_code[0]))  
        demfile_cut = os.path.join(outpath,os.path.basename(demfile).replace('000000',region_code[0])) 
        if not os.path.exists(gridfile_cut):
            tif_clip(infile=gridfile,outfile=gridfile_cut,shpfile=shpfile_cut)
        if not os.path.exists(demfile_cut):
            tif_clip(infile=demfile,outfile=demfile_cut,shpfile=shpfile_cut,outputbounds=None,nodata = -32768)
    else:
        # if '东北' in region_name:
        #     region_list = ['210000','220000','230000']
        # elif '京津冀'  in region_name:
        #     region_list = ['110000','120000','130000']
        # elif '长三角'  in region_name:
        #     region_list = ['310000','320000','330000','340000']
        # elif '长江经济带'  in region_name:
        #     region_list = ['310000','320000','330000','340000']
        # elif '大湾区'  in region_name:
        #     region_list = ['810000','820000','440100','440300','440400','440600','441300','441900']
        # else:
        #     print("该区域无县级矢量数据")
        region_list = region_code

        
        # 从全国文件中裁剪
        if len(region_list)==1:
            gridfile_cut = os.path.join(outpath,os.path.basename(gridfile).replace('000000',region_code[0]))
            shpfile_cut = os.path.join(outpath,os.path.basename(shpfile).replace('000000',region_code[0]))
            demfile_cut = os.path.join(outpath,region_code[0]+'_'+os.path.basename(demfile))
        else:
            gridfile_cut = os.path.join(outpath,os.path.basename(gridfile).replace('000000','999999'))
            shpfile_cut = os.path.join(outpath,os.path.basename(shpfile).replace('000000','999999')) 
            demfile_cut = os.path.join(outpath,'999999_'+os.path.basename(demfile))
        if not os.path.exists(shpfile_cut):
            shp_region_select_PAC(shpfile,shpfile_cut,'PAC',region_list) # PAC中存储了行政区划编码
    
        if not os.path.exists(gridfile_cut):
            tif_clip(infile=gridfile,outfile=gridfile_cut,shpfile=shpfile_cut)
        if not os.path.exists(demfile_cut):
            tif_clip(infile=demfile,outfile=demfile_cut,shpfile=shpfile_cut,outputbounds=None,nodata = -32768)
    return gridfile_cut,shpfile_cut,demfile_cut

def stn_to_tif_RF(data_ds,shpfile_cut,demfile_cut,lulcfile_cut,otherfiles_cut,gridfile_cut,var_name,min_value,max_value,tiffile):
    ml_rf_(data_ds,demfile_cut,lulcfile_cut,otherfiles_cut,gridfile_cut,var_name,min_value,max_value,tiffile)
def stn_to_tif_MLP(data_ds,shpfile_cut,demfile_cut,lulcfile_cut,otherfiles_cut,gridfile_cut,var_name,min_value,max_value,tiffile):
    mlp_(data_ds,demfile_cut,lulcfile_cut,otherfiles_cut,gridfile_cut,var_name,min_value,max_value,tiffile)  
def stn_to_tif_LSM(data_ds,demfile_cut,gridfile_cut,var_name,min_value,max_value,tiffile,block_size=256):
    LSM(data_ds,demfile_cut,gridfile_cut,var_name,min_value,max_value,tiffile)  
def stn_to_tif_LSM_idw(data_ds,demfile_cut,gridfile_cut,dst_epsg,var_name,min_value,max_value,tiffile,block_size=256,radius_dist=2.0, min_num=20, first_size=200):
    LSM_idw(data_ds, demfile_cut,gridfile_cut,dst_epsg, var_name, min_value, max_value, tiffile, block_size=block_size,radius_dist=radius_dist, min_num=min_num, first_size=first_size)  
def stn_to_tif(data_ds, gridfile,tiffile, \
               dst_epsg, interp_method, var_name, min_value=None, max_value=None, nodata=-999, \
               radius_dist=2.0, min_num=20, first_size=200):
    """
    功能:将站点数据插值并输出到tif
    data_ds: 站点结果
    gridfile: 掩膜需要的栅格数据
    dst_epsg: 坐标系, 例如: 4490
    interp_method: 插值方案, 例如:IDW,Krige
    tiffile: 要输出的栅格文件
    dst_epsg: 地理坐标系
    var_name: 要插值的变量名
    min_value,max_value: 插值阈值
    nodata: 缺省值
    """

    # 读取文件

    if type(data_ds) == str:
        try:
            data_ds = pd.read_csv(data_ds, dtype=str, encoding="gbk")
        except:
            data_ds = pd.read_csv(data_ds, dtype=str, encoding="utf-8")
    else:
        data_ds = data_ds

    data_ds = data_ds[~np.isnan(data_ds[var_name].astype(float))]
    data_ds = data_ds[(data_ds[var_name] != nodata)]

    boundary_range, cols, rows = getColsRows(gridfile)

    # 开始数据插值
    # if not os.path.exists(tiffile):
    # obj = RasterTool()
    if interp_method =="idw":
        
        outds = idwInterProcess(data=np.array(data_ds[var_name].astype(float)),
                                latdata=np.array(data_ds['Lat'].astype(float)),
                                londata=np.array(data_ds['Lon'].astype(float)),
                                outfile=None, boundary_range=boundary_range,
                                dst_epsg=dst_epsg, dst_rows=rows, dst_cols=cols,
                                radius_dist=radius_dist, min_num=min_num,
                                min_value=min_value, max_value=max_value, first_size=first_size)
        # 空间掩膜
        outds = maskRasterByRaster(outds, gridfile, tiffile, 0, -999999)
        outds = None
    elif interp_method =="krige":
        outds = krigingInterProcess(data=np.array(data_ds[var_name].astype(float)),
                                    latdata=np.array(data_ds['Lat'].astype(float)),
                                    londata=np.array(data_ds['Lon'].astype(float)),
                                    outfile=None, boundary_range=boundary_range,
                                    dst_epsg=dst_epsg, dst_rows=rows, dst_cols=cols,
                                    radius_dist=radius_dist, min_num=min_num,
                                    min_value=min_value, max_value=max_value,first_size=first_size)
        # 空间掩膜
        outds = maskRasterByRaster(outds, gridfile, tiffile, 0, -999999)
        outds = None
    elif interp_method =="RBF":
        outds = rbfInterProcess(data=np.array(data_ds[var_name].astype(float)),
                                    latdata=np.array(data_ds['Lat'].astype(float)),
                                    londata=np.array(data_ds['Lon'].astype(float)),
                                    outfile=None, boundary_range=boundary_range,
                                    dst_epsg=dst_epsg, dst_rows=rows, dst_cols=cols,
                                    radius_dist=radius_dist, min_num=min_num,
                                    min_value=min_value, max_value=max_value)

        # 空间掩膜
        outds = maskRasterByRaster(outds, gridfile, tiffile, 0, -999999)
        outds = None
        
        
    elif interp_method =="TPS":        #薄板样条插值
        outds=tpsInterProcess(data=np.array(data_ds[var_name].astype(float)),
                                    latdata=np.array(data_ds['Lat'].astype(float)),
                                    londata=np.array(data_ds['Lon'].astype(float)),
                                    outfile=None, boundary_range=boundary_range,
                                    dst_epsg=dst_epsg, dst_rows=rows, dst_cols=cols,
                                    min_num=min_num,min_value=min_value, max_value=max_value,
                                    first_size=first_size)
        
        outds = maskRasterByRaster(outds, gridfile, tiffile, 0, -999999)
        outds = None
        # # 区县级任务
        # region_code_lastnum = int(region_code[-2:])
        # if region_code_lastnum != 0:
        #     dangerIndexParent = os.path.join(os.path.dirname(dangerfile),"dangerIndexParent.tif")
        #     obj.maskRasterByRaster(outds, gridfile_parent, dangerIndexParent, 0, -999)

        #     # 从临时文件里 读取上一次危险性指数，并掩膜
        #     dangerIndexMask = MaskByRaster(dangerIndexParent, outfile=None, boundary_range=boundary_range2, ParaType=None)
        #     outds = obj.maskRasterByRaster(dangerIndexMask, gridfile,dangerfile, 0, -999)



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


def raster_clip_raster(inraster, inshape, outraster):
    """
    栅格裁剪栅格文件
    :param inraster: 待裁剪栅格文件
    :param inshape: 裁剪范围由指定的栅格文件决定
    :param outraster: 裁剪后的栅格文件
    :return:
    """
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


def GetExtent(gt, cols, rows):
    """
    Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
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

if __name__ == '__main__':
    # infile = r'F:\02workprog\22BJNQ\output\1991_2020\temp_folder\sum_tem_0\sum_tem_0.grd'
    # range_file = r'F:\02workprog\22BJNQ\depend_data\000000_30grid.tif'
    # out_range_file = r'F:\02workprog\22BJNQ\output\1991_2020\temp_folder\sum_tem_0\range_sum_tem_0.tif'
    # raster_clip_raster(infile, range_file, out_range_file)
    # mean_tem_file = r'F:\02workprog\22BJNQ\output\1991_2020\≥0℃积温_19910101_20201231.tif'
    # reproject_file = r'F:\02workprog\22BJNQ\output\1991_2020\temp_folder\sum_tem_0\range_reproject_sum_tem_0.tif'
    # reproject_image_to_master(mean_tem_file, out_range_file, dst_filename=reproject_file)

    # tif_file = r'F:\02workprog\22BJNQ\output\1991_2020\region1_19910101_20201231_aunslin.tif'
    # shp_file = r'F:\02workprog\22BJNQ\output\1991_2020\shp_test\region1_19910101_20201231_aunslin.shp'
    # raster_to_vector(tif_file, shp_file)
    data_ds=r"D:\project\农业气候资源普查和区划\code\china\NYQH_ZY\算例\C_PR_A1991_230000_10-00-004_80 - 黑龙江老师提供_同插值方法.csv"
    demfile_cut=r"D:\project\农业气候资源普查和区划\code\china\depend_data\230000_china_dem.tif"
    dst_epsg=4490
    var_name="acc_temp"
    min_value=0
    max_value=None
    tiffile=r"D:/project/农业气候资源普查和区划/code/china/NYQH_ZY/算例/黑龙江老师提供——系统插值方法结果.tif"
    stn_to_tif_LSM_idw(data_ds,demfile_cut,dst_epsg,var_name,min_value,max_value,tiffile,block_size=256,radius_dist=2.0, min_num=20, first_size=200)
