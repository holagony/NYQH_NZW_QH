# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:15:03 2025

@author: HTHT
"""
import os
from qgis.core import QgsProject, QgsRasterLayer,QgsVectorLayer,QgsCoordinateReferenceSystem,QgsMapLayer,QgsWkbTypes,QgsSymbol
from qgis.core import QgsExpressionContextUtils,QgsCoordinateTransform,QgsLayoutExporter,QgsApplication,QgsRuleBasedRenderer
from osgeo import gdal
import shutil
from PyQt5.QtGui import QColor
import numpy as np
from osgeo import gdal,osr
import pandas as pd
import rioxarray as rxr
import geopandas as gpd  

def drawmaps(tiffile, rasterStyle, template, mapinfo):
    # 解决ERROR 6: The PNG driver does not support update access to existing datasets.
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

    # create a reference to the QgsApplication, setting the # second argument to False disables the GUI
    qgs = QgsApplication([], False)
    # load providers
    qgs.initQgis()

    # p1：读取qgs模板文件
    project = QgsProject.instance()
    project.read(template)

    rlayer = QgsRasterLayer(tiffile, "rlayer")
    extent = rlayer.extent()
    project.addMapLayer(rlayer)

    # 修改图层顺序
    root = project.layerTreeRoot()
    rlayerID = root.findLayer(rlayer.id())
    rlayerIDclone = rlayerID.clone()
    rlayerIDparent = rlayerID.parent()
    rlayerIDparent.insertChildNode(5, rlayerIDclone)
    rlayerIDparent.removeChildNode(rlayerID)

    # 渲染图层
    rlayer.loadNamedStyle(rasterStyle)
    rlayer.triggerRepaint()

    # 获取qgs模板文件中的制图模板
    layout = project.layoutManager().layoutByName("layout")
    if layout:
        print("Layout found:", layout.name())  # 打印布局名称来确认布局对象
    else:
        print("No layout found with the name 'layout'.")    
    mapitem = layout.itemById("map1")
    desCRS = mapitem.crs()

    rectangle = layout.itemById("colorbar")
    rectangle.setPicturePath(rasterStyle[:-3] + "png")
    
    sourceCRS = rlayer.crs()
    desCRS = mapitem.crs()
    #transform = QgsCoordinateTransform(sourceCRS, desCRS, QgsProject.instance())
    #map_extent = transform.transformBoundingBox(extent)

    # 设置地图项的范围为模板的完整范围
    #mapitem.setExtent(map_extent)
    mapitem.refresh()

    # 添加标题,图例,卫星名称,分辨率等信息
    for k, v in mapinfo.items():
        if k == "title":
            v = mapinfo["title"]
        layout.itemById(k).setText(v)

    layout.refresh()

    # 输出图片
    pngfile = tiffile.replace("TIFF", "PNG")
    exporter = QgsLayoutExporter(layout)
    exporter.exportToImage(pngfile, QgsLayoutExporter.ImageExportSettings())
    print(pngfile)
    return pngfile

###批量出专题图,适用landsat5/8/9,gf1/6,modis
def drawmaps_(tiffile, rasterStyle, template, code, mapinfo):
    # 解决ERROR 6: The PNG driver does not support update access to existing datasets.
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

    # create a reference to the QgsApplication, setting the # second argument to False disables the GUI
    qgs = QgsApplication([], False)
    # load providers
    qgs.initQgis()

    # p1：读取qgs模板文件
    project = QgsProject.instance()
    project.read(template)

    rlayer = QgsRasterLayer(tiffile, "rlayer")
    project.addMapLayer(rlayer)

    # 修改图层顺序
    root = project.layerTreeRoot()
    rlayerID = root.findLayer(rlayer.id())
    rlayerIDclone = rlayerID.clone()
    rlayerIDparent = rlayerID.parent()
    rlayerIDparent.insertChildNode(5, rlayerIDclone)
    rlayerIDparent.removeChildNode(rlayerID)

    # 渲染图层
    rlayer.loadNamedStyle(rasterStyle)
    rlayer.triggerRepaint()

    # 获取qgs模板文件中的制图模板
    layout = project.layoutManager().layoutByName("layout")

    mapitem = layout.itemById("map1")
    desCRS = mapitem.crs()

    rectangle = layout.itemById("colorbar")
    rectangle.setPicturePath(rasterStyle[:-3] + "png")

    groupname = "xiang"
    # elif str(code) in ['RGUA','RZGH','RNBS']:
    #     groupname = "region"
    # else:
    #     groupname = "nature_reserves"

    # 根据区域等级设置图层组的可见性
    for group in root.findGroups():
        if group.name() == groupname:
            group.setItemVisibilityChecked(True)
        else:
            group.setItemVisibilityChecked(False)

    # 设置工程变量值
    QgsExpressionContextUtils.setProjectVariable(project=project, name='code', value=code)

    # 获得图层组第一个图层,需要注意的是第一个图层和图层组级别要一致,如省图层组第一个图层必须是省图层
    map_layer = root.findGroup(groupname).children()[0].layer()
    # 获得图层范围
    map_layer.selectByExpression("code=%s" % code, QgsVectorLayer.SetSelection)
    # if not selection:
    #     print(f"No features selected with code: {code}")
    #     return None
    extent = map_layer.boundingBoxOfSelected()
    
    # 检查地图范围的宽高比
    # width = extent.width()
    # height = extent.height()
    # if width == 0 or height == 0:
    #     print("Warning: Map extent width or height is zero. This may indicate an invalid selection or geometry.")
    #     # 可以选择跳过后续操作或设置默认值
    #     print(f"Map extent: {extent}")
    #     return None
    
    # print(f"Map extent width: {width}, height: {height}, aspect ratio: {width / height}")
    
    # 转换坐标系
    sourceCRS = map_layer.crs()
    transform = QgsCoordinateTransform(sourceCRS, desCRS, QgsProject.instance())
    map_extent = transform.transformBoundingBox(extent)
    
    # 调整地图范围的宽高比
    # if width / height > 10:  # 如果宽高比大于10，调整比例尺
    #     map_extent.scale(1.7, 1)  # 横向扩展
    # elif height / width > 10:  # 如果高宽比大于10，调整比例尺
    #     map_extent.scale(1, 1.7)  # 纵向扩展
    map_extent.scale(1.7)
    mapitem.zoomToExtent(map_extent)


    # 添加标题,图例,卫星名称,分辨率等信息
    for k, v in mapinfo.items():
        if k == "title":
            v =  mapinfo["title"]
        layout.itemById(k).setText(v)            
    layout.refresh()
    # 输出图片
    basename = os.path.basename(tiffile)
    pngfile = tiffile.replace("TIFF", "PNG")
    exporter = QgsLayoutExporter(layout)
    exporter.exportToImage(pngfile, QgsLayoutExporter.ImageExportSettings())
    print(pngfile)

    return pngfile



def mapinfo_name(year,PACNAME,CROPNAME):

    mapinfo = {"title": PACNAME+CROPNAME+str("分布监测图"),
               "date": year+"年",
               # "sat": f"卫星/传感器: {sat}/{ins}",
               # "resolution": f"空间分辨率: {res}"
               }
    return mapinfo

def crop_tif(regionfile,out_file,crop):
    croplist={"玉米":1,"马铃薯":2,"红薯":3,"水稻":4}
    in_ds = gdal.Open(regionfile, gdal.GA_ReadOnly)
    gt = in_ds.GetGeoTransform()   
    band = in_ds.GetRasterBand(1)
    image_data = band.ReadAsArray().astype(np.int32)
    image_data = np.where(image_data!=croplist[crop],0,image_data)
    save_array_to_tif(image_data, out_file, gt)
    return out_file

#    
def mapj(primaryFile, tempPath,auxPath, L2Name, L2File):

    
    '''
    tempPath周期合成文件路径
    L2File自定义裁剪shp范围
    L2Name自定义裁剪区域名称
    当不进行自定义时，L2File、L2Name为空，默认不进行自定义、直接用宁夏区域裁剪
    auxPath辅助文件所在位置
    通过区域裁剪使所有tiff都变成了宁夏区域大小，然后重采样将分辨率保持一致，proj和gt都用第一个文件的。
    '''
    # primaryFiles 先裁剪第一个作为基础数据   
    out_file1 = os.path.join(tempPath, os.path.basename(primaryFile[0])[:-5] + '_clip.TIFF')
    # L2Name, L2File 为空默认矢量数据为  榆阳区
    if L2Name == "" and L2File == "":
        shpfile = os.path.join(auxPath,"shp", "yuyang_xian.shp")
    else:
        shpfile = L2File
    clipRasterFromShp(shpfile, primaryFile[0], out_file1, nodata=int(0))
    
    # 只有一个文件，无论拼接还是合成都直接输出裁剪后结果
    if len(primaryFile)==1:
        return True,out_file1
    else:
        # 数据的同尺度转换，即根据基础tiff进行重采样
        image_paths = []
        for tif_app_i in primaryFile:
            out_pathfile1 = os.path.join(tempPath, os.path.basename(tif_app_i)[:-5] + '_rererere.TIFF')
            alignRePrjRmp(out_file1, tif_app_i, out_pathfile1, srs_nodata=int(0), dst_nodata=int(0),
                  resample_type=gdal.GRA_NearestNeighbour)

            image_paths.append(out_pathfile1)

        #
        output_file = os.path.join(tempPath, 'lulc_joins.TIFF')
        #tempTIFF = join_mode_NEW(image_paths, output_file,block_size=(500, 500))
        tempTIFF = join_mode(image_paths, output_file)

        if os.path.exists(tempTIFF):
            # 还得裁剪一次
            out_file2 = tempTIFF.replace('.TIFF', '_clip.TIFF')
            # L2Name, L2File 为空默认矢量数据为  宁夏自治区
            if L2Name == "" and L2File == "":
                shpfile = os.path.join(auxPath,"shp", "yuyang_xian.shp")
            else:
                shpfile = L2File
            clipRasterFromShp(shpfile, tempTIFF, out_file2, nodata=int(0))
            return True,out_file2
        else:
            return False,'合成失败，检查join！'    
#求按照众数规则周期合成
def join_mode(image_paths, output_file):
    #土地利用产品输出时无效值按照0输出
    # 打开第一个影像获取行列数
    in_ds = gdal.Open(image_paths[0], gdal.GA_ReadOnly)
    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    proj = in_ds.GetProjection()
    gt = in_ds.GetGeoTransform()

    # 初始化土地利用出现频次和像素数量、频次最大值
    image_count1 = np.zeros((height, width), dtype=np.int32)
    image_count2 = np.zeros((height, width), dtype=np.int32)
    image_count3 = np.zeros((height, width), dtype=np.int32)
    image_count4 = np.zeros((height, width), dtype=np.int32)
    image_count5 = np.zeros((height, width), dtype=np.int32)
    image_count6 = np.zeros((height, width), dtype=np.int32)
    image_count7 = np.zeros((height, width), dtype=np.int32)
    image_count8 = np.zeros((height, width), dtype=np.int32)
    image_count9 = np.zeros((height, width), dtype=np.int32)
    image_count10 = np.zeros((height, width), dtype=np.int32)
    image_count11 = np.zeros((height, width), dtype=np.int32)
    image_count13 = np.zeros((height, width), dtype=np.int32)
    pixel_count = np.zeros((height, width), dtype=np.int32)
    image_mode=np.zeros((height, width), dtype=np.int32)

    # 遍历每个影像
    for image_path in image_paths:
        # 打开影像
        in_dss = gdal.Open(image_path, gdal.GA_ReadOnly)
        band = in_dss.GetRasterBand(1)

        # 读取影像数据
        image_data = band.ReadAsArray().astype(np.int32)
        image_data=np.where(image_data==12,0,image_data)
        
        #image_data0有值为1，无效值为0
        image_data0 = np.where(image_data!=int(0), int(1), image_data)
        # 土地类型为1的标为1其他标为0
        image_data1 = np.where(image_data!= int(1), int(0), 1)
        #土地类型为2的标为1，其他的标为0
        image_data2 = np.where(image_data!=int(2), int(0), 1)
        #土地类型为3的标为1，其他的标为0
        image_data3 = np.where(image_data!= int(3), int(0), 1)
        #土地类型为4的标为1，其他的标为0
        image_data4 = np.where(image_data!= int(4), int(0), 1)
        #土地类型为5的标为1，其他的标为0
        image_data5 = np.where(image_data!= int(5), int(0), 1)
        #土地类型为6的标为1，其他的标为0
        image_data6 = np.where(image_data!= int(6), int(0), 1)
        image_data7 = np.where(image_data!= int(7), int(0), 1)
        image_data8 = np.where(image_data!= int(8), int(0), 1)
        image_data9 = np.where(image_data!= int(9), int(0), 1)  
        image_data10 = np.where(image_data!= int(10), int(0), 1)
        image_data11 = np.where(image_data!= int(11), int(0), 1)
        image_data13 = np.where(image_data!= int(13), int(0), 1)
        #累加影像数据每个点1、2、3、4、5、6出现的次数
        image_count1 += image_data1
        #image_count1=image_count1.chunk((500, 500))
        image_count2 += image_data2
        #image_count2=image_count2.chunk((500, 500))
        image_count3 += image_data3
        #image_count3=image_count3.chunk((500, 500))
        image_count4 += image_data4
        #image_count4=image_count4.chunk((500, 500))
        image_count5 += image_data5
        #image_count5=image_count5.chunk((500, 500))
        image_count6 += image_data6
        #image_count6=image_count6.chunk((500, 500))
        image_count6 += image_data6
        image_count7 += image_data7
        image_count8 += image_data8
        image_count9 += image_data9
        image_count10 += image_data10
        image_count11 += image_data11
        image_count13 += image_data13
        #统计影像每个点位有效数据点数
        pixel_count += image_data0
        #pixel_count=pixel_count.chunk((500, 500))
        # 关闭数据集
        in_dss = None
    
    #in_ds=None
    # 计算众数
    # 使用np.dstack将数组堆叠在一起
    stacked_arrays = np.stack((image_count1, image_count2, image_count3,image_count4,image_count5,image_count6,image_count7,image_count8,image_count9,image_count10,image_count11,image_count13))
    # 使用np.maximum.reduce来求最大值
    max_values = np.max(stacked_arrays, axis=0)
    #为频次最大值赋土地利用类型值，顺序不能变，
    image_mode[image_count13==max_values]=int(13)
    image_mode[image_count6==max_values]=int(6)
    image_mode[image_count9==max_values]=int(9)
    image_mode[image_count8==max_values]=int(8)
    image_mode[image_count7==max_values]=int(7)
    image_mode[image_count3==max_values]=int(3)
    image_mode[image_count11==max_values]=int(11)
    image_mode[image_count10==max_values]=int(10)

    


    image_mode[image_count5==max_values]=int(5)
    image_mode[image_count4==max_values]=int(4)    


    image_mode[image_count2==max_values]=int(2)
    image_mode[image_count1==max_values]=int(1)
    image_mode[pixel_count==0]=int(0)

    save_array_to_tif(image_mode, output_file, gt)

    return output_file

##
def clipRasterFromShp(shpfile, inraster, outraster, nodata):#用shp裁剪栅格数据
    """
    :param shpfile: str, 矢量文件路径
    :param inraster: str, 裁剪前的栅格数据
    :param outraster: str, 裁剪后的栅格数据
    :param nodata: float, 无效值
    """
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    try:
        warp_parameters = gdal.WarpOptions(format='GTiff',
                                            srcSRS=None,  # 源数据的坐标系，None表示自动获取
                                            dstSRS=target_srs,  # 目标数据的坐标系，
                                            cutlineDSName=shpfile,
                                            cropToCutline=True,
                                            dstNodata=nodata,
                                           )
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
    print("alignRrprj ======resample")
def save_array_to_tif(array, out_fullpath, gt):
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(str(out_fullpath), array.shape[1], array.shape[0], 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])     # GDT_Float32
    srs = osr.SpatialReference()  # establish srs by encoding
    srs.ImportFromEPSG(4326)  # WGS84 lat/long
    out_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    out_ds.SetGeoTransform(gt)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(0)
    band = None  # save, close
    out_ds = None  # save, close
def del_file(path):
    """
    删除一个文件夹下的所有文件
    @param path: 删除文件夹
    @return:
    """
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):#如果是文件夹那么递归调用一下
            del_file(c_path)
        else:                    #如果是一个文件那么直接删除
            os.remove(c_path)
    #print('文件已经清空完成')
    
    
def tongjijson_output(tiffile, shpfiles):
    filename=os.path.basename(tiffile)
    items = filename.split("_")
    sat=items[4]#卫星
    ins=items[5]#传感器
    res=items[6]#分辨率
    areaunit=int(res[:4])*int(res[:4])/666.67    ##亩
    gdf_combined = pd.concat([gpd.read_file(shpfile,encoding='UTF-8') for shpfile in shpfiles], ignore_index=True)
    xds = rxr.open_rasterio(tiffile).squeeze()
    levels_list=[]
    for index, row in gdf_combined.iterrows():
        name = str(row['NAME'])
        code = str(row['PAC'])
        geometry = row['geometry']
        # 统计各类面积
        clipped_xds = xds.rio.clip([geometry], gdf_combined.crs, drop=True)
        clipped_xds=clipped_xds.values.astype(np.int32)
        area_1=len(clipped_xds[clipped_xds==1])*areaunit
        area_2=len(clipped_xds[clipped_xds==2])*areaunit
        area_3=len(clipped_xds[clipped_xds==3])*areaunit
        area_4=len(clipped_xds[clipped_xds==4])*areaunit
        area_5=len(clipped_xds[clipped_xds==5])*areaunit
        area_6=len(clipped_xds[clipped_xds==6])*areaunit
        area_7=len(clipped_xds[clipped_xds==7])*areaunit
        area_8=len(clipped_xds[clipped_xds==8])*areaunit        
        area_9=len(clipped_xds[clipped_xds==9])*areaunit
        area_10=len(clipped_xds[clipped_xds==10])*areaunit
        area_11=len(clipped_xds[clipped_xds==11])*areaunit
        area_13=len(clipped_xds[clipped_xds==13])*areaunit        
        #统计区域的总面积
        area_all=sum([area_1,area_2,area_3,area_4,area_5,area_6,area_7,area_8,area_9,area_10,area_11,area_13])
        if area_all==0:
            area_all=0.001
        percent1=area_1/area_all*100
        percent2=area_2/area_all*100
        percent3=area_3/area_all*100
        percent4=area_4/area_all*100    
        lulc_tongji=[{
                "level":1,
                "desc":"玉米",
                "area":f"{area_1:.2f}",
                "percent":f"{percent1:.2f}"
            },
            {
                "level":2,
                "desc":"马铃薯",
                "area":f"{area_2:.2f}",
                "percent":f"{percent2:.2f}"
            },
            {
                "level":3,
                "desc":"红薯",
                "area":f"{area_3:.2f}",
                "percent":f"{percent3:.2f}"
            },
            {
                "level":4,
                "desc":"水稻",
                "area":f"{area_4:.2f}",
                "percent":f"{percent4:.2f}"
            }]
        levels_list.append(
            {"name":name,
             "code":code,
             "levels":lulc_tongji
             })  
    json_out=pd.DataFrame(levels_list)    
    return json_out
def clip_tifs(tiffile, shpfiles):
    #shpfiles:区市县shp路径列表
    filename=os.path.basename(tiffile)
    items = filename.split("_")
    coding=items[2]
    gdf_combined = pd.concat([gpd.read_file(shpfile) for shpfile in shpfiles], ignore_index=True)

    xds = rxr.open_rasterio(tiffile).squeeze()
    tiffiles = []

    for index, row in gdf_combined.iterrows():
        code = str(row['PAC'])
        geometry = row['geometry']

        # 产品
        region_tiffile = tiffile.replace(coding, code)
        region_path = os.path.dirname(region_tiffile)


        if not os.path.exists(region_path):
            os.makedirs(region_path, exist_ok=True)

        clipped_xds = xds.rio.clip([geometry], gdf_combined.crs, drop=True)
        if (clipped_xds.isnull().all()) or (clipped_xds.size == 0) or np.all(clipped_xds == 0):
            print("裁剪后的数据集为空，跳过当前文件。")
            continue  # 跳过当前文件，处理下一个文件
        else:
            print("裁剪后的数据集不为空，程序将继续执行。")            
            # 将裁剪后的数据集保存为新的 TIFF 文件
            clipped_xds.rio.to_raster(region_tiffile, compress="lzw")        
        tiffiles.append(region_tiffile)
    return tiffiles

if __name__ == '__main__':
    auxPath=r"D:\2025_yulin\02_Algo\NYQH\qgis_src\auxpath"
    resultfile =r"D:\2025_yulin\05_Data\04_Output\potato\610802000\CRA_AL_610802000_L4_SITE_SURF_0030M_GLL_CLIM_19910101000000_20201231000000_M0.TIFF"
    if os.path.exists(resultfile):
        print(resultfile)
    rasterStyle =os.path.join(auxPath, "colorbar", "CRA_AL.qml")
    if os.path.exists(rasterStyle):
        print(rasterStyle)    
    template =os.path.join(auxPath, "template", "template.qgs")
    if os.path.exists(template):
        print(template)
    codesdict={"610802000":'榆阳区', "610802001":'鼓楼街道', "610802002":"青山路街道", 
                "610802003":'上郡路街道', "610802004":'新明楼街道', "610802005":'驼峰路街道',
                "610802006":'崇文路街道', "610802007":'航宇路街道',"610802008":'长城路街道',
                "610802100":'鱼河镇', "610802101":'上盐湾镇', "610802102":'镇川镇',
              "610802105":'麻黄梁镇', "610802106":'牛家梁镇', "610802107":'金鸡滩镇',
              "610802108":'马合镇', "610802109":'巴拉素镇', "610802111":'鱼河峁镇',
              "610802112":'青云镇', "610802113":'古塔镇', "610802114":'大河塔镇',
              "610802115":'小纪汗镇', "610802116":'芹河镇', "610802205":'孟家湾乡',
              "610802206":'小壕兔乡',"610802207":'岔河则乡',"610802208":'补浪河乡',"610802209":'红石桥乡'
        }
    codee="610802000"
    year="1990-2020"
    # drawmaps_(resultfile, rasterStyle, template, codee, mapinfo=mapinfo_name(year,codesdict[codee],"作物植被"))
    drawmaps(resultfile,rasterStyle,template,mapinfo=mapinfo_name(year,codesdict[codee],"作物植被"))