# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:15:03 2025

@author: HTHT
"""
import os
from qgis.core import QgsProject, QgsRasterLayer,QgsVectorLayer,QgsCoordinateReferenceSystem,QgsMapLayer,QgsWkbTypes,QgsSymbol
from qgis.core import QgsExpressionContextUtils,QgsCoordinateTransform,QgsLayoutExporter,QgsApplication,QgsRuleBasedRenderer
from osgeo import gdal


 


def drawmaps(tiffile,pngfile, rasterStyle, template, code, mapinfo):
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
    if layout:
        print("Layout found:", layout.name())  # 打印布局名称来确认布局对象
    else:
        print("No layout found with the name 'layout'.")    
    mapitem = layout.itemById("Map 1")
    desCRS = mapitem.crs()

    rectangle = layout.itemById("colorbar")
    rectangle.setPicturePath(rasterStyle[:-3] + "png")

    if str(code).endswith('0000'):
        groupname = "provice"
    elif str(code).endswith('000'):
        groupname = "city"
    else:
        groupname = "xian"


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

    extent = map_layer.boundingBoxOfSelected()

    sourceCRS = map_layer.crs()

    transform = QgsCoordinateTransform(sourceCRS, desCRS, QgsProject.instance())
    map_extent = transform.transformBoundingBox(extent)

    map_extent.scale(1.1)
    mapitem.zoomToExtent(map_extent)

    # 添加标题,图例,卫星名称,分辨率等信息
    for k, v in mapinfo.items():
        if k == "title":
            v = mapinfo["title"]
        layout.itemById(k).setText(v)

    layout.refresh()
    # 输出图片
    #basename = os.path.basename(tiffile)
    #pngfile = tiffile.replace(basename.split("_")[2], str(code)).replace("tif", "png")
    exporter = QgsLayoutExporter(layout)
    exporter.exportToImage(pngfile, QgsLayoutExporter.ImageExportSettings())
    print(pngfile)

    return pngfile


def main(tiffile,pngfile,rasterStyle, template,code,areaname,crop_chinese,element_chinese,startdate,enddate):


    mapinfo = {"title": areaname+crop_chinese+element_chinese+"区划图",
               "date": startdate[:4]+"年"+"-"+enddate[:4]+"年",
               }        
        
    drawmaps(tiffile,pngfile, rasterStyle, template, code, mapinfo)  
    return pngfile



if __name__ == '__main__':
    tiffile="D:/project/农业气候资源普查和区划/code/china/NYQH_NZW_QH_v0/NYQH_SOYBEAN/BH_bean_moth/Q_PR_SOYB-BH_150000_102.tif"
    pngfile="D:/project/农业气候资源普查和区划/code/china/NYQH_NZW_QH_v0/NYQH_SOYBEAN/BH_bean_moth/Q_PP_SOYB-BH_150000_102.png"
    auxPath=r"D:\project\农业气候资源普查和区划\code\china\NYQH_NZW_QH_v0\qgis_src\auxpath"
    rasterStyle =os.path.join(auxPath, "colorbar", "BH.qml")   
    template =os.path.join(auxPath, "template", "template.qgs")
    code="150000"
    areaname="内蒙古自治区"
    cropname="大豆"
    zoningtype="BH"
    element="bean_moth"
    startdate="19910101"
    enddate="20201231"
    main(tiffile,pngfile,rasterStyle, template,code,areaname,cropname,zoningtype,element,startdate,enddate)
