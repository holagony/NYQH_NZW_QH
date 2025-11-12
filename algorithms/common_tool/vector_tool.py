"""
矢量处理工具
@Version<1> 2021-11-20 Created by lyb
"""

try:
    import gdal, osr, ogr
except:
    from osgeo import gdal, osr, ogr
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class VectorTool:

    @staticmethod
    def shpReProject(inshp, outfile, dst_epsg):
        """
        矢量转投影
        :param inshp: str, 输入的矢量文件
        :param outfile: str, 输出的矢量文件
        :param dst_epsg: int. 目标投影epsg, 如4490
        :return:
        """
        try:
            try:
                data = gpd.read_file(inshp, encoding="utf-8")
            except:
                data = gpd.read_file(inshp, encoding="GBK")
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            reprojected_data = data.to_crs(crs=out_srs.ExportToWkt())
            reprojected_data.to_file(outfile, driver="ESRI Shapefile", encoding="utf-8")
        finally:
            data = None
            reprojected_data = None

    @staticmethod
    def shpToRaster(shpfile=None,  outfile=None, boundary_range=None,
                    dst_epsg=None, dst_rows=None, dst_cols=None,
                    conver_filed="Rank", data_type=gdal.GDT_Float32,
                    nodata=0):
        """
        矢量转栅格，基于四至范围和行列转成栅格数据
        :param shpfile: str,矢量文件
        :param outfile: str, 结果文件
        :param boundary_range: list,四至边界，[70, 30, 120, 60]对应（左，下，右，下）
        :param dst_epsg: int., 目标epsg， 如4490
        :param dst_rows: int， 行，120
        :param dst_cols: int, 列，90
        :param conver_filed: str, 转换字段
        :param data_type: obj，数据类型
        :param nodata: float, 无效值
        :return:
        """
        try:
            x_min = boundary_range[0]
            y_max = boundary_range[3]
            x_cell = abs(boundary_range[2] - boundary_range[0]) / dst_cols
            y_cell = -1 * abs(boundary_range[3] - boundary_range[1]) / dst_rows
            if outfile is None:
                driver = gdal.GetDriverByName('MEM')
                out_ds = driver.Create("",  dst_cols, dst_rows, 1, data_type)
            else:
                driver = gdal.GetDriverByName('gtiff')
                out_ds = driver.Create(outfile, dst_cols, dst_rows, 1, data_type)
            out_ds.SetGeoTransform((x_min, x_cell, 0, y_max, 0, y_cell))
            ds = ogr.Open(shpfile)
            in_lyr = ds.GetLayer()
            out_srs = osr.SpatialReference()
            out_srs.ImportFromEPSG(dst_epsg)
            out_ds.SetProjection(out_srs.ExportToWkt())
            band = out_ds.GetRasterBand(1)
            band.SetNoDataValue(nodata)
            band.FlushCache()
            if conver_filed is None:
                gdal.RasterizeLayer(out_ds, [1], in_lyr, options=["ATTRIBUTE=" + "ID_Value"])
            else:
                gdal.RasterizeLayer(out_ds, [1], in_lyr, options=["ATTRIBUTE:" + conver_filed])
            return out_ds
        finally:
            ds = None

    @staticmethod
    def modifyShpAttr(inshpfile, fieldname=None):
        """
        修改矢量的属性表， 增加序号
        :param inshpfile: str,矢量shp文件路径
        :param fieldname: str,增加的属性字段
        :return:
        """
        try:
            try:
                shp_ds = gpd.read_file(inshpfile, encoding="GBK")
            except:
                shp_ds = gpd.read_file(inshpfile, encoding="UTF-8")
            if fieldname is None:
                shp_ds["ID_Value"] = np.arange(shp_ds.shape[0])+1
            else:
                shp_ds[fieldname] = np.arange(shp_ds.shape[0]) + 1
            shp_ds.to_file(inshpfile, driver="ESRI Shapefile", encoding='GBK')
        finally:
            shp_ds = None

    @staticmethod
    def deletShpAttr(inshpfile, fieldname=None):
        """
        删除矢量的属性
        :param inshpfile: str, 矢量文件路径
        :param fieldname: str, 删除属性字段
        :return:
        """
        try:
            try:
                shp_ds = gpd.read_file(inshpfile, encoding="utf-8")
            except:
                shp_ds = gpd.read_file(inshpfile, encoding="GBK")
            if fieldname is None:
                pass
            else:
                del shp_ds[fieldname]
            shp_ds.to_file(inshpfile, driver="ESRI Shapefile", encoding='GBK')
        finally:
            shp_ds = None

    @staticmethod
    def addShpAttr(inshpfile=None, fieldname=None, data=None):
        """
        修改矢量的属性表, 增加字段，
        :param inshpfile: str, 矢量文件
        :param fieldname: str, 字段名称
        :param data: list或array，增加的数据
        :return:
        """
        try:
            try:
                shp_ds = gpd.read_file(inshpfile, encoding="utf-8")
            except:
                shp_ds = gpd.read_file(inshpfile, encoding="GBK")
            if fieldname in shp_ds.columns:
                del shp_ds[fieldname]
            else:
                pass
            shp_ds[fieldname] = data
            shp_ds.to_file(inshpfile,driver="ESRI Shapefile", encoding='utf-8')
        finally:
            shp_ds = None

    @staticmethod
    def shpGetShp(inshp, outfile):
        """
        矢量重新以utf-8输出
        :param inshp: str, 输入的矢量文件
        :param outfile: str, 输出的矢量文件
        :return:
        """
        try:
            try:
                data = gpd.read_file(inshp, encoding="ins")
            except:
                data = gpd.read_file(inshp, encoding="GBK")
            data.to_file(outfile, driver="ESRI Shapefile", encoding="utf-8")
        finally:
            data = None

    @staticmethod
    def addAttrToShp(shpfile, regioncode, regionfile, rank_content,
                     rank_field=None, rank_descript_field=None):
        """
        往属性表中写属性
        :param shpfile: str, 矢量shp
        :param regioncode: str, 区域code
        :param regionfile: str，行政区划信息
        :param rank_content：dict,等级字典
        :param rank_field: str,  rank_content等级字段
        :param rank_descript_field: str, rank_content描述字段
        :return:
        """
        try:
            ds = pd.read_csv(regionfile, dtype=np.str)
            shp_ds = gpd.read_file(shpfile)
            # 省级或直辖市
            if str(regioncode)=="000000":
                shp_ds["_id"] = np.arange(shp_ds.shape[0]) + 1
                shp_ds["province_code"] = np.nan
                shp_ds["province_name"] = np.nan
                shp_ds["city_code"] = np.nan
                shp_ds["city_name"] = np.nan
                shp_ds["county_code"] = np.nan
                shp_ds["county_name"] = np.nan
            else:
                if int(regioncode[2:])==0:
                    region_ds = ds.query("province_code" + ' in ' + '[' + str([regioncode[0:2]])[1:-1] + ']')
                    shp_ds["_id"] = np.arange(shp_ds.shape[0]) + 1
                    shp_ds["province_code"] = region_ds["province_code"].values[0]
                    shp_ds["province_name"] = region_ds["province_name"].values[0]
                    shp_ds["city_code"] = np.nan
                    shp_ds["city_name"] = np.nan
                    shp_ds["county_code"] = np.nan
                    shp_ds["county_name"] = np.nan
                else:
                    # 市一级
                    if int(regioncode[4:])==0:
                        region_ds = ds.query("city_code" + ' in ' + '[' + str([regioncode[0:4]])[1:-1] + ']')
                        shp_ds["_id"] = np.arange(shp_ds.shape[0]) + 1
                        shp_ds["province_code"] = region_ds["province_code"].values[0]
                        shp_ds["province_name"] = region_ds["province_name"].values[0]
                        shp_ds["city_code"] = region_ds["city_code"].values[0]
                        shp_ds["city_name"] = region_ds["city_name"].values[0]
                        shp_ds["county_code"] = np.nan
                        shp_ds["county_name"] = np.nan
                    else:
                        # 县一级
                        region_ds = ds.query("county_code" + ' in ' + '[' + str([regioncode])[1:-1] + ']')
                        shp_ds["_id"] = np.arange(shp_ds.shape[0]) + 1
                        shp_ds["province_code"] = region_ds["province_code"].values[0]
                        shp_ds["province_name"] = region_ds["province_name"].values[0]
                        shp_ds["city_code"] = region_ds["city_code"].values[0]
                        shp_ds["city_name"] = region_ds["city_name"].values[0]
                        shp_ds["county_code"] = region_ds["county_code"].values[0]
                        shp_ds["county_name"] = region_ds["county_name"].values[0]
            rank_data = np.array(shp_ds["class"].astype(np.int).values)
            class_name = np.full(rank_data.size, "低", dtype='<U10')
            rank_numbers = len(rank_content[rank_field])
            rank_list = rank_content[rank_field]
            des_list = rank_content[rank_descript_field]
            for i in range(rank_numbers):
                class_name[rank_data == rank_list[i]] = des_list[i]
            shp_ds["Hazard_class_name"] = class_name
            del shp_ds["class"]
            shp_ds["class"] = rank_data
            shp_ds.to_file(shpfile, driver="ESRI Shapefile", encoding="utf-8")
        finally:
            ds = None
            shp_ds = None

if __name__ == "__main__":
    import glob
    import os
    infile = r"F:\东乡\河网密度处理\格网相关矢量\gs_sizi.shp"
    obj = VectorTool()
    shpfile = r"F:\东乡\河网密度处理\河流相关矢量\gs_sizi_cgcs2000.shp"
    obj.shpReProject(infile, shpfile, 4490)
