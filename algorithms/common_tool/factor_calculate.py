"""
暴雨孕灾环境影响系数计算方法
@version<1> 2021-05-10 Created by lyb
"""

try:
    import gdal
except:
    from osgeo import gdal
import numpy as np
import os
from common_tool.raster_tool import RasterTool
from common_tool.array_tool import ArrayTool

class EnvirImpactCoeff:

    def __init__(self, terrfile, waterfile, geologyfile, index_file, envi_file=None,nodata=None):
        """
        计算暴雨孕灾环境
        :param terrfile: str, 地形因子影响系数文件
        :param waterfile: str, 水系因子影响系数文件
        :param geologyfile: str, 地质灾害易发条件系数文件
        :param index_file: str,指数文件，用于对其他文件数据的掩膜
        :param envi_file: str, 孕灾环境结果文件
        """
        self.terrfile = terrfile
        self.waterfile = waterfile
        self.geologyfile = geologyfile
        self.index_file = index_file
        self.envi_file = envi_file
        self.nodata = nodata

    def statisTerrainFactor(self):
        """
        地形因子影响系数
        :return:
        """
        terr_ds = None
        if self.terrfile:
            if os.path.exists(self.terrfile):
                obj = RasterTool()
                # 栅格数据之间匹配
                outds = obj.alignRePrjRmp(self.index_file, self.terrfile, None,
                                          srs_nodata=None, dst_nodata=None,
                                          resample_type=gdal.GRA_NearestNeighbour)
                # 数据掩膜
                terr_ds = obj.maskRasterByRaster(outds, self.index_file, None, self.nodata, 100)
            else:
                pass
        else:
            pass
        return terr_ds

    def calculateWaterFacter(self):
        """
        计算水系因子的系数
        :return:
        """
        water_ds = None
        if self.waterfile:
            if os.path.exists(self.waterfile):
                obj = RasterTool()
                # 栅格数据之间匹配
                outds = obj.alignRePrjRmp(self.index_file, self.waterfile, None,
                                          srs_nodata=100, dst_nodata=None,
                                          resample_type=gdal.GRA_NearestNeighbour)
                # 数据掩膜
                water_ds = obj.maskRasterByRaster(outds, self.index_file, None, self.nodata, 100)
            else:
                pass
        else:
            pass
        return water_ds

    def statisGeoFactor(self):
        """
        统计地质条件系数
        :return:
        """
        gf_ds = None
        if self.geologyfile:
            if os.path.exists(self.geologyfile):
                obj = RasterTool()
                # 栅格数据之间匹配
                outds = obj.alignRePrjRmp(self.index_file, self.geologyfile, None,
                                          srs_nodata=100, dst_nodata=None,
                                          resample_type=gdal.GRA_NearestNeighbour)
                # 数据掩膜
                gf_ds = obj.maskRasterByRaster(outds, self.index_file, None, self.nodata, 100)
            else:
                pass
        else:
            pass
        return gf_ds

    def calculateImpactCoeff(self, terr_ds=None, water_ds=None, gf_ds=None, envi_file=None):
        """
        计算环境因子
        :return:
        """
        try:
            if (terr_ds is None) & (water_ds is None) & (gf_ds is None):
                outds = None
            else:
                data_list1 = []
                data_list2 = []
                for ds in [terr_ds, water_ds, gf_ds]:
                    if ds is None:
                        pass
                    else:
                        try:
                            ds = gdal.Open(ds)
                        except:
                            ds = ds
                        cols = ds.RasterXSize
                        rows = ds.RasterYSize
                        geo = ds.GetGeoTransform()
                        proj = ds.GetProjection()
                        data = ds.ReadAsArray()
                        max_v = np.max(data[data!=100])
                        min_v = np.min(data[data!=100])
                        data_norm = (data-min_v)/(max_v-min_v)
                        data_norm[data==100] = 100
                        data_list1.append(data_norm)
                        data2 = list(data[data != 100] / 10.0)
                        data_list2.append(data2)
                if len(data_list2)==1:
                    fac_data = data_list1[0]
                else:
                    data = np.array(data_list2).T
                    obj = ArrayTool()
                    data = obj.dataNormalProcess(data)
                    vars_w = obj.calcWeights(data)
                    fac_data = 0
                    for i in range(len(data_list1)):
                        fac_data = fac_data+data_list1[i]*vars_w[i]
                # fac_data[(fac_data<0)|(fac_data>1.1)] = self.nodata
                max_value = np.nanmax(fac_data[fac_data!=self.nodata])
                min_value = np.nanmin(fac_data[fac_data!=self.nodata])
                fac_data[fac_data!=self.nodata] = -1*0.3 + (fac_data[fac_data!=self.nodata]-min_value)/(max_value-min_value)*2*0.3
                if envi_file is None:
                    driver = gdal.GetDriverByName("MEM")
                    outds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)
                else:
                    driver = gdal.GetDriverByName("gtiff")
                    outds = driver.Create(envi_file, cols, rows, 1, gdal.GDT_Float32)
                outds.SetGeoTransform(geo)
                outds.SetProjection(proj)
                outband = outds.GetRasterBand(1)
                outband.WriteArray(fac_data)
                outband.SetNoDataValue(self.nodata)
            return outds
        finally:
            ds = None
            outds = None

    def run(self):
        """
        :return:
        """
        #获取地形因子影响系数
        terr_ds = self.statisTerrainFactor()
        #水系因子统计
        water_ds = self.calculateWaterFacter()
        #地质条件因子系数
        gf_ds = self.statisGeoFactor()
        #暴雨孕灾环境影响系数计算方法
        outds = self.calculateImpactCoeff(terr_ds, water_ds, gf_ds, self.envi_file)
        return outds

if __name__ == "__main__":
    terrfile=r"D:\HT_Project\depend\data_support\RainStorm\china_地形因子影响系数文件_WGS1984.tif"
    waterfile=r"D:\HT_Project\depend\data_support\shp_noo\HYDL_WGS1984_Albers.shp"
    geologyfile=None
    shpfile=r"D:\HT_Project\depend\shp\china_county_WGS1984.shp"
    temdir=r"D:\HT_Project\风险灾害项目\算法测试\暴雨危险性_20210606\中国区域"
    w_id=None
    envi_content=None
    c_value=None
    # demfile = r"D:\HT_Project\depend\data_support\RainStorm\china_dem_1984_clip_2.tif"
    # outfile = r"D:\HT_Project\depend\data_support\RainStorm\china_dem_1984_clip_2_factor.tif"
    obj = EnvirImpactCoeff(terrfile, waterfile, geologyfile, shpfile, temdir, w_id, envi_content, c_value)
    obj.run()
