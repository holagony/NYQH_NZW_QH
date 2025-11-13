import sys
from osgeo import gdal

p = r'C:\Users\mjynj\Desktop\NYQH\output\NYQH_WIWH\ZH_drought\Q_PR_WIWH-ZH_410000_000.tif'
ds = gdal.Open(p)
b = ds.GetRasterBand(1).ReadAsArray()
