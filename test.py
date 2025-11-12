import sys
from osgeo import gdal

p = r'c:\Users\mjynj\Desktop\NYQH\output\NYQH_SOYBEAN\ZH_drought\Q_PR_SOYB-ZH_150000_000.tif'
ds = gdal.Open(p)
b = ds.GetRasterBand(1)
print('size', ds.RasterXSize, ds.RasterYSize)
print('bands', ds.RasterCount)
print('dtype', gdal.GetDataTypeName(b.DataType))
print('nodata', b.GetNoDataValue())
print('transform', ds.GetGeoTransform())
print('projection', ds.GetProjection())
gdal.Translate(p.replace('.tif', '.png'), ds, format='PNG', outputType=gdal.GDT_Byte)
ds = None