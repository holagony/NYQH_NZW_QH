import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from osgeo import gdal


def _mask_to_target_grid(mask_path, meta):
    src = gdal.Open(mask_path)
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', meta['width'], meta['height'], 1, gdal.GDT_Byte)
    dst.SetGeoTransform(meta['transform'])
    dst.SetProjection(meta['crs'])
    gdal.Warp(dst, src, resampleAlg=gdal.GRA_NearestNeighbour)
    arr = dst.GetRasterBand(1).ReadAsArray()
    return arr


class WIWH_ZZ:
    """冬小麦种植气候适宜性区划计算器 - 新疆冬小麦"""

    def __init__(self):
        pass

    def __call__(self, params):
        return self.calculate(params)

    def calculate(self, params):
        """执行种植气候适宜性区划计算"""
        config = params['config']
        self._algorithms = params['algorithms']

        return self._calculate_element(params)

    def _calculate_element(self, params):
        """
        计算冬小麦种植气候适宜性区划
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print(f'开始计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-流程')

        # 统一的计算流程：先插值并计算综合指标
        result = self._calculate_with_interpolation(station_indicators, station_coords, config, algorithmConfig)

        # 插值后先掩膜并处理异常值，然后再分级
        mask_path = config.get("maskFilePath")
        if mask_path:
            mask_arr = _mask_to_target_grid(mask_path, result['meta'])
            result['data'] = np.where(mask_arr == 1, np.maximum(result['data'], 0.0), np.nan)
        else:
            result['data'] = np.maximum(result['data'], 0.0)

        # 保存分级前的综合指标
        composite_index_result = {'data': result['data'].copy(), 'meta': result['meta'].copy()}
        self._save_composite_index(composite_index_result, config, "composite_index")

        # 分级（针对掩膜后的综合指标）
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(result['data'], classification)
        result['data'] = data

        # 读取地形因子并叠加地形适宜性
        # result = self._apply_terrain_suitability(result, config, algorithmConfig)

        print(f'计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-区划完成')
        return result

    def _save_composite_index(self, result: Dict[str, Any], config: Dict[str, Any], name: str):
        """保存分级前的综合指标"""
        try:
            print(f"保存分级前综合指标: {name}")

            # 创建输出目录
            output_dir = Path(config["resultPath"]) / "composite_index"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{name}.tif"

            # 保存为GeoTIFF
            self._save_geotiff_gdal(result['data'], result['meta'], output_path)
            print(f"综合指标保存成功: {output_path}")

            # 同时保存统计信息
            self._save_composite_stats(result['data'], output_dir, name)

        except Exception as e:
            print(f"保存综合指标失败: {str(e)}")

    def _save_composite_stats(self, data: np.ndarray, output_dir: Path, name: str):
        """保存综合指标统计信息"""
        try:
            valid_mask = ~np.isnan(data)
            valid_count = np.sum(valid_mask)
            total_count = data.size

            if valid_count > 0:
                valid_data = data[valid_mask]
                stats = {
                    'min': float(np.nanmin(valid_data)),
                    'max': float(np.nanmax(valid_data)),
                    'mean': float(np.nanmean(valid_data)),
                    'std': float(np.nanstd(valid_data)),
                    'valid_count': int(valid_count),
                    'total_count': int(total_count),
                    'valid_ratio': float(valid_count / total_count)
                }

                # 保存为JSON文件
                import json
                stats_path = output_dir / f"{name}_stats.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                print(f"综合指标统计: 范围[{stats['min']:.4f}, {stats['max']:.4f}], "
                      f"均值{stats['mean']:.4f}, 有效{valid_count}/{total_count}")

        except Exception as e:
            print(f"保存统计信息失败: {str(e)}")

    def _calculate_with_interpolation(self, station_indicators, station_coords, config, crop_config):
        """计算种植气候适宜性区划指标"""

        # 获取插值配置和顺序控制
        interpolation_config = crop_config.get("interpolation", {})
        interpolation_order = interpolation_config.get('order', 'before')
        interpolation_method = interpolation_config.get('method', 'lsm_idw')
        interpolation_params = interpolation_config.get('params', {})

        print(f"使用插值顺序: {interpolation_order}")

        # 获取插值算法
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")
        if interpolator is None:
            raise ValueError(f"不支持的插值方法: {interpolation_method}")

        if interpolation_order == 'before':
            return self._interpolate_before_calculate(station_indicators, station_coords, config, crop_config, interpolator, interpolation_params)
        else:
            raise ValueError(f"不支持的插值顺序: {interpolation_order}")

    def _interpolate_before_calculate(self, station_indicators, station_coords, config, crop_config, interpolator, interpolation_params):
        """先插值各指标，再计算适宜度并加权求和"""
        print("执行 before 插值顺序: 先插值各指标，再计算适宜度并加权求和")

        indicator_configs = crop_config.get("indicators", {})
        interpolated_indicators = {}

        # 将站点数据转换为DataFrame
        station_df = self._convert_station_data_to_dataframe(station_indicators, indicator_configs)
        print(f"站点数据DataFrame形状: {station_df.shape}")

        # 只处理X1-X5气候因子
        climate_indicators = ["acc_tmp", "tmin_extr", "sd_max", "tmax_d"]

        for indicator_name in climate_indicators:
            if indicator_name not in station_df.columns:
                print(f"警告: 指标 {indicator_name} 不在站点数据中，跳过")
                continue

            print(f"正在插值指标: {indicator_name}")

            # 从DataFrame中提取该指标的站点数值
            indicator_values = {}
            valid_count = 0

            for station_id in station_df.index:
                value = station_df.loc[station_id, indicator_name]
                if pd.notna(value):
                    indicator_values[station_id] = value
                    valid_count += 1
                else:
                    indicator_values[station_id] = np.nan

            if valid_count == 0:
                print(f"警告: 指标 {indicator_name} 没有有效数据，跳过插值")
                continue

            # 插值该指标
            try:
                interpolation_data = {
                    'station_values': indicator_values,
                    'station_coords': station_coords,
                    'dem_path': config.get("demFilePath", ""),
                    'shp_path': config.get("shpFilePath", ""),
                    'grid_path': config.get("gridFilePath", ""),
                    'area_code': config.get("areaCode", "")
                }

                # 检查中间结果是否存在
                file_name = indicator_name + ".tif"
                intermediate_dir = Path(config["resultPath"]) / "intermediate"
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                output_path = intermediate_dir / file_name

                if not os.path.exists(output_path):
                    interpolated_dict = interpolator.execute(interpolation_data, interpolation_params)
                    self._save_intermediate_result(interpolated_dict, config, indicator_name)
                    print(f"指标 {indicator_name} 插值完成")
                else:
                    print(f"读取指标 {indicator_name} ")
                    interpolated_dict = self._load_intermediate_result(output_path)

                interpolated_indicators[indicator_name] = interpolated_dict

            except Exception as e:
                print(f"指标 {indicator_name} 插值失败: {str(e)}")
                interpolated_indicators[indicator_name] = None

        # 检查是否所有必要指标都成功插值
        successful_indicators = [name for name in climate_indicators if name in interpolated_indicators and interpolated_indicators[name] is not None]

        if len(successful_indicators) == 0:
            raise ValueError("所有气候指标插值都失败了")

        print(f"成功插值气候指标: {successful_indicators}")

        #计算种植适宜度
        #climate_indicators = ["acc_tmp","tmin_extr","sd_max","tmax_d"]
        #suitability_indicators=interpolated_indicators["acc_tmp"]["data"].copy()
        suitability_indicators = np.zeros_like(interpolated_indicators["acc_tmp"]["data"])

        condition_1 = (interpolated_indicators["acc_tmp"]["data"] >= 2300) & (interpolated_indicators["tmin_extr"]["data"]
                                                                              >= (-30)) & (interpolated_indicators["sd_max"]["data"] >= 10)
        #不适宜

        #次适宜
        condition_2 = condition_1 & (interpolated_indicators["tmax_d"]["data"] >= 18)
        #适宜
        condition_3 = condition_1 & (interpolated_indicators["tmax_d"]["data"] > 6) & (interpolated_indicators["tmax_d"]["data"] < 18)
        #最适宜
        condition_4 = condition_1 & (interpolated_indicators["tmax_d"]["data"] <= 6)

        suitability_indicators = np.where(condition_2, 2.5, suitability_indicators)
        suitability_indicators = np.where(condition_3, 1.5, suitability_indicators)
        suitability_indicators = np.where(condition_4, 0.5, suitability_indicators)
        suitability_indicators = np.where(~condition_1, 3.5, suitability_indicators)

        result = {'data': suitability_indicators, 'meta': interpolated_indicators["acc_tmp"]["meta"]}
        return result

    def _convert_station_data_to_dataframe(self, station_indicators: Dict, indicator_configs: Dict) -> pd.DataFrame:
        """将站点数据转换为DataFrame"""
        data_dict = {}

        for station_id, station_data in station_indicators.items():
            row_data = {}

            for indicator_name in indicator_configs.keys():
                if isinstance(station_data, dict) and indicator_name in station_data:
                    value = station_data[indicator_name]
                else:
                    value = station_data

                # 确保值是数值类型
                if isinstance(value, (int, float)) and not np.isnan(value):
                    row_data[indicator_name] = value
                else:
                    row_data[indicator_name] = np.nan

            data_dict[station_id] = row_data

        # 创建DataFrame
        df = pd.DataFrame.from_dict(data_dict, orient='index')

        # 打印数据统计
        print("站点数据统计:")
        for col in df.columns:
            valid_count = df[col].notna().sum()
            total_count = len(df)
            if valid_count > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                print(f"  {col}: {valid_count}/{total_count} 有效, 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}")
            else:
                print(f"  {col}: 0/{total_count} 有效")

        return df

    # 以下工具方法保持不变
    def _load_intermediate_result(self, file_path: Path) -> Dict[str, Any]:
        """加载中间结果"""
        ds = gdal.Open(str(file_path))
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()
        data = ds.ReadAsArray()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        return {
            'data': data,
            'meta': {
                'transform': geo,
                'crs': proj,
                'height': data.shape[0],
                'width': data.shape[1],
                'dtype': data.dtype,
                'nodata': nodata
            }
        }

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any], indicator_name: str, nodata: float = -32768) -> None:
        """保存中间结果"""
        try:
            print(f"保存中间结果: {indicator_name}")

            file_name = indicator_name + ".tif"
            intermediate_dir = Path(params["resultPath"]) / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            output_path = intermediate_dir / file_name

            if isinstance(result, dict) and 'data' in result and 'meta' in result:
                data = result['data']
                meta = result['meta']
            elif hasattr(result, 'data') and hasattr(result, 'meta'):
                data = result.data
                meta = result.meta
            else:
                print(f"警告: 中间结果 {indicator_name} 格式不支持，跳过保存")
                return

            meta["nodata"] = nodata
            self._save_geotiff_gdal(data, meta, output_path)

        except Exception as e:
            print(f"保存中间结果 {indicator_name} 失败: {str(e)}")

    def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
        """使用GDAL保存为GeoTIFF文件，正确处理0值和缺测值"""
        try:
            from osgeo import gdal, osr

            if len(data.shape) == 1:
                if meta.get('width') and meta.get('height'):
                    data = data.reshape((meta['height'], meta['width']))
                else:
                    data = data.reshape((1, -1))
            elif len(data.shape) > 2:
                data = data.squeeze()

            height, width = data.shape

            # 数据类型映射 - 使用Float32以保留0值和nan
            datatype = gdal.GDT_Float32

            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(str(output_path), width, height, 1, datatype, ['COMPRESS=LZW'])

            if dataset is None:
                raise ValueError(f"无法创建文件: {output_path}")

            transform = meta.get('transform')
            dataset.SetGeoTransform(transform)

            crs = meta.get('crs')
            if crs is not None:
                dataset.SetProjection(crs)

            band = dataset.GetRasterBand(1)

            # 写入数据
            band.WriteArray(data)

            # 设置nodata值为nan
            band.SetNoDataValue(float('nan'))

            # 设置有效值范围（0-1）
            # band.SetMinimum(0.0)
            # band.SetMaximum(1.0)

            dataset = None
            print(f"GeoTIFF文件保存成功: {output_path}")

        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise

    # def _save_geotiff_gdal(self, data: np.ndarray, meta: Dict[str, Any], output_path: Path) -> None:
    #     """使用GDAL保存为GeoTIFF文件"""
    #     try:
    #         from osgeo import gdal, osr

    #         if len(data.shape) == 1:
    #             if meta.get('width') and meta.get('height'):
    #                 data = data.reshape((meta['height'], meta['width']))
    #             else:
    #                 data = data.reshape((1, -1))
    #         elif len(data.shape) > 2:
    #             data = data.squeeze()

    #         height, width = data.shape

    #         # 数据类型映射
    #         dtype_map = {
    #             np.uint8: gdal.GDT_Byte,
    #             np.uint16: gdal.GDT_UInt16,
    #             np.int16: gdal.GDT_Int16,
    #             np.uint32: gdal.GDT_UInt32,
    #             np.int32: gdal.GDT_Int32,
    #             np.float32: gdal.GDT_Float32,
    #             np.float64: gdal.GDT_Float64,
    #         }
    #         datatype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

    #         driver = gdal.GetDriverByName('GTiff')
    #         dataset = driver.Create(
    #             str(output_path),
    #             width,
    #             height,
    #             1,
    #             datatype,
    #             ['COMPRESS=LZW']
    #         )

    #         if dataset is None:
    #             raise ValueError(f"无法创建文件: {output_path}")

    #         transform = meta.get('transform')
    #         dataset.SetGeoTransform(transform)

    #         crs = meta.get('crs')
    #         if crs is not None:
    #             dataset.SetProjection(crs)

    #         band = dataset.GetRasterBand(1)
    #         band.WriteArray(data)

    #         nodata = meta.get('nodata')
    #         if nodata is not None:
    #             band.SetNoDataValue(float(nodata))

    #         dataset = None
    #         print(f"GeoTIFF文件保存成功: {output_path}")

    #     except Exception as e:
    #         print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
    #         raise

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]
