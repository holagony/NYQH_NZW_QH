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

class APPL_CL:
    """苹果气候区划计算器 - 陕西苹果（产量、品质、病虫害通用）"""

    def __init__(self):
        pass

    def __call__(self, params):
        return self.calculate(params)

    def calculate(self, params):
        """执行气候区划计算"""
        config = params['config']
        self._algorithms = params['algorithms']

        return self._calculate_element(params)

    def _calculate_element(self, params):
        """
        计算苹果气候区划
        """
        station_indicators = params['station_indicators']
        station_coords = params['station_coords']
        algorithmConfig = params['algorithmConfig']
        config = params['config']

        print(f'开始计算{config.get("cropCode","")}-{config.get("zoningType","")}-{config.get("element","")}-流程')

        # 统一的计算流程
        result = self._calculate_with_interpolation(station_indicators, station_coords, config, algorithmConfig)

        # 增加掩膜数据
        mask_path = config.get('maskFilePath')
        mask_arr = _mask_to_target_grid(mask_path, result['meta'])
        interp_data_masked = np.where(mask_arr == 1, np.maximum(result['data'], 0.0), np.nan)
        result['data'] = interp_data_masked
        print('数据掩膜完成')

        # 保存分级前的综合指标
        composite_index_result = {
            'data': result['data'].copy(),
            'meta': result['meta'].copy()
        }
        self._save_composite_index(composite_index_result, config, "composite_index")

        # 分级
        classification = algorithmConfig['classification']
        classification_method = classification.get('method', 'natural_breaks')
        classifier = self._get_algorithm(f"classification.{classification_method}")
        data = classifier.execute(result['data'], classification)
        result['data'] = data

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

    def _calculate_with_interpolation(self, station_indicators, station_coords,
                                             config, crop_config):
        """计算气候区划指标"""

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
            return self._interpolate_before_calculate(station_indicators, station_coords,
                                                      config, crop_config, interpolator,
                                                      interpolation_params)
        else:
            raise ValueError(f"不支持的插值顺序: {interpolation_order}")

    def _interpolate_before_calculate(self, station_indicators, station_coords,
                                      config, crop_config, interpolator, interpolation_params):
        """先插值各指标，再计算区划指标"""
        print("执行 before 插值顺序: 先插值各指标，再计算区划指标")

        indicator_configs = crop_config.get("indicators", {})
        interpolated_indicators = {}

        # 将站点数据转换为DataFrame
        station_df = self._convert_station_data_to_dataframe(station_indicators, indicator_configs)
        print(f"站点数据DataFrame形状: {station_df.shape}")

        for indicator_name in indicator_configs.keys():
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
        required_indicators = list(indicator_configs.keys())
        successful_indicators = [name for name in required_indicators if
                                 name in interpolated_indicators and interpolated_indicators[name] is not None]

        if len(successful_indicators) == 0:
            raise ValueError("所有指标插值都失败了")

        print(f"成功读取指标: {successful_indicators}")

        # 计算每个指标的分级赋值
        grading_config = crop_config.get("grading", {})
        graded_indicators = {}
        
        for indicator_name in successful_indicators:
            if indicator_name in grading_config:
                graded_data = self._apply_grading_rule(
                    interpolated_indicators[indicator_name]['data'],
                    grading_config[indicator_name]
                )
                graded_indicators[indicator_name] = {
                    'data': graded_data,
                    'meta': interpolated_indicators[indicator_name]['meta']
                }
                print(f"指标 {indicator_name} 分级完成")
        
        # 使用分级赋值后的指标计算综合评分
        composite_result = self._calculate_composite_indicator(graded_indicators, crop_config)
        return composite_result

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

    def _apply_grading_rule(self, data: np.ndarray, grading_rule: Dict[str, Any]) -> np.ndarray:
        """应用分级规则对数据进行赋值"""
        # 创建输出数组，初始化为0
        graded_data = np.zeros_like(data, dtype=float)
        graded_data[:] = np.nan
        
        # 获取阈值和赋值
        thresholds = grading_rule.get("thresholds", [])
        
        for threshold in thresholds:
            min_val = threshold.get("min", -np.inf)
            max_val = threshold.get("max", np.inf)
            value = threshold.get("value", 0)
            
            # 创建掩码
            mask = (data >= float(min_val)) & (data < float(max_val))
            graded_data[mask] = value
        
        # 处理特殊的上限情况
        if thresholds:
            last_threshold = thresholds[-1]
            if "max" not in last_threshold:
                # 最后一个阈值没有上限，处理所有大于min的值
                min_val = last_threshold.get("min", -np.inf)
                value = last_threshold.get("value", 0)
                mask = data >= float(min_val)
                graded_data[mask] = value
            
        # 处理特殊的下限情况
        if thresholds:
            first_threshold = thresholds[0]
            if "min" not in first_threshold:
                # 第一个阈值没有下限，处理所有小于max的值
                max_val = first_threshold.get("max", np.inf)
                value = first_threshold.get("value", 0)
                mask = data < float(max_val)
                graded_data[mask] = value
                
        return graded_data

    def _calculate_composite_indicator(self, indicators: Dict[str, Any],
                                      crop_config: Dict[str, Any]) -> Any:
        """计算综合评分指标"""
        print("计算综合评分指标")
        
        # 获取公式配置
        formula_config = crop_config.get("formula", {})
        formula_str = formula_config.get("formula", "")
        variables_config = formula_config.get("variables", {})
        
        if not formula_str:
            raise ValueError("未找到公式配置")
        
        # 准备变量数据
        variables_data = {}
        for var_name, var_config in variables_config.items():
            print(f"处理变量: {var_name}")
            ref_name = var_config.get("ref", "")
            if ref_name in indicators:
                variables_data[var_name] = indicators[ref_name]['data']
            else:
                print(f"警告: 变量 {var_name} 引用的指标 {ref_name} 不存在")
                continue
        
        # 计算公式
        try:
            local_env = {'np': np}
            local_env.update(variables_data)
            
            # 执行公式计算
            result_data = eval(formula_str, {"__builtins__": {}}, local_env)
            
            # 计算统计信息
            valid_mask = ~np.isnan(result_data)
            valid_count = np.sum(valid_mask)
            total_count = result_data.size
            
            if valid_count > 0:
                valid_data = result_data[valid_mask]
                min_val = np.nanmin(valid_data)
                max_val = np.nanmax(valid_data)
                mean_val = np.nanmean(valid_data)
                print(f"综合评分统计: 范围[{min_val:.4f}, {max_val:.4f}], 均值{mean_val:.4f}, 有效{valid_count}/{total_count}")
            
            # 包装结果
            first_indicator = next(iter(indicators.values()))
            result = {
                'data': result_data,
                'meta': first_indicator['meta']
            }
            
            return result
            
        except Exception as e:
            print(f"综合评分计算失败: {str(e)}")
            raise

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

    def _save_intermediate_result(self, result: Dict[str, Any], params: Dict[str, Any],
                                  indicator_name: str, nodata: float = -32768) -> None:
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
        """使用GDAL保存为GeoTIFF文件"""
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

            # 数据类型映射
            dtype_map = {
                np.uint8: gdal.GDT_Byte,
                np.uint16: gdal.GDT_UInt16,
                np.int16: gdal.GDT_Int16,
                np.uint32: gdal.GDT_UInt32,
                np.int32: gdal.GDT_Int32,
                np.float32: gdal.GDT_Float32,
                np.float64: gdal.GDT_Float64,
            }
            datatype = dtype_map.get(data.dtype.type, gdal.GDT_Float32)

            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(
                str(output_path),
                width,
                height,
                1,
                datatype,
                ['COMPRESS=LZW']
            )

            if dataset is None:
                raise ValueError(f"无法创建文件: {output_path}")

            transform = meta.get('transform')
            dataset.SetGeoTransform(transform)

            crs = meta.get('crs')
            if crs is not None:
                dataset.SetProjection(crs)

            band = dataset.GetRasterBand(1)
            band.WriteArray(data)

            nodata = meta.get('nodata')
            if nodata is not None:
                band.SetNoDataValue(float(nodata))

            dataset = None
            print(f"GeoTIFF文件保存成功: {output_path}")

        except Exception as e:
            print(f"使用GDAL保存GeoTIFF失败: {str(e)}")
            raise

    def _get_algorithm(self, algorithm_name: str) -> Any:
        """获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]