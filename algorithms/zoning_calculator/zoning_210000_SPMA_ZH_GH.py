import os
import json
import jenkspy
import numpy as np
import pandas as pd
from math import pi
from pathlib import Path
from osgeo import gdal
from algorithms.data_manager import DataManager


def classify_array(data, num_classes):
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.int32)
    try:
        breaks = np.array(jenkspy.jenks_breaks(valid.astype(float), int(num_classes)), dtype=float)
    except Exception:
        mn = float(np.nanmin(valid))
        mx = float(np.nanmax(valid))
        if mn == mx:
            breaks = np.linspace(mn - 0.01, mx + 0.01, int(num_classes) + 1)
        else:
            breaks = np.linspace(mn, mx, int(num_classes) + 1)
    res = np.zeros_like(data, dtype=np.int32)
    for i in range(1, len(breaks)):
        if i < len(breaks) - 1:
            mask = (data > breaks[i - 1]) & (data <= breaks[i]) & ~np.isnan(data)
        else:
            mask = (data > breaks[i - 1]) & ~np.isnan(data)
        res[mask] = i
    return res

def normalize_array(array):
    """将数组归一化到[0,1]，保留NaN；全常数时置为0.5"""
    if array.size == 0:
        return array

    # 非NaN掩码
    mask = ~np.isnan(array)
    if not np.any(mask):
        return np.zeros_like(array)

    valid_values = array[mask]
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    # 常数数组归一化到0.5
    if max_val == min_val:
        normalized_array = np.full_like(array, 0.5, dtype=float)
        normalized_array[~mask] = np.nan
    else:
        normalized_array = (array - min_val) / (max_val - min_val)
        normalized_array[~mask] = np.nan
    return normalized_array


def penman_et0(daily_data, lat_deg, elev_m, albedo=0.23, as_coeff=0.25, bs_coeff=0.5, k_rs=0.16):
    """计算日尺度ET0（Penman-Monteith），单位mm/day"""
    df = daily_data.copy()
    tmax = df['tmax']
    tmin = df['tmin']
    tmean = df['tavg'] if 'tavg' in df.columns else (tmax + tmin) / 2.0

    # 太阳辐射与天文参数
    phi = np.deg2rad(lat_deg)
    J = df.index.dayofyear
    dr = 1.0 + 0.033 * np.cos(2.0 * pi / 365.0 * J)
    delta = 0.409 * np.sin(2.0 * pi / 365.0 * J - 1.39)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))
    Ra = (24.0 * 60.0 / pi) * 0.0820 * dr * (omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s))
    N = 24.0 / pi * omega_s

    # 入射短波辐射估算
    if 'sunshine' in df.columns:
        n = df['sunshine']
        Rs = (as_coeff + bs_coeff * (n / N)) * Ra
    else:
        Rs = k_rs * np.sqrt(np.maximum(tmax - tmin, 0.0)) * Ra

    # 净辐射
    Rso = (0.75 + 2e-5 * elev_m) * Ra
    Rns = (1.0 - albedo) * Rs
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0
    ea = es * (df['rhum'] / 100.0) if 'rhum' in df.columns else es * 0.7
    sigma = 4.903e-9
    tmaxK = tmax + 273.16
    tminK = tmin + 273.16

    # 净长波辐射
    Rnl = sigma * (
        (tmaxK**4 + tminK**4) / 2.0) * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0))) * (1.35 * np.minimum(Rs / np.maximum(Rso, 1e-6), 1.0) - 0.35)
    Rn = Rns - Rnl

    # 心理常数与斜率
    P = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0)**5.26
    gamma = 0.000665 * P
    es_tmean = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098.0 * es_tmean / ((tmean + 237.3)**2)
    u2 = df['wind'] if 'wind' in df.columns else pd.Series(2.0, index=df.index)

    # 主公式
    et0 = (0.408 * delta * (Rn) + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
    return et0.clip(lower=0)


def calculate_cwdi(daily_data, weights, kc_map=None):
    """CWDI计算：ET0->ETc(kc*ET0)，P与ETc统一前移一天，50日窗口按5×10天块加权"""
    df = daily_data.copy()
    if 'P' not in df.columns and 'precip' in df.columns:
        df = df.rename(columns={'precip': 'P'})

    # 估算ET0并构造ETc
    if 'ET0' not in df.columns:
        lat_deg = float(df['lat'].iloc[0])
        elev_m = float(df['altitude'].iloc[0])
        df['ET0'] = penman_et0(df, lat_deg, elev_m)

    kc_series = pd.Series(0.0, index=df.index)
    m = df.index.month
    
    # kc系数映射（未提供则用默认表）
    sid = str(df['station_id'].iloc[0]) if 'station_id' in df.columns else (str(df['Station_Id_C'].iloc[0]) if 'Station_Id_C' in df.columns else None)
    region_liaodong = {'54259', '54346', '54349', '54353', '54365', '54453', '54483', '54493', '54494', '54497', '54660'}
    region_liaonan = {'54336', '54470', '54471', '54472', '54474', '54475', '54476', '54486', '54563', '54575', '54584', '54590', '54662'}
    region_liaoxi = {'54236', '54237', '54321', '54323', '54324', '54326', '54327', '54328', '54331', '54332', '54334', '54335', '54337', '54338', '54342', '54352', '54454', '54455'}
    region_liaobei = {'54243', '54249', '54252', '54254'}
    region_liaozhong = {'54244', '54245', '54333', '54339', '54347', '54351'}
    if sid in region_liaodong:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.47, 5: 0.68, 6: 0.92, 7: 1.13, 8: 1.12, 9: 0.84, 10: 0.0, 11: 0.0, 12: 0.0}
    elif sid in region_liaonan:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.46, 5: 0.70, 6: 0.92, 7: 1.21, 8: 1.11, 9: 0.83, 10: 0.0, 11: 0.0, 12: 0.0}
    elif sid in region_liaoxi:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.36, 5: 0.51, 6: 0.72, 7: 1.12, 8: 1.04, 9: 0.77, 10: 0.0, 11: 0.0, 12: 0.0}
    elif sid in region_liaobei:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.39, 5: 0.50, 6: 0.70, 7: 1.17, 8: 1.12, 9: 0.86, 10: 0.0, 11: 0.0, 12: 0.0}
    elif sid in region_liaozhong:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.40, 5: 0.52, 6: 0.76, 7: 1.21, 8: 1.13, 9: 0.89, 10: 0.0, 11: 0.0, 12: 0.0}
    else:
        upd = {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.22, 5: 1.13, 6: 0.83, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0}
    default_kc_map = upd
    use_kc_map = kc_map if kc_map is not None else default_kc_map
    norm_map = {}
    for k, v in use_kc_map.items():
        try:
            kk = int(k)
            vv = float(v)
            norm_map[kk] = vv
        except Exception:
            pass

    for mo, v in norm_map.items():
        kc_series[m == mo] = v

    df['ETc'] = kc_series * df['ET0']
    etc_shift = df['ETc'].shift(1)
    p_shift = df['P'].shift(1)
    w = np.array([weights[4], weights[3], weights[2], weights[1], weights[0]], dtype=float)
    
    # 50日滚动窗口转为5个10天块，按权重合成
    def _cwdi_window(etc_window):
        p_window = p_shift.loc[etc_window.index].values
        etc_vals = etc_window.values
        if len(etc_vals) < 50:
            return np.nan

        etc_blocks = etc_vals.reshape(5, 10)
        p_blocks = p_window.reshape(5, 10)
        etc_sum = etc_blocks.sum(axis=1)
        p_sum = p_blocks.sum(axis=1)

        # 仅在ETc>0且ETc>=P时计算干旱强度
        cond = (etc_sum > 0) & (etc_sum >= p_sum)
        cwdi_blocks = np.zeros(5, dtype=float)
        cwdi_blocks[cond] = (1 - p_sum[cond] / etc_sum[cond]) * 100.0
        return float(np.dot(w, cwdi_blocks))
    
    # 逐日滚动计算CWDI
    df['CWDI'] = etc_shift.rolling(window=50).apply(_cwdi_window, raw=False)
    df.loc[~df.index.month.isin([4, 5, 6, 7, 8, 9]), 'CWDI'] = np.nan
    return df


class SPMA_ZH:
    """辽宁-玉米-干旱区划计算器"""
    
    def _save_geotiff_gdal(self, data, meta, output_path, nodata=0):
        """保存GeoTIFF，按输入dtype选择GDAL类型"""
        if data.dtype == np.uint8:
            datatype = gdal.GDT_Byte
        elif data.dtype == np.uint16:
            datatype = gdal.GDT_UInt16
        elif data.dtype == np.int16:
            datatype = gdal.GDT_Int16
        elif data.dtype == np.uint32:
            datatype = gdal.GDT_UInt32
        elif data.dtype == np.int32:
            datatype = gdal.GDT_Int32
        elif data.dtype == np.float32:
            datatype = gdal.GDT_Float32
        elif data.dtype == np.float64:
            datatype = gdal.GDT_Float64
        else:
            datatype = gdal.GDT_Float32

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, meta['width'], meta['height'], 1, datatype, ['COMPRESS=LZW'])
        dataset.SetGeoTransform(meta['transform'])
        dataset.SetProjection(meta['crs'])
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(nodata)
        band.FlushCache()
        dataset = None
    
    def _get_algorithm(self, algorithm_name):
        """从注册器获取算法实例"""
        if algorithm_name not in self._algorithms:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        return self._algorithms[algorithm_name]
    
    def _save_params_snapshot(self, params):
        """保存参数快照到结果目录"""
        try:
            cfg = params.get('config', {})
            rp = cfg.get('resultPath') or os.getcwd()
            outdir = Path(rp)
            outdir.mkdir(parents=True, exist_ok=True)
            fp = outdir / "params_SPMA_GH.json"

            snap = {}
            snap['station_coords'] = params.get('station_coords', {})
            snap['algorithmConfig'] = params.get('algorithmConfig', {})
            snap['config'] = cfg
            snap['startDate'] = params.get('startDate') or cfg.get('startDate')
            snap['endDate'] = params.get('endDate') or cfg.get('endDate')

            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            return str(fp)

        except Exception:
            return ""
    
    def _interpolate(self, station_values, station_coords, config, algorithmConfig):
        """统一插值入口，兼容IDW/LSM-IDW"""
        interpolation = algorithmConfig.get("interpolation", {})
        interpolation_method = interpolation.get('method', 'lsm_idw')
        interpolation_params = interpolation.get('params', {})
        interpolator = self._get_algorithm(f"interpolation.{interpolation_method}")

        data = {
            'station_values': station_values,
            'station_coords': station_coords,
            'dem_path': config.get("demFilePath", ""),
            'shp_path': config.get("shpFilePath", ""),
            'grid_path': config.get("gridFilePath", ""),
            'area_code': config.get("areaCode", "")}
        result = interpolator.execute(data, interpolation_params)
        return result
    
    def drought_station_g(self, data, config):
        """
        计算单站点的春玉米干旱指标，返回两类度量：
        - Hcwdi：分生育阶段的干旱强度综合指数（分阶段统计→按权重汇总）
        - Qcwdi：全生育期水分亏缺程度（ETc 与 P 的比例关系）
        
        参数
        - data：某站点逐日气象数据 DataFrame，要求至少包含列：
          station_id/Station_Id_C、lat、altitude、tmax、tmin、rhum、sunshine/风速、precip/P
          索引为日期（DatetimeIndex）
        - config：CWDI 计算配置字典，支持：
          weights：5×10天块的权重（默认 [0.3,0.25,0.2,0.15,0.1]）
          kc_map：月份→作物系数 Kc 映射（可选，未提供时按区域默认）
        
        处理流程
        1) 调用 calculate_cwdi 计算逐日 CWDI（ET0→ETc，P/ETc 比较，50日窗口按 5×10 天块加权）
        2) 根据站点划分区域（辽东/辽南/辽西/辽北/辽中），取对应生育阶段时间窗
        3) 按阶段将 CWDI 求均值并与阈值分级，统计多年的轻/中/重/特旱次数
        4) 用阶段权重汇总得到 Hcwdi
        5) 在全生育期内累计 ETc 与 P，计算年均水分亏缺指数 Qcwdi
        返回
        - dict：{"Hcwdi": float, "Qcwdi": float}
        """
        # 逐日CWDI序列：内部完成ET0估算、ETc构造与P/ETc比较及50日窗口加权
        df = calculate_cwdi(data, config.get("weights", [0.3, 0.25, 0.2, 0.15, 0.1]), config.get("kc_map"))
        series = df["CWDI"] if "CWDI" in df.columns else pd.Series(dtype=float)
        if series.empty:
            return {"Hcwdi": np.nan, "Qcwdi": np.nan}

        # 站点所属区域判定（影响阶段窗口与默认Kc）
        sid = str(data['station_id'].iloc[0]) if 'station_id' in data.columns else (str(data['Station_Id_C'].iloc[0]) if 'Station_Id_C' in data.columns else None)
        region_liaodong = {'54259', '54346', '54349', '54353', '54365', '54453', '54483', '54493', '54494', '54497', '54660'}
        region_liaonan = {'54336', '54470', '54471', '54472', '54474', '54475', '54476', '54486', '54563', '54575', '54584', '54590', '54662'}
        region_liaoxi = {'54236', '54237', '54321', '54323', '54324', '54326', '54327', '54328', '54331', '54332', '54334', '54335', '54337', '54338', '54342', '54352', '54454', '54455'}
        region_liaobei = {'54243', '54249', '54252', '54254'}
        region_liaozhong = {'54244', '54245', '54333', '54339', '54347', '54351'}
        if sid in region_liaodong:
            region = "辽东"
        elif sid in region_liaonan:
            region = "辽南"
        elif sid in region_liaoxi:
            region = "辽西"
        elif sid in region_liaobei:
            region = "辽北"
        elif sid in region_liaozhong:
            region = "辽中"
        else:
            region = "辽东"

        # 分区生育阶段时间窗（按mm-dd），用于阶段均值计算
        stage_windows_defaults = {
            "辽东": {
                "出苗-拔节": ("05-12", "06-28"),
                "拔节-抽雄": ("06-28", "07-19"),
                "抽雄-乳熟": ("07-29", "08-18"),
                "乳熟-成熟": ("08-18", "09-13")},
            "辽西": {
                "出苗-拔节": ("05-13", "06-28"),
                "拔节-抽雄": ("06-28", "07-19"),
                "抽雄-乳熟": ("07-29", "08-29"),
                "乳熟-成熟": ("08-29", "09-13")},
            "辽南": {
                "出苗-拔节": ("05-07", "06-23"),
                "拔节-抽雄": ("06-23", "07-17"),
                "抽雄-乳熟": ("07-17", "08-14"),
                "乳熟-成熟": ("08-14", "09-07")},
            "辽北": {
                "出苗-拔节": ("05-09", "06-24"),
                "拔节-抽雄": ("06-24", "07-20"),
                "抽雄-乳熟": ("07-20", "08-26"),
                "乳熟-成熟": ("08-26", "09-21")},
            "辽中": {
                "出苗-拔节": ("05-09", "06-15"),
                "拔节-抽雄": ("06-15", "07-14"),
                "抽雄-乳熟": ("07-14", "08-12"),
                "乳熟-成熟": ("08-12", "09-21")}
        }
        stage_windows = stage_windows_defaults.get(region)

        # 阶段CWDI分级阈值（均值分级）
        stage_bins = {
            "出苗-拔节": [50, 65, 75, 85],
            "拔节-抽雄": [35, 50, 65, 70],
            "抽雄-乳熟": [35, 45, 55, 65],
            "乳熟-成熟": [50, 60, 70, 80]}

        stage_order = ["出苗-拔节", "拔节-抽雄", "抽雄-乳熟", "乳熟-成熟"]
        stage_weights = np.array([0.13, 0.21, 0.29, 0.37], dtype=float)  # 阶段权重 Wj
        sev_val = {"轻旱": 1.0, "中旱": 2.0, "重旱": 3.0, "特旱": 4.0}      # 等级记分 Di,j
        years = sorted(series.index.year.unique())

        # Hcwdi计算：按式（5-9），Pi,j 固定采用式（5-4）的“年均日数”口径
        h_years = []
        for y in years:
            counts = {st: {k: 0 for k in sev_val.keys()} for st in stage_order}  # 各阶段各级别次数，{'出苗-拔节': {'轻旱': 0, '中旱': 0, '重旱': 0, '特旱': 0}, ...}
            stage_days = {st: 0 for st in stage_order} # 各阶段累计天数，用于频率分母
            for st in stage_order: # "出苗-拔节"
                s_mmdd, e_mmdd = stage_windows[st]
                sdt = pd.to_datetime(f"{y}-{s_mmdd}")
                edt = pd.to_datetime(f"{y}-{e_mmdd}")
                seg = series[(series.index >= sdt) & (series.index <= edt)]
                if len(seg) == 0:
                    continue
                # 阈值
                b = stage_bins[st]  # [50,65,75,85]
                stage_days[st] += int(seg.size)
                counts[st]["轻旱"] += int(((seg > b[0]) & (seg <= b[1])).sum())
                counts[st]["中旱"] += int(((seg > b[1]) & (seg <= b[2])).sum())
                counts[st]["重旱"] += int(((seg > b[2]) & (seg <= b[3])).sum())
                counts[st]["特旱"] += int((seg > b[3]).sum())
            
            # 计算该年的Hcwdi
            h_sum = 0.0
            for i, st in enumerate(stage_order): # 针对每个生育期
                days = max(stage_days[st], 1)
                d_p = sum(sev_val[k] * (counts[st][k] / days) for k in sev_val.keys())
                h_sum += stage_weights[i] * d_p
            h_years.append(h_sum)

        Hcwdi = float(np.mean(h_years))

        # Qcwdi计算
        etc_total = 0.0
        p_total = 0.0
        n_years = 0
        for y in years:
            s_mmdd = stage_windows["出苗-拔节"][0]
            e_mmdd = stage_windows["乳熟-成熟"][1]
            sdt = pd.to_datetime(f"{y}-{s_mmdd}")
            edt = pd.to_datetime(f"{y}-{e_mmdd}")
            etc_sum = float(df.loc[(df.index >= sdt) & (df.index <= edt), "ETc"].sum()) if "ETc" in df.columns else 0.0
            p_sum = float(df.loc[(df.index >= sdt) & (df.index <= edt), "P"].sum()) if "P" in df.columns else 0.0
            if etc_sum > 0:
                etc_total += etc_sum
                p_total += p_sum
                n_years += 1

        if n_years > 0:
            etc_mean = etc_total / n_years
            p_mean = p_total / n_years
            if etc_mean > 0:
                Qcwdi = float(max(1.0 - p_mean / etc_mean, 0.0)) if etc_mean >= p_mean else 0.0
            else:
                Qcwdi = 0.0
        else:
            Qcwdi = 0.0

        return {"Hcwdi": Hcwdi, "Qcwdi": Qcwdi}
    
    def _calculate_GH(self, params):
        """主流程：读取→站点G→插值→分类→输出"""
        self._algorithms = params['algorithms']
        station_coords = params.get('station_coords', {})
        algorithm_config = params.get('algorithmConfig', {})
        cwdi_config = algorithm_config.get('cwdi', {})
        cfg = params.get('config', {})
        data_dir = cfg.get('inputFilePath')
        station_file = cfg.get('stationFilePath')

        dm = DataManager(data_dir, station_file, multiprocess=False, num_processes=1)
        station_ids = list(station_coords.keys())

        if not station_ids:
            station_ids = dm.get_all_stations()

        available = set(dm.get_all_stations())
        station_ids = [sid for sid in station_ids if sid in available]
        start_date = params.get('startDate') or cfg.get('startDate')
        end_date = params.get('endDate') or cfg.get('endDate')

        # 计算出来每个站点的Hcwdi和Qcwdi
        h_map = {}
        q_map = {}
        for sid in station_ids:
            daily = dm.load_station_data(sid, start_date, end_date)
            idxs = self.drought_station_g(daily, cwdi_config)
            h_map[sid] = float(idxs["Hcwdi"]) if np.isfinite(idxs["Hcwdi"]) else np.nan
            q_map[sid] = float(idxs["Qcwdi"]) if np.isfinite(idxs["Qcwdi"]) else np.nan

        # 对Hcwdi和Qcwdi先插值
        h_result = self._interpolate(h_map, station_coords, cfg, algorithm_config)
        q_result = self._interpolate(q_map, station_coords, cfg, algorithm_config)

        num_classes = 5
        h_cls = classify_array(h_result['data'], num_classes)
        q_cls = classify_array(q_result['data'], num_classes)
        combined = (h_cls * q_cls).astype(np.int32)

        # 中间结果输出
        g_tif_path = os.path.join(cfg.get("resultPath"), "intermediate", "干旱综合风险指数.tif")
        meta = h_result['meta']
        self._save_geotiff_gdal(combined, meta, g_tif_path, 0)

        return {
            'data': combined,
            'meta': {
                'width': meta['width'],
                'height': meta['height'],
                'transform': meta['transform'],
                'crs': meta['crs']
            },
            'type': '辽宁玉米干旱'}

    def calculate(self, params):
        """入口：保存参数快照并执行干旱计算"""
        self._save_params_snapshot(params)
        return self._calculate_GH(params)
