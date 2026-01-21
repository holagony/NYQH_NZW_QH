import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


def calc_mi(df):
    '''
    输入日数据，按月尺度标准流程计算并回填到日
    '''
    mon_pre = df['PRE_Time_2020'].resample('M').sum()
    mon_t = df['TEM_Avg'].resample('M').mean().fillna(0).clip(lower=0)
    H = mon_t.groupby(mon_t.index.year).apply(lambda s: np.sum((s / 5) ** 1.514))
    Hm = mon_t.index.to_series().map(lambda idx: H[idx.year])
    A = 6.75e-7 * (Hm**3) - 7.71e-5 * (Hm**2) + 1.7928e-2 * Hm + 0.49
    days = mon_t.index.days_in_month
    pet_mon = 16 * ((10 * mon_t) / Hm) ** A * (days / 30)
    mi_mon = (mon_pre - pet_mon) / pet_mon
    mi_map = mi_mon.set_axis(mi_mon.index.to_period('M')).to_dict()
    df = df.copy()
    df['mi'] = df.index.to_period('M').map(mi_map)
    return df


def calc_spi(df, period):
    '''
    日尺度 SPI(N)：前 N 天累计降水的标准化
    '''
    pre = df['PRE_Time_2020'].shift(1)
    cum = pre.rolling(window=period).sum()
    out = pd.Series(index=cum.index, dtype='float64')
    for m in range(1, 13):
        s = cum[cum.index.month == m].dropna()
        q = (s == 0).mean()
        pos = s[s > 0]
        if len(pos) < 10:
            out.loc[s.index] = np.nan
            continue
        a, loc, scale = gamma.fit(pos, floc=0)
        G = gamma.cdf(s, a, loc, scale)
        P = q + (1 - q) * G
        out.loc[s.index] = norm.ppf(P)
    out = out.round(5)
    return out.tolist()


def calc_spiw(df, N=60, alpha=0.85):
    '''
    日尺度 SPIW(N)：先计算加权累计降水 WAP，再按历月标准化
    '''
    pre = df['PRE_Time_2020'].shift(1)
    w = np.power(alpha, np.arange(N, 0, -1))
    wap = pre.rolling(window=N).apply(lambda x: np.dot(x, w), raw=True)
    out = pd.Series(index=wap.index, dtype='float64')
    for m in range(1, 13):
        s = wap[wap.index.month == m].dropna()
        q = (s == 0).mean()
        pos = s[s > 0]
        if len(pos) < 10:
            out.loc[s.index] = np.nan
            continue
        a, loc, scale = gamma.fit(pos, floc=0)
        G = gamma.cdf(s, a, loc, scale)
        P = q + (1 - q) * G
        out.loc[s.index] = norm.ppf(P)
    out = out.round(5)
    return out.tolist()


def calc_mci(df, a, b, c, d):
    '''
    使用日数据计算
    a北方西部0.3 南方0.5
    b北方西北0.5 南方0.6
    c北方西北0.3 南方0.2
    d北方西北0.2 南方0.1
    '''
    df_cp = df.copy()
    
    Ka = [0, 0, 0, 0.6, 1.0, 1.2, 1.2, 1.0, 0.9, 0.4, 0, 0]
    df_cp['Ka'] = np.take(Ka, df_cp.index.month - 1)

    df_cp = calc_mi(df_cp)
    spiw60 = calc_spiw(df_cp, N=60, alpha=0.85)
    spi90 = calc_spi(df_cp, 90)
    spi150 = calc_spi(df_cp, 150)

    df_cp['spiw60'] = spiw60
    df_cp['spi90'] = spi90
    df_cp['spi150'] = spi150
    df_cp['干旱指数'] = df_cp['Ka'] * (a * df_cp['spiw60'] + b * df_cp['mi'] + c * df_cp['spi90'] + d * df_cp['spi150'])
    df_cp['干旱指数'] = df_cp['干旱指数'].fillna(0).round(2)
    df_cp['干旱指数'] = df_cp['干旱指数'].replace(-0.0, 0.0)
    df_cp.dropna(subset=['TEM_Avg'], inplace=True)

    mci = df_cp['干旱指数'].to_frame()
    mci['轻度干旱'] = np.where((mci['干旱指数'] > -1) & (mci['干旱指数'] <= -0.5), 1, 0)
    mci['中度干旱'] = np.where((mci['干旱指数'] > -1.5) & (mci['干旱指数'] <= -1), 1, 0)
    mci['重度干旱'] = np.where((mci['干旱指数'] > -2) & (mci['干旱指数'] <= -1.5), 1, 0)
    mci['特度干旱'] = np.where(mci['干旱指数'] <= -2, 1, 0)

    return mci
