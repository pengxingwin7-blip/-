# -*- coding: utf-8 -*-
"""
华生 v4.1 §2-§6 自算脚本 (VRA 2026-10-08)

用法:  python3 -I watson_selfcalc.py <证据包目录>
输出:  <证据包目录>/14_watson_selfcalc.json  (全部数值, 供福尔摩斯读取)

数据口径 (经本脚本核验后确定, 与流水线默认口径不同之处已标注):
  1. depth5.parquet = MBO 全部事件 + 每行"事件后"合并五档 (post-event snapshot)
     依据: B侧A事件 price>bid_p1 的比例=0, 即新挂单已计入同行DOM.
  2. 本数据源每个 F(成交) 事件后紧跟一条 同时间戳/同size 的 C 记录 (仅为删簿),
     ⇒ 必须把 "F后同ts同size的C" 视为成交删簿, 不是撤单. 流水线未剔除这一点,
       导致 wall_candidates / CANCEL 家族把已成交订单误判为撤单.
  3. trades.parquet 的 publisher 81 = Nasdaq 场内成交 (与 MBO XNAS T 事件逐笔重合),
     82/83 = FINRA TRF (真正场外), 88/89 = Nasdaq BX/PSX 场内.
     ⇒ 只用 82/83 作 TRF; 81 中未与 MBO 重合的部分 = 开/收盘集合竞价 (各被重复报送一次).
  4. 订单键 = (publisher_id, channel_id, order_id), TICK = 0.01.
"""
import json
import sys

import numpy as np
import pandas as pd

BASE = sys.argv[1].rstrip('/') + '/'
TICK = 0.01
W = np.array([1.0, 0.6, 0.35, 0.2, 0.1])          # 委比 L1~L5 权重
RTH0, RTH1 = '2026-10-08 09:30:00', '2026-10-08 16:00:00'
OUT = {}


def et(s):
    """UTC -> 美东时间"""
    return s.dt.tz_convert('America/New_York')


def ts(x):
    return pd.Timestamp(x, tz='America/New_York')


# ---------------------------------------------------------------- 读数
d = pd.read_parquet(BASE + '07_VRA_depth5_2026-10-08.parquet')
d['et'] = et(d.ts_event)
d['size'] = d['size'].astype('int64')          # uint32 相减会溢出, 统一转 int64
d = d.reset_index(drop=True)
trf_all = pd.read_parquet(BASE + '06_VRA_ALL_trades_20261008.parquet')
trf_all['et'] = et(trf_all.ts_event)
trf_all['size'] = trf_all['size'].astype('int64')
day = pd.read_csv(BASE + '03_VRA_day.csv')
spy = pd.read_csv(BASE + '04_SPY_day.csv')
m30 = pd.read_csv(BASE + '02_VRA_30min.csv')

rth = (d.et >= ts(RTH0)) & (d.et < ts(RTH1))

# ---------------------------------------------------------------- 场内成交 (T事件)
T = d[(d.action == 'T') & rth].copy()
# MBO 中 Nasdaq 开盘竞价以 side=N 单笔出现 (09:30:00.63), 收盘竞价在 16:00:00.43 (已被 RTH 过滤)
# 竞价单独计量, 从连续竞价成交中剔除
T = T[~((T.publisher_id == 2) & (T.side == 'N') & (T.et < ts('2026-10-08 09:30:01')))]
# 主动方: T.side B=买方主动, A=卖方主动, N=无方向(隐藏单/交叉), N 不计入 Delta
# 同一主动单在同一 publisher 同一时间戳拆成多笔 -> 聚合为一个"父成交"
T['grp'] = T.groupby(['publisher_id', 'ts_event', 'side']).ngroup()
P = (T.groupby('grp')
       .agg(et=('et', 'first'), side=('side', 'first'), pub=('publisher_id', 'first'),
            size=('size', 'sum'), pmin=('price', 'min'), pmax=('price', 'max'),
            vwap=('price', lambda x: np.average(x, weights=T.loc[x.index, 'size'])))
       .sort_values('et').reset_index(drop=True))
Pd = P[P.side.isin(['B', 'A'])]

# ---------------------------------------------------------------- 竞价 / TRF
xnas_T = d[(d.action == 'T') & (d.publisher_id == 2)][['ts_event', 'price', 'size']].assign(hit=1)
p81 = trf_all[trf_all.publisher_id == 81].merge(xnas_T, on=['ts_event', 'price', 'size'], how='left')
cross = p81[p81.hit.isna()]
# 开盘竞价: 09:30:00.6298 汇总单 + 09:30:00.657 拆分明细 (两者相等, 重复报送) -> 只取汇总单
open_cross = cross[(cross.et < ts('2026-10-08 09:30:01'))]['size'].max()
close_cross = cross[(cross.et >= ts(RTH1))]['size'].max()
OUT['auction'] = {'open_cross_px': float(cross[cross.et < ts('2026-10-08 09:30:01')].price.iloc[0]),
                  'open_cross_sz': int(open_cross),
                  'close_cross_px': float(cross[cross.et >= ts(RTH1)].price.iloc[0]),
                  'close_cross_sz': int(close_cross),
                  'cross_double_report_check': {
                      'open_detail_sum': int(cross[(cross.et < ts('2026-10-08 09:30:01'))]['size'].sum() - open_cross),
                      'close_detail_sum': int(cross[(cross.et >= ts(RTH1))]['size'].sum() - close_cross)}}

trf = trf_all[trf_all.publisher_id.isin([82, 83]) & (trf_all.et >= ts(RTH0)) & (trf_all.et < ts(RTH1))].copy()
trf_ext = trf_all[trf_all.publisher_id.isin([82, 83]) & ~((trf_all.et >= ts(RTH0)) & (trf_all.et < ts(RTH1)))]
bxpsx = trf_all[trf_all.publisher_id.isin([88, 89])]

# ---------------------------------------------------------------- §2 基础快照
lit_vol = int(T['size'].sum())
trf_vol = int(trf['size'].sum())
day_csv_vol = int(day.iloc[-1].volume)
pre_lit = int(d[(d.action == 'T') & (d.et < ts(RTH0))]['size'].sum())   # 盘前场内 (盘后仅收盘竞价)
mbo_cov = lit_vol / day_csv_vol
allpx = pd.concat([T[['et', 'price', 'size']], trf[['et', 'price', 'size']]], ignore_index=True)
hi_row = allpx.loc[allpx.price.idxmax()]
lo_row = allpx.loc[allpx.price.idxmin()]
lit_hi = T.loc[T.price.idxmax()]
lit_lo = T.loc[T.price.idxmin()]
# 真 VWAP: 场内连续竞价 + TRF + 开收盘竞价
vw_num = (allpx.price * allpx['size']).sum() + OUT['auction']['open_cross_px'] * open_cross \
         + OUT['auction']['close_cross_px'] * close_cross
vw_den = allpx['size'].sum() + open_cross + close_cross
vwap = vw_num / vw_den
close = OUT['auction']['close_cross_px']
prev_close = float(day.iloc[-2].close)
avg20 = float(day.iloc[-21:-1].volume.mean())
spy_ret = spy.iloc[-1].close / spy.iloc[-2].close - 1
vra_ret = close / prev_close - 1

TB = int(Pd[Pd.side == 'B']['size'].sum())
TA = int(Pd[Pd.side == 'A']['size'].sum())
TN = int(P[P.side == 'N']['size'].sum())
FA = int(d[(d.action == 'F') & (d.side == 'A') & rth]['size'].sum())
FB = int(d[(d.action == 'F') & (d.side == 'B') & rth]['size'].sum())

# 开盘区间 ORH/ORL = 09:30-10:00
orr = allpx[allpx.et < ts('2026-10-08 10:00:00')]
ORH, ORL = float(orr.price.max()), float(orr.price.min())
# 成交量分布 (连续竞价: 场内+TRF, 不含竞价), 按 1 tick 取整
vp = allpx.assign(px=(allpx.price / TICK).round().astype(int) * TICK).groupby('px')['size'].sum().sort_index()
POC = float(vp.idxmax())
# 价值区 70%: 从 POC 向两侧扩展
vp_idx = list(vp.index)
lo_i = hi_i = vp_idx.index(vp.idxmax())
acc = vp.max()
while acc < 0.7 * vp.sum():
    up = vp.iloc[hi_i + 1] if hi_i + 1 < len(vp) else -1
    dn = vp.iloc[lo_i - 1] if lo_i - 1 >= 0 else -1
    if up >= dn:
        hi_i += 1; acc += up
    else:
        lo_i -= 1; acc += dn
VAH, VAL = float(vp.index[hi_i]), float(vp.index[lo_i])

# 尾盘 30min
last30 = Pd[Pd.et >= ts('2026-10-08 15:30:00')]
l30_vol = int(T[T.et >= ts('2026-10-08 15:30:00')]['size'].sum() + trf[trf.et >= ts('2026-10-08 15:30:00')]['size'].sum())

# 30min K线 (自建, 场内+TRF)
allpx['bar'] = allpx.et.dt.floor('30min')
bars = allpx.sort_values('et').groupby('bar').agg(o=('price', 'first'), h=('price', 'max'), l=('price', 'min'),
                                                  c=('price', 'last'), v=('size', 'sum'))
OUT['bars30_selfbuilt'] = {k.strftime('%H:%M'): {kk: (round(float(vv), 4) if kk != 'v' else int(vv)) for kk, vv in r.items()}
                           for k, r in bars.iterrows()}
OUT['bars30_csv_anomaly'] = {'14:30_csv_volume': int(m30.iloc[-3].volume),
                             '14:30_selfbuilt_volume': int(bars.loc[ts('2026-10-08 14:30:00')].v)}

OUT['snapshot'] = {
    'open': OUT['auction']['open_cross_px'], 'high_all': float(hi_row.price), 'high_time': hi_row.et.strftime('%H:%M:%S'),
    'high_lit': float(lit_hi.price), 'high_lit_time': lit_hi.et.strftime('%H:%M:%S'),
    'low_all': float(lo_row.price), 'low_time': lo_row.et.strftime('%H:%M:%S'),
    'low_lit': float(lit_lo.price), 'low_lit_time': lit_lo.et.strftime('%H:%M:%S'),
    'close': close, 'prev_close': prev_close, 'ret_pct': round(vra_ret * 100, 2),
    'lit_mbo_rth_vol': lit_vol, 'trf_rth_vol': trf_vol, 'trf_ext_hours_vol': int(trf_ext['size'].sum()),
    'bx_psx_vol': int(bxpsx['size'].sum()), 'mbo_premarket_vol': pre_lit,
    'open_cross': int(open_cross), 'close_cross': int(close_cross),
    'day_csv_vol': day_csv_vol, 'avg20_vol': round(avg20), 'rvol': round(day_csv_vol / avg20, 2),
    'unseen_venues_est': day_csv_vol - lit_vol - trf_vol - int(trf_ext['size'].sum()) - int(open_cross) - int(close_cross)
                         - int(bxpsx['size'].sum()) - pre_lit,
    'TB': TB, 'TA': TA, 'TN_no_side': TN, 'delta': TB - TA, 'FA': FA, 'FB': FB,
    'identity_TB_vs_FA_pct': round((TB - FA) / TB * 100, 1), 'identity_TA_vs_FB_pct': round((TA - FB) / TA * 100, 1),
    'vwap': round(vwap, 4), 'close_vs_vwap_pct': round((close / vwap - 1) * 100, 2),
    'spy_ret_pct': round(spy_ret * 100, 2), 'rs_pp': round((vra_ret - spy_ret) * 100, 2),
    'last30_vol_cont': l30_vol, 'last30_vol_pct_of_cont': round(l30_vol / (lit_vol + trf_vol) * 100, 1),
    'last30_delta': int(last30[last30.side == 'B']['size'].sum() - last30[last30.side == 'A']['size'].sum()),
    'ORH': ORH, 'ORL': ORL, 'POC': POC, 'VAH': VAH, 'VAL': VAL,
    'coverage_mbo_lit_pct': round(mbo_cov * 100, 1),
    'coverage_mbo_plus_auction_pct': round((lit_vol + open_cross + close_cross) / day_csv_vol * 100, 1),
}

# ---------------------------------------------------------------- §3.1 委比快照
def imb(row):
    """加权委比 = (加权Bid - 加权Ask)/(加权Bid + 加权Ask)*100"""
    b = np.array([row[f'bid_s{i}'] for i in range(1, 6)], dtype=float) @ W
    a = np.array([row[f'ask_s{i}'] for i in range(1, 6)], dtype=float) @ W
    return b, a, (b - a) / (b + a) * 100 if b + a > 0 else np.nan


book = d[rth & (d.bid_p1.notna()) & (d.ask_p1.notna())].sort_values('ts_event')


def snap_before(t, lag='1s'):
    """t 时刻前 lag 的最后一行 DOM"""
    s = book[book.et <= t - pd.Timedelta(lag)]
    return s.iloc[-1] if len(s) else None


rows = []
for bt in pd.date_range(ts(RTH0), ts('2026-10-08 15:30:00'), freq='30min'):
    s = book[book.et >= bt + pd.Timedelta('1s')]
    if len(s):
        r = s.iloc[0]; b, a, x = imb(r)
        rows.append({'t': bt.strftime('%H:%M'), 'ev': '30min bar', 'wb': round(b), 'wa': round(a), 'imb': round(x, 1)})
# P99 父成交前 1s
p99 = float(np.percentile(Pd['size'], 99))
for _, p in Pd[Pd['size'] >= p99].iterrows():
    r = snap_before(p.et)
    if r is None:
        continue
    b, a, x = imb(r)
    agree = '顺' if (p.side == 'B' and x > 0) or (p.side == 'A' and x < 0) else '逆'
    rows.append({'t': p.et.strftime('%H:%M:%S'), 'ev': f'P99 {"买" if p.side == "B" else "卖"}{int(p["size"])}@{p.vwap:.2f}',
                 'wb': round(b), 'wa': round(a), 'imb': round(x, 1), 'agree': agree})
# 触及关键位前 1s (10:00 后首次)
after = allpx[allpx.et >= ts('2026-10-08 10:00:00')].sort_values('et')
for nm, lv in [('ORH', ORH), ('ORL', ORL), ('VAH', VAH), ('VAL', VAL), ('POC', POC)]:
    hit = after[(after.price >= lv - 1e-9) & (after.price <= lv + 1e-9)] if nm == 'POC' else \
        (after[after.price >= lv - 1e-9] if nm in ('ORH', 'VAH') else after[after.price <= lv + 1e-9])
    if len(hit):
        t0 = hit.iloc[0].et
        r = snap_before(t0); b, a, x = imb(r)
        rows.append({'t': t0.strftime('%H:%M:%S'), 'ev': f'触及{nm} {lv:.2f}前1s', 'wb': round(b), 'wa': round(a), 'imb': round(x, 1)})
OUT['imbalance_snapshots'] = rows
OUT['imbalance_bar_mean'] = round(float(np.nanmean([r['imb'] for r in rows if r['ev'] == '30min bar'])), 1)

# ---------------------------------------------------------------- §3.2 ISO 扫单
# 定义: 5ms 内同向主动成交, 跨 ≥2 价位 (严格) 或 ≥2 交易所 (宽口径), 总量 ≥ 单笔 T 的 P99
t99 = float(np.percentile(T[T.side.isin(['A', 'B'])]['size'], 99))
iso = []
for side in ['B', 'A']:
    s = T[T.side == side].sort_values('ts_event')
    start = None; bucket = []
    for _, r in s.iterrows():
        if start is None or (r.ts_event - start) > pd.Timedelta('5ms'):
            if bucket:
                iso.append(bucket)
            start = r.ts_event; bucket = [r]
        else:
            bucket.append(r)
    if bucket:
        iso.append(bucket)
iso_rows = []
for b in iso:
    df = pd.DataFrame(b)
    tot = int(df['size'].sum()); npx = df.price.nunique(); nven = df.publisher_id.nunique()
    if tot >= t99 and (npx >= 2 or nven >= 2):
        iso_rows.append({'t': df.et.iloc[0].strftime('%H:%M:%S.%f')[:-3], 'side': df.side.iloc[0], 'levels': int(npx),
                         'venues': int(nven), 'size': tot, 'px': f'{df.price.min():.2f}-{df.price.max():.2f}',
                         'strict': bool(npx >= 2)})
iso_df = pd.DataFrame(iso_rows)
OUT['iso'] = {'T_p99': t99, 'rows': iso_rows,
              'strict_buy': [int((iso_df.strict & (iso_df.side == 'B')).sum()), int(iso_df[iso_df.strict & (iso_df.side == 'B')]['size'].sum())] if len(iso_df) else [0, 0],
              'strict_sell': [int((iso_df.strict & (iso_df.side == 'A')).sum()), int(iso_df[iso_df.strict & (iso_df.side == 'A')]['size'].sum())] if len(iso_df) else [0, 0],
              'broad_buy': [int((iso_df.side == 'B').sum()), int(iso_df[iso_df.side == 'B']['size'].sum())] if len(iso_df) else [0, 0],
              'broad_sell': [int((iso_df.side == 'A').sum()), int(iso_df[iso_df.side == 'A']['size'].sum())] if len(iso_df) else [0, 0]}

# ---------------------------------------------------------------- 订单生命周期 (修正: 剔除 F 后的删簿 C)
ev = d[d.action.isin(['A', 'M', 'F', 'C'])].copy()
ev['key'] = list(zip(ev.publisher_id, ev.channel_id, ev.order_id))
# 成交删簿 C: 同订单, 同 ts_event, 与上一条 F 同 size
ev['prev_act'] = ev.groupby('key').action.shift(1)
ev['prev_ts'] = ev.groupby('key').ts_event.shift(1)
ev['prev_sz'] = ev.groupby('key')['size'].shift(1)
ev['fill_removal'] = (ev.action == 'C') & (ev.prev_act == 'F') & (ev.prev_ts == ev.ts_event) & (ev.prev_sz == ev['size'])
OUT['c_fill_removal'] = {'C_total': int((ev.action == 'C').sum()), 'C_fill_removal': int(ev.fill_removal.sum()),
                         'C_size_total': int(ev[ev.action == 'C']['size'].sum()),
                         'C_size_fill_removal': int(ev[ev.fill_removal]['size'].sum())}

adds = ev[ev.action == 'A'].drop_duplicates('key').set_index('key')
# 挂单档位 (事件后快照, 同行 DOM 直接匹配)
def level_of(r):
    pre = 'bid' if r.side == 'B' else 'ask'
    for i in range(1, 6):
        if abs(r[f'{pre}_p{i}'] - r.price) < 1e-9:
            return i
    return 99


adds['lvl'] = adds.apply(level_of, axis=1)
# 距离 (tick): 相对同侧最优价 (事件后快照, 若挂单本身成为最优则=0)
adds['dist'] = np.where(adds.side == 'B', (adds.bid_p1 - adds.price) / TICK, (adds.price - adds.ask_p1) / TICK).round()
fills = ev[ev.action == 'F'].groupby('key').agg(fill=('size', 'sum'), f_first=('ts_event', 'min'))
canc = ev[(ev.action == 'C') & ~ev.fill_removal].groupby('key').agg(c_sz=('size', 'sum'), c_last=('ts_event', 'max'))
endt = ev.groupby('key').ts_event.max()
O = adds[['et', 'ts_event', 'side', 'price', 'size', 'lvl', 'dist', 'publisher_id']].join(fills).join(canc)
O['fill'] = O['fill'].fillna(0); O['c_sz'] = O['c_sz'].fillna(0)
O['fill_ratio'] = O['fill'] / O['size']
O['end'] = endt.reindex(O.index)
O['life'] = (O['end'] - O['ts_event']).dt.total_seconds()
O = O[(O.et >= ts(RTH0)) & (O.et < ts(RTH1))]

# 极速撤挂比 (≤1s 纯撤)
fast = {}
for s in ['B', 'A']:
    o = O[O.side == s]
    fast[s] = round(float(((o.c_sz > 0) & (o.fill == 0) & (o.life <= 1)).mean() * 100), 1)
OUT['fast_cancel_pct'] = fast

# ---------------------------------------------------------------- §3.3 挂单墙 (修正版: 真正撤单 vs 成交)
a99 = float(np.percentile(O['size'], 99))
wall = O[(O['size'] >= a99) & (O.life > 15)]
wall_tab = wall.assign(outcome=np.where(wall.fill_ratio >= 0.999, 'FILLED', np.where(wall.fill > 0, 'PARTIAL', 'CANCELLED')))
OUT['walls_corrected'] = {
    'A_p99': a99,
    'by_side_outcome': wall_tab.groupby(['side', 'outcome'])['size'].agg(['count', 'sum']).reset_index().to_dict('records'),
    'top': wall_tab.sort_values('size', ascending=False).head(10)[['et', 'side', 'price', 'size', 'life', 'lvl', 'fill', 'outcome']]
        .assign(et=lambda x: x.et.dt.strftime('%H:%M:%S')).to_dict('records')}

# ---------------------------------------------------------------- §3.4 Delta / CVD
q = {k: float(np.percentile(Pd['size'], k)) for k in [90, 95, 98, 99]}
big = Pd[Pd['size'] >= q[95]]; sm = Pd[Pd['size'] < q[95]]
cvd = Pd.assign(sd=np.where(Pd.side == 'B', Pd['size'], -Pd['size'])).set_index('et').sd.cumsum()
l2h = Pd[Pd.et >= ts('2026-10-08 14:00:00')]
lo_day = min(float(lo_row.price), float(lit_lo.price))
OUT['delta'] = {
    'pct': q,
    'big': [int(big[big.side == 'B']['size'].sum()), int(big[big.side == 'A']['size'].sum())],
    'small': [int(sm[sm.side == 'B']['size'].sum()), int(sm[sm.side == 'A']['size'].sum())],
    'cvd_peak': [int(cvd.max()), cvd.idxmax().strftime('%H:%M:%S')],
    'cvd_trough': [int(cvd.min()), cvd.idxmin().strftime('%H:%M:%S')],
    'cvd_end': int(cvd.iloc[-1]),
    'cvd_by_30min': {k.strftime('%H:%M'): int(v) for k, v in cvd.resample('30min').last().items()},
    'last2h_delta': int(l2h[l2h.side == 'B']['size'].sum() - l2h[l2h.side == 'A']['size'].sum()),
    'close_vs_low_pct': round((close - lo_day) / close * 100, 2)}
# 分时段 Delta
seg = Pd.assign(sd=np.where(Pd.side == 'B', Pd['size'], -Pd['size']), bar=Pd.et.dt.floor('30min')).groupby('bar').sd.sum()
OUT['delta']['delta_by_30min'] = {k.strftime('%H:%M'): int(v) for k, v in seg.items()}

# ---------------------------------------------------------------- §3.5 谎骗确认 (150s 穿越)
# 候选: 1~5档挂入, 有真正撤单, fill_ratio<15%
lit_tr = T[['ts_event', 'price']].sort_values('ts_event')
lt_ts = lit_tr.ts_event.values; lt_px = lit_tr.price.values
close_ts = ts(RTH1)


def cross150(r):
    if r.c_last is pd.NaT or pd.isna(r.c_last):
        return 'NA'
    if r.c_last + pd.Timedelta('150s') > close_ts.tz_convert('UTC'):
        return 'NA'
    i0 = np.searchsorted(lt_ts, r.c_last.to_datetime64()); i1 = np.searchsorted(lt_ts, (r.c_last + pd.Timedelta('150s')).to_datetime64())
    px = lt_px[i0:i1]
    if r.side == 'B':
        return 'SPOOF' if (px < r.price - 1e-9).any() else 'PROP'
    return 'SPOOF' if (px > r.price + 1e-9).any() else 'PROP'


cand = O[(O.lvl <= 5) & (O.c_sz > 0) & (O.fill_ratio < 0.15)].copy()
cand['res'] = cand.apply(cross150, axis=1)
sp = {}
for s in ['B', 'A']:
    c = cand[cand.side == s]
    for lab, sub in [('all', c), ('ge_p95', c[c['size'] >= np.percentile(O['size'], 95)])]:
        n = len(sub); ok = (sub.res == 'SPOOF').sum(); pr = (sub.res == 'PROP').sum(); na = (sub.res == 'NA').sum()
        sp[f'{s}_{lab}'] = {'n': int(n), 'spoof': int(ok), 'spoof_pct': round(ok / n * 100, 1) if n else None,
                            'prop': int(pr), 'na': int(na)}
OUT['spoof150'] = sp
# 深预埋纯撤单: size≥P98, dist>10tick, life>120s, 无成交
a98 = float(np.percentile(O['size'], 98))
deep = O[(O['size'] >= a98) & (O.dist > 10) & (O.life > 120) & (O.fill == 0) & (O.c_sz > 0)]
OUT['deep_cancel'] = {'A_p98': a98, 'B': [int((deep.side == 'B').sum()), int(deep[deep.side == 'B']['size'].sum())],
                      'A': [int((deep.side == 'A').sum()), int(deep[deep.side == 'A']['size'].sum())]}

# ---------------------------------------------------------------- §3.6 TRF 场外 (仅 FINRA 82/83)
bk = book[['ts_event', 'bid_p1', 'ask_p1']].sort_values('ts_event')
trf = pd.merge_asof(trf.sort_values('ts_event'), bk, on='ts_event', direction='backward')
trf['cls'] = np.where(trf.price >= trf.ask_p1 - 1e-9, 'near_ask', np.where(trf.price <= trf.bid_p1 + 1e-9, 'near_bid', 'mid'))
tc = trf.groupby('cls')['size'].sum()
hi_day = max(float(hi_row.price), float(lit_hi.price))
rng = hi_day - lo_day
t995 = float(np.percentile(trf['size'], 99.5))
top_trf = trf[trf['size'] >= t995].sort_values('size', ascending=False).head(5)
trf['px'] = (trf.price / TICK).round() * TICK
top3 = trf.groupby('px')['size'].sum().sort_values(ascending=False).head(3)
OUT['trf'] = {'total': int(trf['size'].sum()),
              'near_ask': int(tc.get('near_ask', 0)), 'near_bid': int(tc.get('near_bid', 0)), 'mid': int(tc.get('mid', 0)),
              'net': int(tc.get('near_ask', 0) - tc.get('near_bid', 0)),
              'p995': t995,
              'top5': [{'t': r.et.strftime('%H:%M:%S'), 'px': round(r.price, 4), 'sz': int(r['size']), 'cls': r.cls,
                        'bid': r.bid_p1, 'ask': r.ask_p1} for _, r in top_trf.iterrows()],
              'top3_px': [{'px': round(float(k), 2), 'sz': int(v), 'pct_of_range': round((k - lo_day) / rng * 100, 1)} for k, v in top3.items()],
              'vwap_trf': round(float(np.average(trf.price, weights=trf['size'])), 4),
              'vol_wtd_range_pct': round(float((np.average(trf.price, weights=trf['size']) - lo_day) / rng * 100), 1)}
# 分时段 TRF
OUT['trf']['by_30min'] = {k.strftime('%H:%M'): {'v': int(g['size'].sum()),
                                                 'net': int(g[g.cls == 'near_ask']['size'].sum() - g[g.cls == 'near_bid']['size'].sum())}
                          for k, g in trf.groupby(trf.et.dt.floor('30min'))}

# ---------------------------------------------------------------- §3.7 预谋强度
pre = O[(O['size'] >= a98) & (O.fill > 0)].copy()
pre['T_travel'] = (pre.f_first - pre.ts_event).dt.total_seconds()
pre['cls'] = np.where((pre.dist > 10) & (pre.T_travel > 120), 'strong', np.where(pre.dist <= 2, 'realtime', 'fast'))
pm = {}
for s in ['B', 'A']:
    for c in ['strong', 'fast', 'realtime']:
        x = pre[(pre.side == s) & (pre.cls == c)]
        pm[f'{s}_{c}'] = [int(len(x)), int(x['size'].sum()), int(x['fill'].sum())]
sB, sA = pm['B_strong'][2], pm['A_strong'][2]
strong = pre[pre.cls == 'strong']
pub_conc = float(strong.groupby('publisher_id')['fill'].sum().max() / strong['fill'].sum()) if len(strong) else None
OUT['premed'] = {'A_p98': a98, 'table': pm, 'ratio_BA_fill': (round(sB / sA, 2) if sA else ('inf' if sB else 'n/a')),
                 'pub_concentration': pub_conc,
                 'strong_list': strong[['et', 'side', 'price', 'size', 'fill', 'dist', 'T_travel', 'publisher_id']]
                     .assign(et=lambda x: x.et.dt.strftime('%H:%M:%S')).to_dict('records')}

# ---------------------------------------------------------------- 关键事件: 10:35 吃墙 + 10:53-10:57 急跌
win = lambda a, b: Pd[(Pd.et >= ts(a)) & (Pd.et < ts(b))]
def flow(a, b):
    w = win(a, b); t2 = trf[(trf.et >= ts(a)) & (trf.et < ts(b))]
    px = allpx[(allpx.et >= ts(a)) & (allpx.et < ts(b))].sort_values('et')
    return {'TB': int(w[w.side == 'B']['size'].sum()), 'TA': int(w[w.side == 'A']['size'].sum()),
            'trf': int(t2['size'].sum()), 'trf_net': int(t2[t2.cls == 'near_ask']['size'].sum() - t2[t2.cls == 'near_bid']['size'].sum()),
            'px_first': float(px.price.iloc[0]) if len(px) else None, 'px_last': float(px.price.iloc[-1]) if len(px) else None,
            'hi': float(px.price.max()) if len(px) else None, 'lo': float(px.price.min()) if len(px) else None}
OUT['windows'] = {
    '09:30-10:00': flow('2026-10-08 09:30:00', '2026-10-08 10:00:00'),
    '10:00-10:34:59': flow('2026-10-08 10:00:00', '2026-10-08 10:34:59'),
    '10:34:59-10:35:02 (吃墙)': flow('2026-10-08 10:34:59', '2026-10-08 10:35:02'),
    '10:35:02-10:50': flow('2026-10-08 10:35:02', '2026-10-08 10:50:00'),
    '10:50-10:58 (急跌)': flow('2026-10-08 10:50:00', '2026-10-08 10:58:00'),
    '10:58-11:45': flow('2026-10-08 10:58:00', '2026-10-08 11:45:00'),
    '11:45-14:00': flow('2026-10-08 11:45:00', '2026-10-08 14:00:00'),
    '14:00-15:45': flow('2026-10-08 14:00:00', '2026-10-08 15:45:00'),
    '15:45-16:00': flow('2026-10-08 15:45:00', '2026-10-08 16:00:00')}

# ---------------------------------------------------------------- Liquidity Hole (自算)
# 60s 窗口: 同向主动量占比≥75%, 价格位移≥5tick, 且窗口内 spread 峰值 ≥ 全日 spread 中位数×3
book2 = book.assign(spr=(book.ask_p1 - book.bid_p1) / TICK)
med_spr = float(book2.spr.median())
lh = []
for t0 in pd.date_range(ts(RTH0), ts('2026-10-08 15:59:00'), freq='60s'):
    w = Pd[(Pd.et >= t0) & (Pd.et < t0 + pd.Timedelta('60s'))]
    px = allpx[(allpx.et >= t0) & (allpx.et < t0 + pd.Timedelta('60s'))].sort_values('et')
    if len(w) < 3 or len(px) < 2:
        continue
    b = w[w.side == 'B']['size'].sum(); a = w[w.side == 'A']['size'].sum()
    mv = (px.price.iloc[-1] - px.price.iloc[0]) / TICK
    sp_ = book2[(book2.et >= t0) & (book2.et < t0 + pd.Timedelta('60s'))].spr
    if len(sp_) == 0:
        continue
    dom = b / (a + b)
    if ((dom >= 0.75 and mv >= 5) or (dom <= 0.25 and mv <= -5)) and sp_.max() >= 3 * med_spr:
        lh.append({'t': t0.strftime('%H:%M'), 'dir': 'UP' if mv > 0 else 'DOWN', 'move_tick': round(float(mv), 1),
                   'buy_share': round(float(dom), 2), 'spr_max_tick': round(float(sp_.max()), 1)})
OUT['liquidity_hole'] = {'median_spread_tick': med_spr, 'events': lh}

# ---------------------------------------------------------------- 均线 / 近期
cl = day.close.values
OUT['ma'] = {'MA5': round(float(cl[-5:].mean()), 3), 'MA10': round(float(cl[-10:].mean()), 3),
             'MA20': round(float(cl[-20:].mean()), 3), 'last5': [float(x) for x in cl[-5:][::-1]],
             'ret_3d_pct': round((cl[-1] / cl[-4] - 1) * 100, 2), 'ret_5d_pct': round((cl[-1] / cl[-6] - 1) * 100, 2),
             'ret_20d_pct': round((cl[-1] / cl[-21] - 1) * 100, 2),
             'hi_52w': float(day.high.iloc[-252:].max()), 'lo_52w': float(day.low.iloc[-252:].min())}

with open(BASE + '14_watson_selfcalc.json', 'w') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=str)
print('written', BASE + '14_watson_selfcalc.json')
