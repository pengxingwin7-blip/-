"""
华生 v4.1 §2/§3 自算脚本 —— 从 MBO / depth5 / tick / TRF 原始数据计算盘口指标。
用法: python3 -I watson_selfcalc.py <证据包目录，如 ADCT_20261008> <YYYYMMDD>
所有时间输出为美东时间 (ET)。RTH = 09:30:00 ~ 16:00:05（含收盘集合竞价成交）。
"""
import sys, json
import numpy as np
import pandas as pd

D, DAY = sys.argv[1], sys.argv[2]
import os
TICKER = os.path.basename(os.path.normpath(D)).split('_')[0]  # 从目录名推断 ticker
DATE = f'{DAY[:4]}-{DAY[4:6]}-{DAY[6:]}'
TICK = 0.01
TZ = 'America/New_York'
out = {}

# ---------------- 读数据 ----------------
dep = pd.read_parquet(f'{D}/07_{TICKER}_depth5_{DATE}.parquet')
tick = pd.read_parquet(f'{D}/08_{TICKER}_tick_{DATE}.parquet')
trf = pd.read_parquet(f'{D}/06_{TICKER}_ALL_trades_{DAY}.parquet')
day = pd.read_csv(f'{D}/03_{TICKER}_day.csv', parse_dates=['date'])
spy = pd.read_csv(f'{D}/04_SPY_day.csv', parse_dates=['date'])
for df in (dep, tick, trf):
    df['et'] = df.ts_event.dt.tz_convert(TZ)
    df['hms'] = df.et.dt.strftime('%H:%M:%S.%f')

def rth(df):
    return df[(df.hms >= '09:30:00') & (df.hms <= '16:00:05')]

T_START = pd.Timestamp(f'{DATE} 09:30:00', tz=TZ)

# depth5 时间索引：用于"某时刻之前最后一行DOM"查询（DOM为事件后状态，跨4个交易所合并）
dep = dep.sort_values(['ts_event', 'sequence']).reset_index(drop=True)
dep_ts = dep.ts_event.values
W = np.array([1.0, 0.6, 0.35, 0.2, 0.1])  # 委比L1~L5权重

def dom_at(ts):
    """返回 ts 时刻(含)之前最后一行 DOM"""
    i = np.searchsorted(dep_ts, np.datetime64(ts.tz_convert('UTC').tz_localize(None)), side='right') - 1
    return dep.iloc[max(i, 0)]

def imbalance(row):
    """加权委比 = (加权Bid - 加权Ask)/(加权Bid + 加权Ask) × 100"""
    b = sum(W[k] * row[f'bid_s{k+1}'] for k in range(5))
    a = sum(W[k] * row[f'ask_s{k+1}'] for k in range(5))
    return b, a, (b - a) / (b + a) * 100 if (b + a) > 0 else np.nan

# ---------------- §2 基础快照 ----------------
lit = tick[tick.action == 'T'].copy()
lit_r = rth(lit)
trf_r = rth(trf)
d0 = day[day.date == DATE].iloc[0]
prev = day[day.date < DATE].iloc[-1]
avg20 = day[day.date < DATE].tail(20).volume.mean()
spy0 = spy[spy.date == DATE].iloc[0]; spyp = spy[spy.date < DATE].iloc[-1]
spy_ret = (spy0.close / spyp.close - 1) * 100
ret = (d0.close / prev.close - 1) * 100

# 集合竞价（NYSE side=N 大单）
open_px = lit_r[lit_r.publisher_id == 9].iloc[0].price if len(lit_r) else np.nan
close_auc = lit[(lit.hms >= '16:00:00') & (lit.hms <= '16:00:05') & (lit.side == 'N')]
close_px = close_auc.price.iloc[0] if len(close_auc) else d0.close
hi_row = lit_r.loc[lit_r.price.idxmax()]; lo_row = lit_r.loc[lit_r.price.idxmin()]

# 主动成交聚合：同一 ts_event + publisher + side 视为同一笔主动单
agg = (lit_r[lit_r.side.isin(['B', 'A'])]
       .groupby(['ts_event', 'publisher_id', 'side'], as_index=False)
       .agg(size=('size', 'sum'), price=('price', 'last'), pmin=('price', 'min'), pmax=('price', 'max'), et=('et', 'first')))
agg = agg.sort_values('ts_event').reset_index(drop=True)
P = {q: float(np.percentile(agg['size'], q)) for q in (90, 95, 98, 99, 99.5)}
agg['sgn'] = np.where(agg.side == 'B', 1, -1)
agg['d'] = agg['size'] * agg.sgn
TB = int(agg[agg.side == 'B']['size'].sum()); TA = int(agg[agg.side == 'A']['size'].sum())

# 恒等式 T(B) ≡ F(A)
F = rth(dep[dep.action == 'F'])
FA = int(F[F.side == 'A']['size'].sum()); FB = int(F[F.side == 'B']['size'].sum())

# VWAP（lit + TRF, RTH）
allp = pd.concat([lit_r[['et', 'price', 'size']], trf_r[['et', 'price', 'size']]])
vwap_calc = (allp.price * allp['size']).sum() / allp['size'].sum()

# 量价分布 POC / VAH / VAL（$0.01 分箱，70% 价值区）
allp['bin'] = (allp.price / TICK).round().astype(int)
prof = allp.groupby('bin')['size'].sum().sort_index()
poc = prof.idxmax(); tot = prof.sum(); inc = {poc}; cum = prof[poc]
lo_i, hi_i = poc, poc
bins = prof.index.tolist()
while cum < 0.7 * tot:
    up = prof.get(hi_i + 1, 0) if hi_i + 1 <= max(bins) else -1
    dn = prof.get(lo_i - 1, 0) if lo_i - 1 >= min(bins) else -1
    if up < 0 and dn < 0: break
    if up >= dn: hi_i += 1; cum += max(up, 0)
    else: lo_i -= 1; cum += max(dn, 0)
POC, VAH, VAL = poc * TICK, hi_i * TICK, lo_i * TICK

# 开盘区间（前30min）
orr = lit_r[(lit_r.hms < '10:00:00')]
ORH, ORL = orr.price.max(), orr.price.min()

# 尾盘30min
tail30 = agg[agg.et.dt.strftime('%H:%M') >= '15:30']
tail30_vol = int(lit_r[lit_r.hms >= '15:30']['size'].sum() + trf_r[trf_r.hms >= '15:30']['size'].sum())
tail2h_delta = int(agg[agg.et.dt.strftime('%H:%M') >= '14:00'].d.sum())

lit_vol_day = int(lit['size'].sum()); trf_vol_day = int(trf['size'].sum())
out['snapshot'] = dict(
    day_csv=dict(open=d0.open, high=d0.high, low=d0.low, close=d0.close, volume=int(d0.volume), vwap=d0.VWAP),
    tick_open_auction=open_px, tick_close_auction=close_px,
    rth_high=(hi_row.price, hi_row.et.strftime('%H:%M:%S')), rth_low=(lo_row.price, lo_row.et.strftime('%H:%M:%S')),
    prev_close=prev.close, ret_pct=round(ret, 2),
    lit_vol_day=lit_vol_day, trf_vol_day=trf_vol_day, sum_data=lit_vol_day + trf_vol_day,
    avg20=int(avg20), rvol=round(d0.volume / avg20, 2),
    TB=TB, TA=TA, delta=TB - TA, FA=FA, FB=FB,
    vwap_calc=round(vwap_calc, 4), close_vs_vwap_pct=round((d0.close / d0.VWAP - 1) * 100, 2),
    spy_ret=round(spy_ret, 2), rs_pp=round(ret - spy_ret, 2),
    tail30_vol=tail30_vol, tail30_pct=round(tail30_vol / (lit_r['size'].sum() + trf_r['size'].sum()) * 100, 1),
    tail30_delta=int(tail30.d.sum()), tail2h_delta=tail2h_delta,
    ORH=ORH, ORL=ORL, POC=round(POC, 2), VAH=round(VAH, 2), VAL=round(VAL, 2),
    coverage_pct=round((lit_vol_day + trf_vol_day) / d0.volume * 100, 1),
    lit_coverage_pct=round(lit_vol_day / (d0.volume - trf_vol_day) * 100, 1),
    pct=P,
)

# ---------------- §3.1 委比关键时刻 ----------------
snaps = []
for h in pd.date_range(f'{DATE} 09:30', f'{DATE} 15:30', freq='30min', tz=TZ):
    r = dom_at(h + pd.Timedelta(seconds=1)); b, a, im = imbalance(r)
    snaps.append(dict(t=h.strftime('%H:%M:%S'), trig='30min bar', wb=round(b), wa=round(a), imb=round(im, 1), dirn='—'))
big = agg[agg['size'] >= P[99]].copy()
big_rows = []
for _, x in big.iterrows():
    r = dom_at(x.et - pd.Timedelta(seconds=1)); b, a, im = imbalance(r)
    with_trend = (im > 0 and x.side == 'B') or (im < 0 and x.side == 'A')
    big_rows.append(dict(t=x.et.strftime('%H:%M:%S'), side=x.side, size=int(x['size']), px=x.price,
                         wb=round(b), wa=round(a), imb=round(im, 1), dirn='顺' if with_trend else '逆'))
big_df = pd.DataFrame(big_rows)
lvl_rows = []
after10 = lit_r[lit_r.hms >= '10:00:00']
for name, lv in [('ORH', ORH), ('ORL', ORL), ('VAH', VAH), ('VAL', VAL), ('POC', POC)]:
    hit = after10[(after10.price - lv).abs() < 1e-6]
    if len(hit):
        t0 = hit.iloc[0].et; r = dom_at(t0 - pd.Timedelta(seconds=1)); b, a, im = imbalance(r)
        lvl_rows.append(dict(t=t0.strftime('%H:%M:%S'), trig=f'触及{name} {lv:.2f}', wb=round(b), wa=round(a), imb=round(im, 1)))
    else:
        lvl_rows.append(dict(t='-', trig=f'{name} {lv:.2f} 10:00后未触及'))
out['imb_bars'] = snaps
out['imb_big_summary'] = dict(n=len(big_df), with_trend=int((big_df.dirn == '顺').sum()) if len(big_df) else 0,
                              B=int((big_df.side == 'B').sum()) if len(big_df) else 0, A=int((big_df.side == 'A').sum()) if len(big_df) else 0)
out['imb_big_top'] = big_df.sort_values('size', ascending=False).head(8).to_dict('records') if len(big_df) else []
out['imb_levels'] = lvl_rows
out['imb_bar_mean'] = round(float(np.nanmean([s['imb'] for s in snaps])), 1)

# ---------------- §3.2 ISO扫单（5ms内同向、≥2价位、总量≥P99） ----------------
ls = lit_r[lit_r.side.isin(['B', 'A'])].sort_values('ts_event').reset_index(drop=True)
iso = []
for sd in ('B', 'A'):
    s = ls[ls.side == sd].reset_index(drop=True)
    if not len(s): continue
    gap = s.ts_event.diff().dt.total_seconds().fillna(1e9) * 1000
    gid = (gap > 5).cumsum()
    for _, g in s.groupby(gid):
        if g.price.nunique() >= 2 and g['size'].sum() >= P[99]:
            iso.append(dict(t=g.et.iloc[0].strftime('%H:%M:%S.%f')[:-3], side=sd, levels=g.price.nunique(),
                            vol=int(g['size'].sum()), pr=f'{g.price.min():.2f}~{g.price.max():.2f}', venues=g.publisher_id.nunique()))
iso_df = pd.DataFrame(iso)
out['iso'] = dict(buy_n=int((iso_df.side == 'B').sum()) if len(iso_df) else 0, buy_v=int(iso_df[iso_df.side == 'B'].vol.sum()) if len(iso_df) else 0,
                  sell_n=int((iso_df.side == 'A').sum()) if len(iso_df) else 0, sell_v=int(iso_df[iso_df.side == 'A'].vol.sum()) if len(iso_df) else 0,
                  top=iso_df.sort_values('vol', ascending=False).head(8).to_dict('records') if len(iso_df) else [])

# ---------------- 订单生命周期（订单键=(publisher_id, channel_id, order_id)） ----------------
ev = dep[dep.action.isin(['A', 'C', 'F', 'M']) & (dep.order_id > 0)].copy()
ev['key'] = ev.publisher_id.astype(str) + '_' + ev.channel_id.astype(str) + '_' + ev.order_id.astype(str)
adds = ev[ev.action == 'A'].drop_duplicates('key').copy()
adds = adds[(adds.price > 0.05) & (adds.price < 100)]  # 剔除 MOC/竞价哨兵价
# 档位：A事件 price 与同行 bid_p1~p5 / ask_p1~p5 逐档比较（DOM已包含该单）
def level_of(r):
    pre = 'bid' if r.side == 'B' else 'ask'
    for k in range(1, 6):
        if abs(r[f'{pre}_p{k}'] - r.price) < 1e-9: return k
    return 99  # OUTSIDE_L5
adds['lvl'] = adds.apply(level_of, axis=1)
# 同侧最优价距离（tick），仅用于 >5档的估算（depth5 无法逐档识别 >5档）
adds['dist_tick'] = np.where(adds.side == 'B', (adds.bid_p1 - adds.price) / TICK, (adds.price - adds.ask_p1) / TICK).round()
adds['dist_tick'] = adds.dist_tick.clip(lower=0)
fills = ev[ev.action == 'F'].groupby('key').agg(filled=('size', 'sum'), f_ts=('ts_event', 'min'))
canc = ev[ev.action == 'C'].groupby('key').agg(cancelled=('size', 'sum'), c_ts=('ts_event', 'max'))
mods = ev[ev.action == 'M'].groupby('key').agg(m_n=('size', 'size'), m_maxsize=('size', 'max'))
life = adds.set_index('key').join(fills).join(canc).join(mods)
life['filled'] = life.filled.fillna(0); life['cancelled'] = life.cancelled.fillna(0)
life['fill_ratio'] = life.filled / life['size']
life['life_s'] = (life.c_ts - life.ts_event).dt.total_seconds()
life['t_travel'] = (life.f_ts - life.ts_event).dt.total_seconds()
A_P = {q: float(np.percentile(adds['size'], q)) for q in (95, 98, 99)}
out['add_size_pct'] = A_P

# ---------------- §3.3 挂单墙（size≥P99 且 A→C 存活>15s，价格在日高低±5tick；按30min分段） ----------------
dl, dh = lo_row.price, hi_row.price
wall = life[(life['size'] >= A_P[99]) & life.c_ts.notna() & (life.life_s > 15) &
            (life.price >= dl - 5 * TICK) & (life.price <= dh + 5 * TICK)].copy()
wall['bar'] = wall.et.dt.floor('30min').dt.strftime('%H:%M')
wb = wall.groupby(['bar', 'side'])['size'].sum().unstack(fill_value=0)
out['wall'] = dict(n=len(wall), B_n=int((wall.side == 'B').sum()), B_v=int(wall[wall.side == 'B']['size'].sum()),
                   A_n=int((wall.side == 'A').sum()), A_v=int(wall[wall.side == 'A']['size'].sum()),
                   by_bar={k: {c: int(v) for c, v in r.items()} for k, r in wb.iterrows()})
wall_rth_inbook = wall[(wall.hms >= '09:30:00') & (wall.lvl <= 5)]
out['wall_rth_L1_5'] = dict(B_n=int((wall_rth_inbook.side == 'B').sum()), B_v=int(wall_rth_inbook[wall_rth_inbook.side == 'B']['size'].sum()),
                            A_n=int((wall_rth_inbook.side == 'A').sum()), A_v=int(wall_rth_inbook[wall_rth_inbook.side == 'A']['size'].sum()))
out['wall_top'] = (wall.sort_values('size', ascending=False).head(10)
                   [['et', 'price', 'side', 'size', 'life_s', 'lvl', 'filled', 'm_n', 'publisher_id']]
                   .assign(et=lambda x: x.et.dt.strftime('%H:%M:%S'), c_et=lambda x: (wall.loc[x.index].c_ts.dt.tz_convert(TZ).dt.strftime('%H:%M:%S')))
                   .to_dict('records'))

# ---------------- §3.4 Delta / CVD ----------------
lg = agg[agg['size'] >= P[95]]; sm = agg[agg['size'] < P[95]]
agg['cvd'] = agg.d.cumsum()
pk = agg.loc[agg.cvd.idxmax()]; tr = agg.loc[agg.cvd.idxmin()]
cvd_bar = agg.groupby(agg.et.dt.floor('30min').dt.strftime('%H:%M')).d.sum().to_dict()
out['delta'] = dict(L_TB=int(lg[lg.side == 'B']['size'].sum()), L_TA=int(lg[lg.side == 'A']['size'].sum()),
                    S_TB=int(sm[sm.side == 'B']['size'].sum()), S_TA=int(sm[sm.side == 'A']['size'].sum()),
                    cvd_peak=(int(pk.cvd), pk.et.strftime('%H:%M:%S')), cvd_trough=(int(tr.cvd), tr.et.strftime('%H:%M:%S')),
                    cvd_end=int(agg.cvd.iloc[-1]), cvd_by_bar={k: int(v) for k, v in cvd_bar.items()},
                    close_vs_low_pct=round((d0.close - lo_row.price) / d0.close * 100, 2))

# ---------------- §3.5 谎骗确认（150s穿越） ----------------
# 口径：RTH 内挂单、size≥P95(A事件)、挂单时在L1~L5、以C结束、fill_ratio<15%
trades_all = pd.concat([lit[['ts_event', 'price']], trf[['ts_event', 'price']]]).sort_values('ts_event')
tp_ts = trades_all.ts_event.values; tp_px = trades_all.price.values
close_ts = pd.Timestamp(f'{DATE} 16:00:00', tz=TZ)
def spoof_eval(sub):
    res = {'B': [0, 0, 0], 'A': [0, 0, 0]}
    for _, o in sub.iterrows():
        c = o.c_ts
        if c + pd.Timedelta(seconds=150) > close_ts: res[o.side][2] += 1; continue
        i0 = np.searchsorted(tp_ts, np.datetime64(c.tz_convert('UTC').tz_localize(None)), side='right')
        i1 = np.searchsorted(tp_ts, np.datetime64((c + pd.Timedelta(seconds=150)).tz_convert('UTC').tz_localize(None)), side='right')
        w = tp_px[i0:i1]
        if len(w) == 0: res[o.side][2] += 1; continue
        crossed = (w < o.price - 1e-9).any() if o.side == 'B' else (w > o.price + 1e-9).any()
        res[o.side][0 if crossed else 1] += 1
    return res
cand = life[(life.hms >= '09:30:00') & (life.hms < '16:00:00') & (life.lvl <= 5) & life.c_ts.notna() & (life.fill_ratio < 0.15)]
sp95 = spoof_eval(cand[cand['size'] >= A_P[95]])
sp99 = spoof_eval(cand[cand['size'] >= A_P[99]])
def rate(r): n = sum(r[:2]); return round(r[0] / n * 100, 1) if n else np.nan
out['spoof_p95'] = {k: v + [rate(v)] for k, v in sp95.items()}
out['spoof_p99'] = {k: v + [rate(v)] for k, v in sp99.items()}
deep = life[(life['size'] >= A_P[98]) & (life.lvl == 99) & (life.dist_tick > 10) & (life.life_s > 120) & (life.filled == 0) & life.c_ts.notna()]
out['deep_cancel'] = dict(B_n=int((deep.side == 'B').sum()), B_v=int(deep[deep.side == 'B']['size'].sum()),
                          A_n=int((deep.side == 'A').sum()), A_v=int(deep[deep.side == 'A']['size'].sum()))
# 极速撤挂比（≤1s 撤单占同侧 C 结束订单比例，RTH）
rc = life[(life.hms >= '09:30:00') & (life.hms < '16:00:00') & life.c_ts.notna()]
out['fast_cancel_pct'] = {sd: round(float((rc[rc.side == sd].life_s <= 1).mean() * 100), 1) for sd in ('B', 'A')}

# ---------------- §3.6 TRF 场外 ----------------
tr_ = trf_r.copy()
doms = [dom_at(t) for t in tr_.et]
tr_['bp1'] = [r.bid_p1 for r in doms]; tr_['ap1'] = [r.ask_p1 for r in doms]
tr_['cls'] = np.where(tr_.price >= tr_.ap1 - 1e-9, 'near_ask', np.where(tr_.price <= tr_.bp1 + 1e-9, 'near_bid', 'mid'))
cv = tr_.groupby('cls')['size'].sum(); tt = tr_['size'].sum()
trf_p995 = np.percentile(tr_['size'], 99.5)
top5 = tr_.nlargest(5, 'size')
top3 = tr_.assign(b=(tr_.price / TICK).round() * TICK).groupby('b')['size'].sum().nlargest(3)
rng = dh - dl
out['trf'] = dict(total=int(tt), near_ask=int(cv.get('near_ask', 0)), near_bid=int(cv.get('near_bid', 0)), mid=int(cv.get('mid', 0)),
                  net=int(cv.get('near_ask', 0)) - int(cv.get('near_bid', 0)), p995=float(trf_p995),
                  top5=[dict(t=r.et.strftime('%H:%M:%S'), px=r.price, size=int(r['size']), cls=r.cls) for _, r in top5.iterrows()],
                  top3=[dict(px=round(p, 2), vol=int(v), pct=round((p - dl) / rng * 100, 1)) for p, v in top3.items()],
                  top3_vw_pct=round(float(((top3.index.values - dl) / rng * 100 * top3.values).sum() / top3.values.sum()), 1))

# ---------------- §3.7 预谋强度（size≥P98 被动成交订单） ----------------
pre = life[(life['size'] >= A_P[98]) & (life.filled > 0)].copy()
def cat(r):
    if r.dist_tick > 10 and r.t_travel > 120: return 'strong'
    if r.lvl <= 2 or r.dist_tick <= 2: return 'realtime'
    return 'fast'
pre['cat'] = pre.apply(cat, axis=1)
pm = {}
for sd in ('B', 'A'):
    s = pre[pre.side == sd]
    pm[sd] = {c: [int((s.cat == c).sum()), int(s[s.cat == c].filled.sum())] for c in ('strong', 'fast', 'realtime')}
    st = s[s.cat == 'strong']
    pm[sd]['strong_top_pub_share'] = round(float(st.groupby('publisher_id').filled.sum().max() / st.filled.sum()), 2) if len(st) else None
    pm[sd]['strong_list'] = (st.sort_values('filled', ascending=False).head(5)
                             .assign(et=lambda x: x.et.dt.strftime('%H:%M:%S'))[['et', 'price', 'size', 'filled', 'dist_tick', 't_travel', 'publisher_id']]
                             .round(1).to_dict('records'))
out['premed'] = pm

# ---------------- 价差 / 流动性洞 粗筛 ----------------
dr = dep[(dep.hms >= '09:31:00') & (dep.hms < '15:59:00')].copy()
dr['spr'] = ((dr.ask_p1 - dr.bid_p1) / TICK).round()
out['spread'] = dict(p50=float(dr.spr.median()), p99=float(dr.spr.quantile(0.99)), max=float(dr.spr.max()),
                     n_ge4=int((dr.spr >= 4).sum()))


# ---------------- §3.7b 预谋强度（仅 RTH 内挂出的订单；盘前 NYSE 簿交叉/锁定，dist 无意义） ----------------
pre_r = pre[(pre.hms >= '09:30:01.1') & (pre.hms < '16:00:00')]
pmr = {}
for sd in ('B', 'A'):
    s = pre_r[pre_r.side == sd]
    pmr[sd] = {c: [int((s.cat == c).sum()), int(s[s.cat == c].filled.sum())] for c in ('strong', 'fast', 'realtime')}
    st = s[s.cat == 'strong']
    pmr[sd]['top_pub_share'] = round(float(st.groupby('publisher_id').filled.sum().max() / st.filled.sum()), 2) if len(st) else None
    pmr[sd]['list'] = (st.sort_values('filled', ascending=False).head(6).assign(et=lambda x: x.et.dt.strftime('%H:%M:%S'),
                        f_et=lambda x: x.f_ts.dt.tz_convert(TZ).dt.strftime('%H:%M:%S'))[['et', 'f_et', 'price', 'size', 'filled', 'dist_tick', 'publisher_id']].to_dict('records'))
out['premed_rth'] = pmr

print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
