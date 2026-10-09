"""
华生 v4.1 §2/§3 自算部分（CRDL 2026-10-08）
数据口径：
  - 场内主动成交：MBO action=T（publisher 2 XNAS / 5 BATS / 9 XNYS / 43 ARCX），side=B买方主动 / A卖方主动 / N无方向
  - TRF：trades.parquet 中 publisher 82(FINN) / 83(FINC)；81=XNAS交易所(与MBO重复且竞价重复计数)剔除；88 XBOS/89 XPSX 计入场内其他
  - 盘口：rebuild_book.py 从MBO重建的4场所合并L1~L5（证据包depth5的ask_p1为幽灵档，弃用）
"""
import sys
import json
import numpy as np
import pandas as pd

S = sys.argv[1]
P = sys.argv[2]
TICK = 0.01
dom = pd.read_parquet(f'{S}/dom.parquet')
tr = pd.read_parquet(f'{P}/06_CRDL_ALL_trades_20261008.parquet')
tr['et'] = tr.ts_event.dt.tz_convert('America/New_York')
day = pd.read_csv(f'{P}/03_CRDL_day.csv')
spy = pd.read_csv(f'{P}/04_SPY_day.csv')

D0 = pd.Timestamp('2026-10-08 09:30:00', tz='America/New_York')
D1 = pd.Timestamp('2026-10-08 16:00:01', tz='America/New_York')   # 含16:00:00收盘竞价
rth = lambda x: x[(x.et >= D0) & (x.et < D1)]

out = {}
# ---------------- 成交合并 ----------------
T = dom[dom.action == 'T'].copy()
Tr = rth(T)
trf = tr[tr.publisher_id.isin([82, 83]) & (tr['size'] > 0)].copy()
oth = tr[tr.publisher_id.isin([88, 89]) & (tr['size'] > 0)].copy()
trf_r, oth_r = rth(trf), rth(oth)
prints = pd.concat([
    Tr[['et', 'price', 'size']].assign(src='EXCH'),
    oth_r[['et', 'price', 'size']].assign(src='EXCH_OTH'),
    trf_r[['et', 'price', 'size']].assign(src='TRF')]).sort_values('et')
prints['size'] = prints['size'].astype(float)

vol_ex = Tr['size'].sum() + oth_r['size'].sum()
vol_trf = trf_r['size'].sum()
vol_all_day = T['size'].sum() + oth['size'].sum() + trf['size'].sum()
out['vol'] = dict(rth_exch=int(vol_ex), rth_trf=int(vol_trf), rth_total=int(vol_ex + vol_trf),
                  allday_captured=int(vol_all_day), day_csv=int(day.iloc[-1].volume))
# 覆盖率 = 带DOM的4场所MBO成交 / 全日成交(day.csv)
out['coverage_mbo_vs_daycsv'] = round(T['size'].sum() / day.iloc[-1].volume * 100, 1)
out['coverage_mbo_vs_rth_captured'] = round(Tr['size'].sum() / (vol_ex + vol_trf) * 100, 1)

# ---------------- OHLC ----------------
xn = Tr[Tr.publisher_id == 2]
open_cross = xn[(xn.side == 'N')].iloc[0]
close_cross = xn[(xn.side == 'N') & (xn.et >= pd.Timestamp('2026-10-08 16:00:00', tz='America/New_York'))]
# 价格极值只用场内（TRF存在非实时/衍生定价）
ex_prints = prints[prints.src != 'TRF']
hi = ex_prints.loc[ex_prints.price.idxmax()]
lo = ex_prints.loc[ex_prints.price.idxmin()]
close_px = float(close_cross.price.iloc[0])
prev_close = float(day.iloc[-2].close)
out['ohlc'] = dict(open=float(open_cross.price), open_t=str(open_cross.et.time()), open_size=int(open_cross['size']),
                   high=float(hi.price), high_t=str(hi.et.time()), low=float(lo.price), low_t=str(lo.et.time()),
                   close=close_px, close_cross_size=int(close_cross['size'].sum()), prev_close=prev_close,
                   chg_pct=round((close_px / prev_close - 1) * 100, 2),
                   trf_px_range=[float(trf_r.price.min()), float(trf_r.price.max())])
# 高点的全部触及时间
out['high_touch_times'] = sorted(set(ex_prints[ex_prints.price >= hi.price].et.dt.strftime('%H:%M:%S')))[:12]
out['low_touch_times'] = sorted(set(ex_prints[ex_prints.price <= lo.price].et.dt.strftime('%H:%M:%S')))[:12]

# ---------------- VWAP / SPY / RVOL ----------------
vwap = (prints.price * prints['size']).sum() / prints['size'].sum()
out['vwap'] = round(vwap, 4)
out['close_vs_vwap_pct'] = round((close_px / vwap - 1) * 100, 2)
spy_ret = (spy.iloc[-1].close / spy.iloc[-2].close - 1) * 100
out['spy_pct'] = round(spy_ret, 2)
out['rs_pp'] = round(out['ohlc']['chg_pct'] - spy_ret, 2)
avg20 = day.iloc[-21:-1].volume.mean()
avg5 = day.iloc[-6:-1].volume.mean()
out['avg20_vol'] = int(avg20)
out['avg5_vol'] = int(avg5)
out['rvol20'] = round(day.iloc[-1].volume / avg20, 2)
out['rvol5'] = round(day.iloc[-1].volume / avg5, 2)

# ---------------- Delta ----------------
tb = Tr[Tr.side == 'B']['size'].sum()
ta = Tr[Tr.side == 'A']['size'].sum()
tn = Tr[Tr.side == 'N']['size'].sum()
F = rth(dom[dom.action == 'F'])
fa = F[F.side == 'A']['size'].sum()
fb = F[F.side == 'B']['size'].sum()
out['delta'] = dict(TB=int(tb), TA=int(ta), net=int(tb - ta), TN=int(tn),
                    FA=int(fa), FB=int(fb), identity_TB_eq_FA=bool(tb == fa), identity_TA_eq_FB=bool(ta == fb))

# 聚合主动单：同publisher+同ts_event+同side 视为同一笔主动委托（跨多档成交）
agg = Tr[Tr.side.isin(['B', 'A'])].groupby(['publisher_id', 'ts_event', 'side']).agg(
    size=('size', 'sum'), pmin=('price', 'min'), pmax=('price', 'max'), n=('size', 'count')).reset_index()
agg['et'] = agg.ts_event.dt.tz_convert('America/New_York')
q = agg['size'].quantile([.90, .95, .98, .99]).round(0)
out['aggr_pct'] = {k: int(v) for k, v in zip(['P90', 'P95', 'P98', 'P99'], q.values)}
p95 = q.loc[.95]
big, small = agg[agg['size'] >= p95], agg[agg['size'] < p95]
f = lambda x, s: int(x[x.side == s]['size'].sum())
out['delta_layers'] = dict(big_TB=f(big, 'B'), big_TA=f(big, 'A'), big_net=f(big, 'B') - f(big, 'A'),
                           small_TB=f(small, 'B'), small_TA=f(small, 'A'), small_net=f(small, 'B') - f(small, 'A'),
                           big_n=len(big))
# 原始逐笔口径的分位（供对照）
q2 = Tr[Tr.side.isin(['B', 'A'])]['size'].quantile([.90, .95, .98, .99]).round(0)
out['print_pct'] = {k: int(v) for k, v in zip(['P90', 'P95', 'P98', 'P99'], q2.values)}

# CVD
c = Tr[Tr.side.isin(['B', 'A'])].copy()
c['d'] = np.where(c.side == 'B', c['size'].astype(float), -c['size'].astype(float))
c['cvd'] = c.d.cumsum()
out['cvd'] = dict(peak=int(c.cvd.max()), peak_t=str(c.loc[c.cvd.idxmax()].et.time())[:8],
                  trough=int(c.cvd.min()), trough_t=str(c.loc[c.cvd.idxmin()].et.time())[:8], end=int(c.cvd.iloc[-1]))
for lab, t0 in [('last2h', '14:00:00'), ('last1h', '15:00:00'), ('last30m', '15:30:00')]:
    x = c[c.et >= pd.Timestamp(f'2026-10-08 {t0}', tz='America/New_York')]
    out[f'delta_{lab}'] = int(x.d.sum())
# 每30min Delta / 量 / 收盘
c['bar'] = c.et.dt.floor('30min').dt.strftime('%H:%M')
prints['bar'] = prints.et.dt.floor('30min').dt.strftime('%H:%M')
bars = prints.groupby('bar').agg(o=('price', 'first'), h=('price', 'max'), l=('price', 'min'), cl=('price', 'last'), v=('size', 'sum'))
bars['delta'] = c.groupby('bar').d.sum()
bars['cvd_end'] = c.groupby('bar').cvd.last()
out['bars30'] = bars.reset_index().round(4).to_dict('records')
out['tail30_vol'] = int(prints[prints.et >= pd.Timestamp('2026-10-08 15:30', tz='America/New_York')]['size'].sum())
out['tail30_pct'] = round(out['tail30_vol'] / prints['size'].sum() * 100, 1)
out['close_vs_low_pct'] = round((close_px - lo.price) / close_px * 100, 2)

# ---------------- 量价分布 POC/VAH/VAL / ORH/ORL ----------------
pr = prints.copy()
pr['b'] = (pr.price / TICK).round().astype(int)
vp = pr.groupby('b')['size'].sum().sort_index()
poc = vp.idxmax()
tot = vp.sum()
inc = {poc}
lo_i, hi_i = poc, poc
acc = vp[poc]
while acc < 0.7 * tot:
    up = vp.get(hi_i + 1, 0)
    dn = vp.get(lo_i - 1, 0)
    if up == 0 and dn == 0:
        break
    if up >= dn:
        hi_i += 1
        acc += up
    else:
        lo_i -= 1
        acc += dn
or_ = ex_prints[ex_prints.et < pd.Timestamp('2026-10-08 10:00', tz='America/New_York')]
out['levels'] = dict(POC=poc * TICK, VAH=hi_i * TICK, VAL=lo_i * TICK, ORH=float(or_.price.max()), ORL=float(or_.price.min()))
out['vprofile'] = {f'{k * TICK:.2f}': int(v) for k, v in vp.items()}

json.dump(out, open(f'{S}/watson_core.json', 'w'), ensure_ascii=False, indent=1, default=str)
agg.to_parquet(f'{S}/aggr.parquet')
prints.to_parquet(f'{S}/prints.parquet')
for k, v in out.items():
    if k not in ('vprofile', 'bars30'):
        print(k, v)
print(pd.DataFrame(out['bars30']).to_string())
print(out['vprofile'])
