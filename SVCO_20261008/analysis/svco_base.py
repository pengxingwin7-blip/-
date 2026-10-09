# -*- coding: utf-8 -*-
"""
SVCO 2026-10-08 华生§2 基础快照 + 数据口径核查
- 统一时间：UTC -> ET；RTH = 09:30:00~16:00:00 ET
- 成交来源：08_tick（4个场内MBO venue的T事件，含主动方向）+ 06_trades 中 pub82/83 (FINRA TRF)
- pub81 = Nasdaq场内成交(与ITCH逐笔纳秒级匹配)，非TRF；其 20:00:00.8996 的52笔flags=0成交疑似收盘竞价分配重复，剔除
"""
import sys, json
import pandas as pd, numpy as np
D = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
ET = 'America/New_York'

tk = pd.read_parquet(f'{D}/08_SVCO_tick_2026-10-08.parquet')
tr = pd.read_parquet(f'{D}/06_SVCO_ALL_trades_20261008.parquet')
mbo = pd.read_parquet(f'{D}/05_SVCO_ALL_mbo_20261008.parquet')
day = pd.read_csv(f'{D}/03_SVCO_day.csv', parse_dates=['date'])
spy = pd.read_csv(f'{D}/04_SPY_day.csv', parse_dates=['date'])
for df in (tk, tr, mbo):
    df['et'] = df.ts_event.dt.tz_convert(ET)

t0 = pd.Timestamp('2026-10-08 09:30:00', tz=ET); t1 = pd.Timestamp('2026-10-08 16:00:00', tz=ET)
rth = lambda df: df[(df.et >= t0) & (df.et < t1 + pd.Timedelta('2s'))]

# ---------- 场内（lit）成交 ----------
lit = rth(tk[tk.action == 'T']).copy()
# 识别竞价：XNAS side N 且 size 很大 / 09:30:00 开盘竞价
cross_close = lit[(lit.publisher_id == 2) & (lit.side == 'N') & (lit.et >= t1)]
cross_open = lit[(lit.publisher_id == 2) & (lit.side == 'N') & (lit.et < t0 + pd.Timedelta('1s'))]
print('开盘竞价', cross_open[['et', 'price', 'size']].to_string())
print('收盘竞价', cross_close[['et', 'price', 'size']].to_string())
cont = lit[(lit.et < t1) & ~lit.index.isin(cross_open.index)]
# TRF
trf = rth(tr[tr.publisher_id.isin([82, 83])]).copy()
trf = trf[(trf['size'] > 0) & (trf.et < t1 + pd.Timedelta('2s'))]
oth = rth(tr[tr.publisher_id.isin([88, 89])])
print('lit连续竞价量', cont['size'].sum(), '笔', len(cont), '| TRF量', trf['size'].sum(), '笔', len(trf), '| pub88/89量', oth['size'].sum())
print('lit by venue/side:'); print(cont.groupby(['publisher_id', 'side'])['size'].sum().unstack(fill_value=0))

# ---------- OHLC 校核 ----------
allp = pd.concat([cont[['et', 'price', 'size']], trf[['et', 'price', 'size']], cross_open[['et', 'price', 'size']], cross_close[['et', 'price', 'size']]]).sort_values('et')
rl = allp[allp['size'] >= 100]
print('RTH全部成交: H', allp.price.max(), allp.loc[allp.price.idxmax(), 'et'], 'L', allp.price.min(), allp.loc[allp.price.idxmin(), 'et'])
print('RTH整手(>=100)成交: H', rl.price.max(), rl.loc[rl.price.idxmax(), 'et'].strftime('%H:%M:%S'), 'L', rl.price.min(), rl.loc[rl.price.idxmin(), 'et'].strftime('%H:%M:%S'))
print('<=7.91 成交:'); print(allp[allp.price <= 7.91].to_string())
print('>=8.29 成交:'); print(allp[allp.price >= 8.29].to_string())
pre = tk[(tk.action == 'T') & (tk.et < t0)]; post = tk[(tk.action == 'T') & (tk.et > t1 + pd.Timedelta('2s'))]
print('盘前成交', pre[['et', 'publisher_id', 'price', 'size']].to_string()); print('盘后成交量', post['size'].sum(), post.price.unique())

# ---------- VWAP（真实） ----------
v_all = allp; vwap = (v_all.price * v_all['size']).sum() / v_all['size'].sum()
v_cont = pd.concat([cont[['price', 'size']], trf[['price', 'size']]])
vwap_c = (v_cont.price * v_cont['size']).sum() / v_cont['size'].sum()
print(f'VWAP(含竞价)={vwap:.4f}  VWAP(连续竞价lit+TRF)={vwap_c:.4f}')

# ---------- 量能 / RVOL ----------
d = day.set_index('date')
prev20 = d.loc[:'2026-10-07'].tail(20)
print('20日均量', prev20.volume.mean().round(0), '| 今日', d.loc['2026-10-08', 'volume'], 'RVOL', round(d.loc['2026-10-08', 'volume'] / prev20.volume.mean(), 2))
cl = d.close
print('MA5', cl.tail(5).mean().round(3), 'MA10', cl.tail(10).mean().round(3), 'MA20', cl.tail(20).mean().round(3), 'MA50', cl.tail(50).mean().round(3))
print('近6日收盘(新->旧)', cl.tail(6)[::-1].tolist())
print('近3日涨跌', round((cl.iloc[-1] / cl.iloc[-4] - 1) * 100, 2), '近5日', round((cl.iloc[-1] / cl.iloc[-6] - 1) * 100, 2), '距10/1高点9.72', round((cl.iloc[-1] / 9.72 - 1) * 100, 2))
s = spy.set_index('date').close
print('SPY 10-08', round((s.iloc[-1] / s.iloc[-2] - 1) * 100, 3))
print('20日最高/最低收盘', cl.tail(20).max(), cl.tail(20).min(), '20日最高/最低', d.high.tail(20).max(), d.low.tail(20).min())

# ---------- 场内识别：T(B)=F(A) 恒等式 ----------
f = rth(mbo[mbo.action == 'F'])
f = f[f.et < t1]
for p in [2, 5, 9, 43]:
    tb = cont[(cont.publisher_id == p) & (cont.side == 'B')]['size'].sum(); ta = cont[(cont.publisher_id == p) & (cont.side == 'A')]['size'].sum()
    fa = f[(f.publisher_id == p) & (f.side == 'A')]['size'].sum(); fb = f[(f.publisher_id == p) & (f.side == 'B')]['size'].sum()
    print(f'pub{p}: T(B)={tb} F(A)={fa} | T(A)={ta} F(B)={fb}')
tb = cont[cont.side == 'B']['size'].sum(); ta = cont[cont.side == 'A']['size'].sum(); tn = cont[cont.side == 'N']['size'].sum()
fa = f[f.side == 'A']['size'].sum(); fb = f[f.side == 'B']['size'].sum()
print(f'合计 T(B)={tb} T(A)={ta} T(N 隐藏/未知)={tn} 净Delta={tb-ta} | F(A)={fa} F(B)={fb}')

# ---------- Volume Profile (连续竞价 lit+TRF, $0.01 bin) ----------
vp = v_cont.copy(); vp['bin'] = (vp.price + 1e-9).round(2)
prof = vp.groupby('bin')['size'].sum().sort_index()
poc = prof.idxmax(); tot = prof.sum()
# 70% value area：从POC向两侧扩展
idx = list(prof.index); i = idx.index(poc); lo = hi = i; acc = prof.iloc[i]
while acc < 0.7 * tot:
    up = prof.iloc[hi + 1] if hi + 1 < len(idx) else -1
    dn = prof.iloc[lo - 1] if lo - 1 >= 0 else -1
    if up >= dn: hi += 1; acc += up
    else: lo -= 1; acc += dn
print('POC', poc, prof.max(), 'VAH', idx[hi], 'VAL', idx[lo], 'VA占比', round(acc / tot, 3))
print('Top8价位量:'); print(prof.sort_values(ascending=False).head(8))
# Opening range (前30分钟)
orr = pd.concat([cont, trf])
orr = orr[(orr.et >= t0) & (orr.et < t0 + pd.Timedelta('30min'))]
print('OR全部 H/L', orr.price.max(), orr.price.min(), '| OR整手 H/L', orr[orr['size'] >= 100].price.max(), orr[orr['size'] >= 100].price.min())

# ---------- 30min 分段：量、Delta、OHLC ----------
cc = pd.concat([cont.assign(src='lit'), trf.assign(src='trf', side='N')])
cc['bar'] = cc.et.dt.floor('30min')
g = cc.groupby('bar')
bars = pd.DataFrame({'O': g.price.first(), 'H': g.price.max(), 'L': g.price.min(), 'C': g.price.last(),
                     'vol_lit': cc[cc.src == 'lit'].groupby('bar')['size'].sum(), 'vol_trf': cc[cc.src == 'trf'].groupby('bar')['size'].sum(),
                     'TB': cc[cc.side == 'B'].groupby('bar')['size'].sum(), 'TA': cc[cc.side == 'A'].groupby('bar')['size'].sum()}).fillna(0)
bars['delta'] = bars.TB - bars.TA
bars.index = bars.index.strftime('%H:%M')
print(bars.to_string())
# 尾盘30min / 2h
last30 = cc[cc.et >= pd.Timestamp('2026-10-08 15:30', tz=ET)]; last2h = cc[cc.et >= pd.Timestamp('2026-10-08 14:00', tz=ET)]
print('尾盘30min 连续量', last30['size'].sum(), 'Delta', int(last30[last30.side == 'B']['size'].sum()) - int(last30[last30.side == 'A']['size'].sum()), '+收盘竞价', cross_close['size'].sum())
print('尾盘2h Delta', int(last2h[last2h.side == 'B']['size'].sum()) - int(last2h[last2h.side == 'A']['size'].sum()))
# 覆盖率
dayvol = d.loc['2026-10-08', 'volume']
print('覆盖率(4所MBO连续/连续综合量)', round(cont['size'].sum() / (dayvol - cross_close['size'].sum() - cross_open['size'].sum()) * 100, 1),
      '| (4所MBO+TRF)/连续综合量', round((cont['size'].sum() + trf['size'].sum()) / (dayvol - cross_close['size'].sum() - cross_open['size'].sum()) * 100, 1))
