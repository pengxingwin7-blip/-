# -*- coding: utf-8 -*-
"""§4 压单/托单AI判断辅助 + 关键时段复核"""
import sys
import pandas as pd, numpy as np
D = sys.argv[1]; ET = 'America/New_York'; TICK = 0.01
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
t0 = pd.Timestamp('2026-10-08 09:30:00', tz=ET); t1 = pd.Timestamp('2026-10-08 16:00:00', tz=ET)
dp = pd.read_parquet(f'{D}/07_SVCO_depth5_2026-10-08.parquet'); dp['size'] = dp['size'].astype('int64'); dp['et'] = dp.ts_event.dt.tz_convert(ET)
tk = pd.read_parquet(f'{D}/08_SVCO_tick_2026-10-08.parquet'); tk['size'] = tk['size'].astype('int64'); tk['et'] = tk.ts_event.dt.tz_convert(ET)
lit = tk[(tk.action == 'T') & (tk.et >= t0 + pd.Timedelta('1s')) & (tk.et < t1)]
dp['key'] = list(zip(dp.publisher_id, dp.channel_id, dp.order_id))
o = dp[dp.action.isin(['A', 'C', 'F', 'M']) & (dp.order_id > 0)]
A = o[o.action == 'A'].drop_duplicates('key').set_index('key')
lv = np.zeros(len(A), int)
for k in range(5, 0, -1):
    lv[(((A.side == 'B') & (A[f'bid_p{k}'] == A.price)) | ((A.side == 'A') & (A[f'ask_p{k}'] == A.price))).values] = k
A['lvl'] = lv
F = o[o.action == 'F'].groupby('key')['size'].sum().rename('fill'); C = o[o.action == 'C'].groupby('key').et.max().rename('c_t')
X = A.join(F).join(C); X['fill'] = X.fill.fillna(0)
X['end'] = X.c_t.fillna(t1); X['life'] = (X.end - X.et).dt.total_seconds()
W = X[(X.et >= t0) & (X.et < t1) & (X.lvl.between(1, 3)) & (X['size'] >= 1000) & (X.life > 30)].copy()
# 对面成交：墙存活期内，对手方向主动成交量（Ask墙→同期主动买；Bid墙→同期主动卖），以及存活期内价格位移
rows = []
for k, w in W.iterrows():
    seg = lit[(lit.et >= w.et) & (lit.et <= w.end)]
    opp = seg[seg.side == ('B' if w.side == 'A' else 'A')]['size'].sum(); same = seg[seg.side == w.side]['size'].sum()
    after = lit[(lit.et > w.end) & (lit.et <= w.end + pd.Timedelta('15min'))]
    p_after = after.price.iloc[-1] if len(after) else np.nan
    rows.append((w.et.strftime('%H:%M:%S'), w.side, w.price, int(w['size']), f'L{w.lvl}', round(w.life), int(w.fill), '撤' if pd.notna(w.c_t) else '存续', int(opp), int(same), p_after, w.publisher_id))
print('== L1-L3 大单墙 (size>=1000, life>30s)')
print(pd.DataFrame(rows, columns=['挂出', '侧', '价', '量', '档', '存活s', '成交', '结局', '对手主动量', '同向主动量', '撤后15m价', 'pub']).to_string(index=False))

# 11:40 事件前后 DOM、事件后5/15min价格
for t in ['11:39:00', '11:40:16', '11:40:17.4', '11:40:30', '11:45:17', '11:55:17', '12:17:40', '12:18:30', '15:41:00', '15:59:55', '16:00:01']:
    tt = pd.Timestamp(f'2026-10-08 {t}', tz=ET); r = dp[dp.et <= tt].iloc[-1]
    print(t, 'bid', [(r[f'bid_p{k}'], r[f'bid_s{k}']) for k in range(1, 6)], '| ask', [(r[f'ask_p{k}'], r[f'ask_s{k}']) for k in range(1, 6)])
L = lit[lit.side.isin(['A', 'B'])]
ev = L[(L.et >= pd.Timestamp('2026-10-08 11:40:17.13', tz=ET)) & (L.et < pd.Timestamp('2026-10-08 11:40:17.35', tz=ET))]
print('事件逐venue:', ev.groupby(['publisher_id', 'side'])['size'].sum().to_dict(), '价位分布卖:', ev[ev.side == 'A'].groupby('price')['size'].sum().to_dict())
# 事件剔除后 LOW 带吸收率
Z1 = 7.90 + 0.4 / 3
low = L[(L.price <= Z1) & ~L.index.isin(ev.index)]
ta, tb = low[low.side == 'A']['size'].sum(), low[low.side == 'B']['size'].sum()
print(f'LOW带(剔除11:40事件) T(A)/被动买承接={ta} T(B)={tb} Bid吸率={ta/(ta+tb)*100:.1f}%')
# 12:17 冲击 8.30 的过程
s = lit[(lit.et >= pd.Timestamp('2026-10-08 12:15', tz=ET)) & (lit.et < pd.Timestamp('2026-10-08 12:25', tz=ET))]
print('12:15-12:25 主动买/卖', s[s.side == 'B']['size'].sum(), s[s.side == 'A']['size'].sum(), '高', s.price.max(), '末价', s.price.iloc[-1])
# 09:36:38 流动性洞方向
s = lit[(lit.et >= pd.Timestamp('2026-10-08 09:36:36', tz=ET)) & (lit.et < pd.Timestamp('2026-10-08 09:36:41', tz=ET))]
print('09:36:38 段', s.groupby('side')['size'].sum().to_dict(), s.price.tolist()[:20])
s = lit[(lit.et >= pd.Timestamp('2026-10-08 09:32:38', tz=ET)) & (lit.et < pd.Timestamp('2026-10-08 09:32:43', tz=ET))]
print('09:32:40 段', s.groupby('side')['size'].sum().to_dict(), s.price.tolist()[:20])
# 15:40-16:00 跌破7.93至7.90过程
s = lit[(lit.et >= pd.Timestamp('2026-10-08 15:30', tz=ET))]
s = s.assign(b=s.et.dt.floor('5min'))
print('尾盘5min:', {k.strftime('%H:%M'): (int(g[g.side=='B']['size'].sum()), int(g[g.side=='A']['size'].sum()), g.price.min(), g.price.iloc[-1]) for k, g in s.groupby('b')})
