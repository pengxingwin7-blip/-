# -*- coding: utf-8 -*-
"""
华生§3 自算7项：委比/ISO/挂单墙/Delta-CVD/谎骗150s/TRF/预谋
数据：05 MBO（订单生命周期）+ 07 depth5（与MBO逐行对齐，DOM为事件后状态）+ 08 tick（场内T事件，side=主动方）+ 06 trades pub82/83（TRF）
"""
import sys
import pandas as pd, numpy as np
D = sys.argv[1]
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60); pd.set_option('display.max_rows', 300)
ET = 'America/New_York'; TICK = 0.01
t0 = pd.Timestamp('2026-10-08 09:30:00', tz=ET); t1 = pd.Timestamp('2026-10-08 16:00:00', tz=ET)
DL, DH = 7.90, 8.30                      # 校正后RTH整手日低/日高
Z1, Z2 = DL + (DH - DL) / 3, DL + 2 * (DH - DL) / 3   # LOW/MID/HIGH 三分带

dp = pd.read_parquet(f'{D}/07_SVCO_depth5_2026-10-08.parquet')
dp['size'] = dp['size'].astype('int64'); dp['et'] = dp.ts_event.dt.tz_convert(ET)
tk = pd.read_parquet(f'{D}/08_SVCO_tick_2026-10-08.parquet'); tk['size'] = tk['size'].astype('int64'); tk['et'] = tk.ts_event.dt.tz_convert(ET)
tr = pd.read_parquet(f'{D}/06_SVCO_ALL_trades_20261008.parquet'); tr['size'] = tr['size'].astype('int64'); tr['et'] = tr.ts_event.dt.tz_convert(ET)

# ---------------- 成交序列 ----------------
lit = tk[(tk.action == 'T') & (tk.et >= t0 + pd.Timedelta('1s')) & (tk.et < t1)].copy()          # 连续竞价（剔除09:30:00.46开盘竞价）
trf = tr[tr.publisher_id.isin([82, 83]) & (tr.et >= t0) & (tr.et < t1 + pd.Timedelta('2s')) & (tr['size'] > 0)].copy()
allt = pd.concat([lit[['et', 'price', 'size']], trf[['et', 'price', 'size']]]).sort_values('et').reset_index(drop=True)
T_ns = allt.et.astype('int64').values; T_px = allt.price.values

# 主动单聚合：同venue+同纳秒+同side 视为1笔主动订单
agg = lit[lit.side.isin(['A', 'B'])].groupby(['publisher_id', 'ts_event', 'side'], as_index=False).agg(size=('size', 'sum'), pmin=('price', 'min'), pmax=('price', 'max'), n=('size', 'size'))
agg['et'] = agg.ts_event.dt.tz_convert(ET); agg = agg.sort_values('et').reset_index(drop=True)
q = agg['size'].quantile([.9, .95, .98, .99]).round(0)
print('== 3.4 主动单(聚合)分位 P90/P95/P98/P99 =', q.tolist(), '| 原始逐笔分位', lit[lit.side.isin(['A','B'])]['size'].quantile([.9,.95,.98,.99]).tolist())
P95, P99a = q.loc[.95], q.loc[.99]
big = agg[agg['size'] >= P95]; small = agg[agg['size'] < P95]
for nm, x in [('大单>=P95', big), ('散单<P95', small)]:
    b = x[x.side == 'B']['size'].sum(); a = x[x.side == 'A']['size'].sum(); print(f'{nm}: T(B)={b} T(A)={a} 净={b-a} 笔数B/A={len(x[x.side=="B"])}/{len(x[x.side=="A"])}')
# 剔除 11:40:17 crossed-book 事件后的 Delta 敏感性
ev = (agg.et >= pd.Timestamp('2026-10-08 11:40:17.13', tz=ET)) & (agg.et < pd.Timestamp('2026-10-08 11:40:17.35', tz=ET))
e = agg[ev]; print('11:40:17.13-.35 事件: T(B)=', e[e.side == 'B']['size'].sum(), 'T(A)=', e[e.side == 'A']['size'].sum(), '| ARCX@8.00买方=', e[(e.side=='B')&(e.publisher_id==43)&(e.pmin==8.0)]['size'].sum())
x = agg[~ev]; print('剔除该事件后 全日Delta=', x[x.side == 'B']['size'].sum() - x[x.side == 'A']['size'].sum(),
                   '大单净=', x[(x['size'] >= P95) & (x.side == 'B')]['size'].sum() - x[(x['size'] >= P95) & (x.side == 'A')]['size'].sum())
# CVD
agg['sd'] = np.where(agg.side == 'B', agg['size'], -agg['size']); agg['cvd'] = agg.sd.cumsum()
i_pk, i_tr = agg.cvd.idxmax(), agg.cvd.idxmin()
print('CVD 峰', agg.cvd[i_pk], agg.et[i_pk].strftime('%H:%M:%S'), '谷', agg.cvd[i_tr], agg.et[i_tr].strftime('%H:%M:%S'), '收', agg.cvd.iloc[-1])
for hh in ['10:00', '11:00', '11:40', '11:41', '12:00', '12:30', '13:00', '14:00', '15:00', '15:30', '15:59']:
    tt = pd.Timestamp(f'2026-10-08 {hh}', tz=ET); s = agg[agg.et <= tt]
    print(f'  CVD@{hh}={s.cvd.iloc[-1] if len(s) else 0}', end='')
print()
l2 = agg[agg.et >= pd.Timestamp('2026-10-08 14:00', tz=ET)]; print('尾盘2h Delta', int(l2.sd.sum()), '| 尾盘30min Delta', int(agg[agg.et >= pd.Timestamp('2026-10-08 15:30', tz=ET)].sd.sum()))

# ---------------- DOM 时间序列 / 委比 ----------------
W = [1.0, .6, .35, .2, .1]
dom = dp[(dp.et >= t0 - pd.Timedelta('5min')) & (dp.et < t1 + pd.Timedelta('1s'))].copy()
dom['wB'] = sum(W[k] * dom[f'bid_s{k+1}'] for k in range(5)); dom['wA'] = sum(W[k] * dom[f'ask_s{k+1}'] for k in range(5))
dom['wr'] = (dom.wB - dom.wA) / (dom.wB + dom.wA) * 100
D_ns = dom.et.astype('int64').values
def dom_at(t):
    i = np.searchsorted(D_ns, pd.Timestamp(t).value, side='right') - 1
    return dom.iloc[i]
rows = []
for k in range(13):
    tt = t0 + pd.Timedelta(minutes=30 * k) + pd.Timedelta('1s'); r = dom_at(tt)
    rows.append((tt.strftime('%H:%M:%S'), '30min bar', round(r.wB), round(r.wA), round(r.wr, 1), '—'))
bigs = agg[agg['size'] >= P99a]
for _, b in bigs.iterrows():
    r = dom_at(b.et - pd.Timedelta('1s'))
    same = (r.wr > 0 and b.side == 'B') or (r.wr < 0 and b.side == 'A')
    rows.append((b.et.strftime('%H:%M:%S'), f'P99大单{"买" if b.side=="B" else "卖"}{b["size"]}@{b.pmin}', round(r.wB), round(r.wA), round(r.wr, 1), '顺' if same else '逆'))
LV = {'ORH': 8.30, 'ORL': 8.13, 'POC': 8.00, 'VAH': 8.13, 'VAL': 7.93}
after = allt[allt.et >= t0 + pd.Timedelta('30min')]
for nm, lv in LV.items():
    hit = after[(after.price >= lv) if nm in ('ORH',) else (after.price <= lv)] if nm in ('ORH', 'ORL', 'POC', 'VAL') else after[after.price >= lv]
    if nm == 'VAH':  # VAH 自下而上触及：先跌破后再回到>=8.13
        below = after[after.price < lv]; hit = after[(after.price >= lv) & (after.et > below.et.iloc[0])] if len(below) else hit
    if len(hit):
        th = hit.et.iloc[0]; r = dom_at(th - pd.Timedelta('1s'))
        rows.append((th.strftime('%H:%M:%S'), f'触及{nm}={lv}前1s', round(r.wB), round(r.wA), round(r.wr, 1), ''))
print('== 3.1 委比快照'); print(pd.DataFrame(rows, columns=['时间ET', '触发', '加权Bid', '加权Ask', '委比%', '顺/逆']).to_string(index=False))
dr = dom[(dom.et >= t0) & (dom.et < t1)]
# 时间加权全天委比
dt = np.diff(np.append(dr.et.astype('int64').values, t1.value)) / 1e9
print('全天时间加权委比=', round(np.average(dr.wr.fillna(0), weights=dt), 1), '%', '| 各30min时间加权:', end=' ')
dr = dr.assign(dt=dt, bar=dr.et.dt.floor('30min'))
print({k.strftime('%H:%M'): round(np.average(g.wr.fillna(0), weights=g.dt), 1) for k, g in dr.groupby('bar')})

# ---------------- 3.2 ISO 扫单 ----------------
L = lit[lit.side.isin(['A', 'B'])].sort_values('et').reset_index(drop=True)
iso = []
for sd in ['A', 'B']:
    s = L[L.side == sd].reset_index(drop=True); ns = s.et.astype('int64').values; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ns[j + 1] - ns[i] <= 5_000_000: j += 1
        c = s.iloc[i:j + 1]
        if c.price.nunique() >= 2 and c['size'].sum() >= P99a:
            iso.append((c.et.iloc[0].strftime('%H:%M:%S.%f')[:-3], '卖' if sd == 'A' else '买', c.price.nunique(), c.publisher_id.nunique(), int(c['size'].sum()), f'{c.price.max()}~{c.price.min()}'))
        i = j + 1
iso = pd.DataFrame(iso, columns=['时间', '方向', '档位数', 'venue数', '总量', '价格区间']).sort_values('时间')
print('== 3.2 ISO扫单 (阈值P99=', P99a, ')'); print(iso.to_string(index=False))
print('ISO买', (iso.方向 == '买').sum(), iso[iso.方向 == '买'].总量.sum(), '| ISO卖', (iso.方向 == '卖').sum(), iso[iso.方向 == '卖'].总量.sum())

# ---------------- 订单生命周期 ----------------
dp['key'] = list(zip(dp.publisher_id, dp.channel_id, dp.order_id))
ords = dp[dp.action.isin(['A', 'M', 'C', 'F']) & (dp.order_id > 0)]
A = ords[ords.action == 'A'].drop_duplicates('key').set_index('key')
# 档位：A行价格与同行 bid_p1~p5 / ask_p1~p5 比较（事件后DOM）
lvl = np.full(len(A), 0)
for k in range(5, 0, -1):
    mb = (A.side == 'B') & (A[f'bid_p{k}'] == A.price); ma = (A.side == 'A') & (A[f'ask_p{k}'] == A.price)
    lvl[(mb | ma).values] = k
A['lvl'] = lvl
A['dist_tk'] = np.where(A.side == 'B', (A.bid_p1 - A.price) / TICK, (A.price - A.ask_p1) / TICK).round(0)
F = ords[ords.action == 'F'].groupby('key').agg(fill=('size', 'sum'), f_first=('et', 'min'), f_last=('et', 'max'))
C = ords[ords.action == 'C'].groupby('key').agg(cxl=('size', 'sum'), c_last=('et', 'max'), c_px=('price', 'last'))
Mx = ords[ords.action == 'M'].groupby('key').agg(m_n=('size', 'size'), m_last_px=('price', 'last'), m_max=('size', 'max'))
O = A[['et', 'publisher_id', 'side', 'price', 'size', 'lvl', 'dist_tk', 'bid_p1', 'ask_p1']].join(F).join(C).join(Mx)
O[['fill', 'cxl', 'm_n']] = O[['fill', 'cxl', 'm_n']].fillna(0)
O['tot'] = np.maximum(O['size'], O.m_max.fillna(0)); O['fr'] = O.fill / (O.fill + O.cxl).replace(0, np.nan)
O['life'] = (O.c_last - O.et).dt.total_seconds()
O['px_end'] = O.m_last_px.fillna(O.price)
rthA = O[(O.et >= t0) & (O.et < t1) & O.side.isin(['A', 'B'])]
pa = rthA['size'].quantile([.9, .98, .99]); print('== 新增挂单size分位 P90/P98/P99 =', pa.tolist(), '| RTH新增笔数', len(rthA))
P90o, P98o, P99o = pa.loc[.9], pa.loc[.98], pa.loc[.99]

# ---------------- 3.3 挂单墙候选 ----------------
wall = rthA[(rthA['size'] >= P99o)].copy()
wall['life2'] = wall.life.fillna((t1 - wall.et).dt.total_seconds())
wall = wall[wall.life2 > 15]
w = wall.assign(t=wall.et.dt.strftime('%H:%M:%S'), 档位=wall.lvl.map(lambda v: f'L{v}' if v else '>L5'), 成交=wall.fill.astype(int), 存活=wall.life2.round(0),
                状态=np.where(wall.cxl > 0, '撤', '存续/成交'))[['t', 'price', 'side', 'size', '存活', '档位', 'dist_tk', '成交', '状态', 'publisher_id']]
print('== 3.3 挂单墙候选 (size>=P99=', P99o, ', life>15s) 共', len(w)); print(w.sort_values('size', ascending=False).head(25).to_string(index=False))
for sd in 'BA':
    x = wall[wall.side == sd]; print(f'{sd}墙: {len(x)}个/{int(x["size"].sum())}股 | 其中L1-L5: {len(x[x.lvl>0])}个/{int(x[x.lvl>0]["size"].sum())}股')

# ---------------- 3.5 谎骗 150s 穿越 ----------------
cand = rthA[(rthA.lvl > 0) & (rthA.cxl > 0) & (rthA.fr < 0.15)].copy()
res = []
for key, o in cand.iterrows():
    c = o.c_last.value; e = c + 150 * 1_000_000_000
    i, j = np.searchsorted(T_ns, c, 'right'), np.searchsorted(T_ns, e, 'right')
    px = T_px[i:j]; p = o.px_end
    crossed = (px < p - 1e-9).any() if o.side == 'B' else (px > p + 1e-9).any()
    if crossed: res.append('确认')
    elif e > t1.value or len(px) == 0: res.append('不可判定')
    else: res.append('托单嫌疑')
cand['res'] = res
cand['fast'] = cand.life <= 1
print('== 3.5 谎骗确认（L1-L5, fill<15%, 撤单, 150s穿越）')
for nm, sub in [('全部', cand), (f'size>=P90({P90o})', cand[cand['size'] >= P90o])]:
    for sd in 'BA':
        x = sub[sub.side == sd]; n = len(x); vc = x.res.value_counts()
        print(f' [{nm}] {sd}: 总{n} 确认{vc.get("确认",0)}({vc.get("确认",0)/max(n,1)*100:.1f}%) 托单嫌疑{vc.get("托单嫌疑",0)} 不可判定{vc.get("不可判定",0)} | 量加权确认率 {x[x.res=="确认"]["size"].sum()/max(x["size"].sum(),1)*100:.1f}%')
# 极速撤挂比
for sd in 'BA':
    x = rthA[rthA.side == sd]; fc = x[(x.cxl > 0) & (x.fill == 0) & (x.life <= 1)]
    print(f'极速撤挂(≤1s,零成交) {sd}: {len(fc)}/{len(x)} = {len(fc)/len(x)*100:.1f}%')
deep = rthA[(rthA['size'] >= P98o) & (rthA.dist_tk > 10) & (rthA.life > 120) & (rthA.fill == 0) & (rthA.cxl > 0)]
for sd in 'BA':
    x = deep[deep.side == sd]; print(f'深预埋撤单 {sd}: {len(x)}笔/{int(x["size"].sum())}股', x[['price', 'size', 'dist_tk', 'life']].assign(t=x.et.dt.strftime('%H:%M')).head(8).values.tolist())

# ---------------- 3.6 TRF ----------------
trf = trf.sort_values('et'); trf_ns = trf.et.astype('int64').values
idx = np.searchsorted(D_ns, trf_ns, 'right') - 1
bp, ap = dom.bid_p1.values[idx], dom.ask_p1.values[idx]
trf['cls'] = np.where(trf.price >= ap - 1e-9, 'near_ask', np.where(trf.price <= bp + 1e-9, 'near_bid', 'mid'))
tv = trf['size'].sum(); g = trf.groupby('cls')['size'].sum()
print('== 3.6 TRF 总', tv, {k: f'{v}({v/tv*100:.1f}%)' for k, v in g.items()}, 'TRF净=', g.get('near_ask', 0) - g.get('near_bid', 0))
p995 = trf['size'].quantile(.995); t5 = trf[trf['size'] >= p995].nlargest(5, 'size')
print('TRF P99.5=', p995); print(t5.assign(t=t5.et.dt.strftime('%H:%M:%S'), bid=bp[np.searchsorted(trf_ns, t5.et.astype("int64").values)], ask=ap[np.searchsorted(trf_ns, t5.et.astype("int64").values)])[['t', 'price', 'size', 'cls', 'bid', 'ask']].to_string(index=False))
tb = trf.assign(bin=(trf.price + 1e-9).round(2)).groupby('bin')['size'].sum().nlargest(3)
print('TRF Top3价位:', {k: (v, round((k - DL) / (DH - DL) * 100, 1)) for k, v in tb.items()}, '| TRF VWAP', round((trf.price * trf['size']).sum() / tv, 4), '区间%', round(((trf.price * trf['size']).sum() / tv - DL) / (DH - DL) * 100, 1))
ex = trf[trf['size'] < 5000]; ge = ex.groupby('cls')['size'].sum(); print('剔除>=5000大单后TRF净', ge.get('near_ask', 0) - ge.get('near_bid', 0), ge.to_dict())
trf['bar'] = trf.et.dt.floor('30min'); print('TRF分段净:', {k.strftime('%H:%M'): int(v.loc[v.cls=='near_ask','size'].sum()-v.loc[v.cls=='near_bid','size'].sum()) for k, v in trf.groupby('bar')})

# ---------------- 3.7 预谋强度 ----------------
allA = O[O.side.isin(['A', 'B'])]
P98all = allA[(allA.et >= t0) & (allA.et < t1)]['size'].quantile(.98)
pm = allA[(allA['size'] >= P98all) & (allA.fill > 0) & (allA.f_first >= t0) & (allA.f_first < t1 + pd.Timedelta('1s'))].copy()
pm['tt'] = (pm.f_first - pm.et).dt.total_seconds()
pm['cls'] = np.where((pm.dist_tk > 10) & (pm.tt > 120), '强预谋', np.where(pm.dist_tk <= 2, '实时反应', '快速到达'))
print('== 3.7 预谋 P98=', P98all, '样本', len(pm))
for sd in 'BA':
    x = pm[pm.side == sd]
    print(f'{"Bid" if sd=="B" else "Ask"}被动:', {c: f'{len(x[x.cls==c])}笔/{int(x[x.cls==c].fill.sum())}股成交' for c in ['强预谋', '快速到达', '实时反应']})
s = pm[pm.cls == '强预谋']; print(s.assign(t=s.et.dt.strftime('%H:%M:%S'), ft=s.f_first.dt.strftime('%H:%M:%S'))[['t', 'ft', 'publisher_id', 'side', 'price', 'size', 'fill', 'dist_tk', 'tt']].to_string(index=False))
print('强预谋 publisher分布:', s.groupby(['side', 'publisher_id']).fill.sum().to_dict())

# ---------------- §4/§6 辅助：分带吸收率 ----------------
agg['zone'] = np.where(agg.pmin <= Z1, 'LOW', np.where(agg.pmin >= Z2, 'HIGH', 'MID'))
print('== 分带（LOW<=%.3f HIGH>=%.3f）' % (Z1, Z2))
for z, x in agg.groupby('zone'):
    ta, tb_ = x[x.side == 'A']['size'].sum(), x[x.side == 'B']['size'].sum()
    print(f'{z}: T(A)=F(B)被动买承接={ta} T(B)=F(A)被动卖供给={tb_} | Bid吸率={ta/(ta+tb_)*100:.1f}% Ask供给率={tb_/(ta+tb_)*100:.1f}%')
# 价差/流动性洞：1秒内价格位移>=8tick 且 价差>=6tick
dd = dom[(dom.et >= t0) & (dom.et < t1)][['et', 'bid_p1', 'ask_p1']].copy(); dd['spr'] = ((dd.ask_p1 - dd.bid_p1) / TICK).round(0)
print('价差分位 P50/P90/P99 tick:', dd.spr.quantile([.5, .9, .99]).tolist(), '| crossed行占比', round((dd.spr <= 0).mean() * 100, 2), '%')
at = allt.set_index('et').price; r1 = at.rolling('1s').agg(lambda v: v.max() - v.min())
hole = r1[r1 >= 0.08]
if len(hole):
    hs = hole.index.to_series().diff().dt.total_seconds().fillna(999).gt(60).cumsum()
    for _, gg in hole.groupby(hs.values):
        st = gg.index[0]; w_ = allt[(allt.et >= st - pd.Timedelta('2s')) & (allt.et <= st + pd.Timedelta('2s'))]
        sp = dd[(dd.et >= st - pd.Timedelta('2s')) & (dd.et <= st + pd.Timedelta('2s'))].spr
        print(f'  流动性洞候选 {st.strftime("%H:%M:%S")}: 1s价幅={gg.max():.2f} 高{w_.price.max()} 低{w_.price.min()} 期间最大价差={sp.max()}tick')
