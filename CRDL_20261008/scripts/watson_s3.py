"""
华生 §3.1 委比 / §3.2 ISO / §3.5 谎骗150s穿越 / §3.6 TRF / §3.7 预谋强度 / Liquidity Hole
盘口全部使用 rebuild_book.py 重建的4场所合并盘口。
"""
import sys
import json
import numpy as np
import pandas as pd

S = sys.argv[1]
P = sys.argv[2]
TICK = 0.01
ET = 'America/New_York'
ts = lambda s: pd.Timestamp(f'2026-10-08 {s}', tz=ET)
dom = pd.read_parquet(f'{S}/dom.parquet')
prints = pd.read_parquet(f'{S}/prints.parquet')
agg = pd.read_parquet(f'{S}/aggr.parquet')
core = json.load(open(f'{S}/watson_core.json'))
tr = pd.read_parquet(f'{P}/06_CRDL_ALL_trades_20261008.parquet')
tr['et'] = tr.ts_event.dt.tz_convert(ET)
D0, D1 = ts('09:30:00'), ts('16:00:01')
out = {}
W = np.array([1.0, 0.6, 0.35, 0.2, 0.1])
bs = dom[[f'bid_s{i}' for i in range(1, 6)]].values.astype(float)
as_ = dom[[f'ask_s{i}' for i in range(1, 6)]].values.astype(float)
dom['wb'] = (bs * W).sum(1)
dom['wa'] = (as_ * W).sum(1)
dom['wr'] = (dom.wb - dom.wa) / (dom.wb + dom.wa) * 100


def snap(t):
    """t 时刻之前最后一个事件后的盘口（即 t 时刻的盘口状态）"""
    x = dom[dom.et < t]
    r = x.iloc[-1]
    return dict(t=str(t.time())[:12], bid1=r.bid_p1, ask1=r.ask_p1, wb=int(r.wb), wa=int(r.wa), wr=round(r.wr, 1))


# ---------------- §3.1 委比 ----------------
snaps = []
for s in ['09:30:05', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00', '12:30:00', '13:00:00',
          '13:30:00', '14:00:00', '14:30:00', '15:00:00', '15:30:00', '15:59:50']:
    d = snap(ts(s))
    d['trig'] = '30min bar'
    snaps.append(d)
# P99 大单前1s（聚合主动单 ≥ P99）
p99 = core['aggr_pct']['P99']
bigs = agg[(agg['size'] >= p99) & (agg.et >= D0) & (agg.et < D1)].sort_values('et')
for r in bigs.itertuples():
    d = snap(r.et - pd.Timedelta('1s'))
    d['trig'] = f'P99 {"买" if r.side == "B" else "卖"}主动 {int(r.size)}股 @{r.pmin:.3f}-{r.pmax:.3f} {str(r.et.time())[:8]}'
    # 顺势=委比方向与大单方向一致
    d['flow'] = '顺' if (d['wr'] > 0) == (r.side == 'B') else '逆'
    snaps.append(d)
# 关键价位首次触及前（ORH/VAH/POC/VAL 用场内成交）
lv = core['levels']
exp = prints[(prints.src != 'TRF') & (prints.et > ts('09:35:00'))]
for name in ['ORH', 'VAH', 'POC', 'VAL']:
    px = lv[name]
    hit = exp[(exp.price - px).abs() < 1e-9]
    if len(hit):
        t = hit.et.iloc[0]
        d = snap(t - pd.Timedelta('1s'))
        d['trig'] = f'触及{name} {px:.3f} 前1s ({str(t.time())[:8]})'
        snaps.append(d)
out['weibi'] = snaps
day_wr = dom[(dom.et >= D0) & (dom.et < ts('16:00:00'))]
out['weibi_rth_mean'] = round(day_wr.wr.mean(), 1)
out['weibi_rth_median'] = round(day_wr.wr.median(), 1)

# ---------------- §3.2 ISO（5ms内同向、≥2个价位、总量≥P99） ----------------
Tr = dom[(dom.action == 'T') & dom.side.isin(['B', 'A']) & (dom.et >= D0) & (dom.et < D1)].sort_values('ts_event').copy()
iso = []
for side in ['B', 'A']:
    x = Tr[Tr.side == side].copy()
    gap = x.ts_event.diff().dt.total_seconds().fillna(1e9)
    x['g'] = (gap > 0.005).cumsum()
    g = x.groupby('g').agg(t=('et', 'first'), size=('size', 'sum'), npx=('price', 'nunique'),
                           pmin=('price', 'min'), pmax=('price', 'max'), nven=('publisher_id', 'nunique'))
    g = g[(g.npx >= 2) & (g['size'] >= p99)]
    for r in g.itertuples():
        iso.append(dict(t=str(r.t.time())[:12], side=side, size=int(r.size), npx=int(r.npx), nven=int(r.nven),
                        px=f'{r.pmin:.3f}-{r.pmax:.3f}'))
# 放宽版：总量≥P95
iso_loose = []
p95 = core['aggr_pct']['P95']
for side in ['B', 'A']:
    x = Tr[Tr.side == side].copy()
    gap = x.ts_event.diff().dt.total_seconds().fillna(1e9)
    x['g'] = (gap > 0.005).cumsum()
    g = x.groupby('g').agg(t=('et', 'first'), size=('size', 'sum'), npx=('price', 'nunique'),
                           pmin=('price', 'min'), pmax=('price', 'max'))
    g = g[(g.npx >= 2) & (g['size'] >= p95)]
    iso_loose += [dict(t=str(r.t.time())[:8], side=side, size=int(r.size), px=f'{r.pmin:.3f}-{r.pmax:.3f}') for r in g.itertuples()]
out['iso_p99'] = sorted(iso, key=lambda z: z['t'])
out['iso_p95'] = sorted(iso_loose, key=lambda z: z['t'])

# ---------------- 订单生命周期 ----------------
ev = dom[dom.action.isin(['A', 'C', 'F', 'M']) & dom.side.isin(['B', 'A'])].copy()
ev['key'] = list(zip(ev.publisher_id, ev.channel_id, ev.order_id))
adds = ev[ev.action == 'A'].copy()
adds = adds[(adds.et >= D0) & (adds.et < ts('16:00:00'))]
adds = adds.drop_duplicates('key')
fills = ev[ev.action == 'F'].groupby('key').agg(fill=('size', 'sum'), f_first=('et', 'min'))
canc = ev[ev.action == 'C'].groupby('key').agg(c_last=('et', 'max'), c_px=('price', 'last'), c_n=('size', 'count'))
mods = ev[ev.action == 'M'].groupby('key').agg(m_n=('size', 'count'))
o = adds.set_index('key').join(fills).join(canc).join(mods)
o['fill'] = o['fill'].fillna(0)
o['fill_ratio'] = o['fill'] / o['size']


def level(r):
    """A事件price直接比同行(事件后)bid_p1~p5 / ask_p1~p5"""
    pre = 'bid' if r.side == 'B' else 'ask'
    for i in range(1, 6):
        v = getattr(r, f'{pre}_p{i}')
        if v == v and abs(v - r.price) < 1e-9:
            return i
    return 0


o['lvl'] = [level(r) for r in o.itertuples()]
# 与同侧最优价距离（tick，事件前）
o['dist'] = np.where(o.side == 'B', (o.pre_bid1 - o.price) / TICK, (o.price - o.pre_ask1) / TICK)
o['dist'] = o['dist'].clip(lower=0).round()
o['life'] = (o.c_last - o.et).dt.total_seconds()
o['travel'] = (o.f_first - o.et).dt.total_seconds()
o.reset_index(drop=True).drop(columns=['et']).assign(et=o.et.values).to_parquet(f'{S}/orders.parquet')

# 极速撤挂比（≤1s撤单 / 全部撤单）
cx = o[o.c_last.notna()]
out['fast_cancel_le1s'] = {s: round((cx[cx.side == s].life <= 1).mean() * 100, 1) for s in ['B', 'A']}
out['cancel_count'] = {s: int((cx.side == s).sum()) for s in ['B', 'A']}

# ---------------- §3.5 谎骗 150s 穿越 ----------------
ap = prints[['et', 'price']].sort_values('et')
ap_t = ap.et.values
ap_p = ap.price.values
afterhrs_cut = ts('16:00:00')


def spoof_eval(sub):
    res = {'confirmed': 0, 'prop': 0, 'undet': 0}
    for r in sub.itertuples():
        t0 = r.c_last
        t1 = t0 + pd.Timedelta('150s')
        if t1 > afterhrs_cut:
            res['undet'] += 1
            continue
        i0 = np.searchsorted(ap_t, t0.to_datetime64(), 'right')
        i1 = np.searchsorted(ap_t, t1.to_datetime64(), 'right')
        w = ap_p[i0:i1]
        px = r.c_px
        if r.side == 'B':
            hit = (w < px - 1e-9).any()
        else:
            hit = (w > px + 1e-9).any()
        res['confirmed' if hit else 'prop'] += 1
    n = sum(res.values())
    res['n'] = n
    res['rate'] = round(res['confirmed'] / n * 100, 1) if n else None
    return res


q = o[(o.lvl >= 1) & o.c_last.notna() & (o.fill_ratio < 0.15)]
# 仅统计最终全部撤单的（剩余量被撤）
add_p95 = o['size'].quantile(0.95)
out['add_size_pct'] = {k: int(v) for k, v in zip(['P90', 'P95', 'P98', 'P99'], o['size'].quantile([.9, .95, .98, .99]).values)}
out['spoof_all'] = {s: spoof_eval(q[q.side == s]) for s in ['B', 'A']}
out['spoof_big'] = {s: spoof_eval(q[(q.side == s) & (q['size'] >= add_p95)]) for s in ['B', 'A']}
# 深预埋撤单：size≥P98 且 dist>10 且 life>120s 且纯撤(无成交)
p98a = o['size'].quantile(0.98)
deep = o[(o['size'] >= p98a) & (o.dist > 10) & (o.life > 120) & (o['fill'] == 0)]
out['deep_cancel'] = {s: dict(n=int((deep.side == s).sum()), vol=int(deep[deep.side == s]['size'].sum())) for s in ['B', 'A']}
out['deep_cancel_list'] = [dict(t=str(r.et.time())[:8], side=r.side, px=r.price, size=int(r.size), dist=int(r.dist),
                                life=round(r.life, 0), pub=int(r.publisher_id)) for r in deep.sort_values('size', ascending=False).head(8).itertuples()]

# ---------------- §3.7 预谋强度（size≥P98 的被动单） ----------------
big = o[(o['size'] >= p98a) & (o['fill'] > 0)].copy()


def cat(r):
    if r.dist > 10 and r.travel > 120:
        return 'strong'
    if r.dist <= 2:
        return 'realtime'
    return 'fast'


big['cat'] = [cat(r) for r in big.itertuples()]
pm = {}
for s in ['B', 'A']:
    for c in ['strong', 'fast', 'realtime']:
        x = big[(big.side == s) & (big.cat == c)]
        pm[f'{s}_{c}'] = dict(n=len(x), fill=int(x['fill'].sum()))
out['premed_P98'] = int(p98a)
out['premed'] = pm
sb, sa = pm['B_strong']['fill'], pm['A_strong']['fill']
out['premed_ratio_strong'] = (round(sb / sa, 2) if sa else ('inf' if sb else 'n/a'))
# 全部 P98 被动成交量比（备用）
tb_all = big[big.side == 'B']['fill'].sum()
ta_all = big[big.side == 'A']['fill'].sum()
out['premed_ratio_allP98_fill'] = round(tb_all / ta_all, 2) if ta_all else None
out['premed_strong_list'] = [dict(t=str(r.et.time())[:8], side=r.side, px=r.price, size=int(r.size), fill=int(r.fill),
                                  dist=int(r.dist), travel=round(r.travel, 0), pub=int(r.publisher_id))
                             for r in big[big.cat == 'strong'].itertuples()]
out['big_passive_fill_list'] = [dict(t=str(r.et.time())[:8], side=r.side, px=r.price, size=int(r.size), fill=int(r.fill),
                                     dist=int(r.dist), travel=round(r.travel, 0), lvl=int(r.lvl), pub=int(r.publisher_id))
                                for r in big.sort_values('fill', ascending=False).head(12).itertuples()]

# ---------------- §3.6 TRF ----------------
trf = tr[tr.publisher_id.isin([82, 83]) & (tr['size'] > 0) & (tr.et >= D0) & (tr.et < D1)].sort_values('ts_event').copy()
bk = dom[['ts_event', 'bid_p1', 'ask_p1']].sort_values('ts_event')
trf = pd.merge_asof(trf, bk, on='ts_event', direction='backward', allow_exact_matches=False)
trf['cls'] = np.where(trf.price >= trf.ask_p1 - 1e-9, 'near_ask', np.where(trf.price <= trf.bid_p1 + 1e-9, 'near_bid', 'mid'))
tv = trf.groupby('cls')['size'].sum()
tot = trf['size'].sum()
out['trf_cls'] = {k: dict(vol=int(v), pct=round(v / tot * 100, 1)) for k, v in tv.items()}
out['trf_net'] = int(tv.get('near_ask', 0) - tv.get('near_bid', 0))
p995 = trf['size'].quantile(0.995)
hi, lo = core['ohlc']['high'], core['ohlc']['low']
top5 = trf[trf['size'] >= p995].sort_values('size', ascending=False).head(5)
out['trf_p995'] = int(p995)
out['trf_top5'] = [dict(t=str(r.et.time())[:8], px=r.price, size=int(r.size), cls=r.cls, bbo=f'{r.bid_p1:.2f}/{r.ask_p1:.2f}') for r in top5.itertuples()]
trf['b'] = (trf.price / TICK).round().astype(int) * TICK
lv3 = trf.groupby('b')['size'].sum().sort_values(ascending=False).head(3)
out['trf_top3_levels'] = [dict(px=round(p, 2), vol=int(v), range_pct=round((p - lo) / (hi - lo) * 100, 0)) for p, v in lv3.items()]
out['trf_top3_wavg_rangepct'] = round(sum((p - lo) / (hi - lo) * 100 * v for p, v in lv3.items()) / lv3.sum(), 0)
# TRF 30min 分布
trf['bar'] = trf.et.dt.floor('30min').dt.strftime('%H:%M')
out['trf_by_bar'] = trf.groupby(['bar', 'cls'])['size'].sum().unstack(fill_value=0).astype(int).to_dict('index')

# ---------------- Liquidity Hole ----------------
# 定义：5s 内同向主动成交使成交价位移≥3tick，且位移后1s内 合并盘口价差≥3tick（常态中位1tick）
holes = []
Tr2 = Tr.sort_values('ts_event')
for side in ['B', 'A']:
    x = Tr2[Tr2.side == side]
    for i, r in enumerate(x.itertuples()):
        w = x[(x.ts_event > r.ts_event) & (x.ts_event <= r.ts_event + pd.Timedelta('5s'))]
        if not len(w):
            continue
        move = (w.price.max() - r.price) if side == 'B' else (r.price - w.price.min())
        if move >= 3 * TICK - 1e-9:
            t_end = w.et.iloc[-1]
            after = dom[(dom.et >= t_end) & (dom.et <= t_end + pd.Timedelta('1s'))]
            spr = (after.ask_p1 - after.bid_p1).max()
            holes.append(dict(t=str(r.et.time())[:8], side=side, move_ticks=round(move / TICK, 1), max_spread_ticks=round(spr / TICK, 1)))
hd = pd.DataFrame(holes)
if len(hd):
    hd = hd[hd.max_spread_ticks >= 3].drop_duplicates(['side', 't'])
    hd['min'] = hd.t.str[:5]
    hd = hd.drop_duplicates(['side', 'min'])
out['liq_hole'] = hd.to_dict('records') if len(hd) else []

json.dump(out, open(f'{S}/watson_s3.json', 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items():
    if k == 'weibi':
        print(pd.DataFrame(v).to_string())
    else:
        print(k, v)
