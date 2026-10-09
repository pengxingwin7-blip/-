"""
华生 v4.1 自算部分（§2 基础快照 + §3 七项自算）· 用法: python3 -I watson_selfcalc.py <证据包目录> <输出json>（当前文件名硬编码为 SFIX 2026-10-08）
数据口径：
  depth5.parquet = 4 个场内交易所 MBO（XNAS/ARCX/BZX/XNYS），每行附事件后的合并 DOM 五档
  trades.parquet = XNAS.BASIC：81=Nasdaq 交易所（与 MBO pub2 重复，剔除）
                   82/83=FINRA TRF（场外），88/89=BX/PSX（场内但无方向）
  T 事件 side：B=主动买 A=主动卖 N=隐藏单/集合竞价（无方向，不计 Delta）
"""
import sys, json
import numpy as np
import pandas as pd

D = sys.argv[1]
TZ = "America/New_York"
TICK = 0.01
W = np.array([1.0, 0.6, 0.35, 0.2, 0.1])          # 委比权重 L1~L5
OUT = {}

# ---------------------------------------------------------------- 读数据
d = pd.read_parquet(f"{D}/07_SFIX_depth5_2026-10-08.parquet")
d["et"] = d.ts_event.dt.tz_convert(TZ)
d = d.reset_index(drop=True)
d["size"] = d["size"].astype("int64")
for _c in [c for c in d.columns if c.startswith(("bid_s", "ask_s"))]: d[_c] = d[_c].astype("int64")
tr = pd.read_parquet(f"{D}/06_SFIX_ALL_trades_20261008.parquet")
tr["et"] = tr.ts_event.dt.tz_convert(TZ)
tr["size"] = tr["size"].astype("int64")

day = pd.Timestamp("2026-10-08", tz=TZ)
OPEN, CLOSE = day + pd.Timedelta("09:30:00"), day + pd.Timedelta("16:00:00")

# 连续竞价时段 = 09:30:00 开盘集合竞价之后 ~ 16:00 之前
cont = (d.et >= OPEN) & (d.et < CLOSE)
T = d[(d.action == "T") & cont].copy()
cross_mask = (d.action == "T") & (d.side == "N") & (d["size"] > 10000) & ((d.et < OPEN + pd.Timedelta("1s")) | (d.et >= CLOSE))
crosses = d[cross_mask]
T = T[~T.index.isin(crosses.index)]

# ---------------------------------------------------------------- §2 量能
lit_all = int(d.loc[d.action == "T", "size"].sum())
close_cross = int(crosses[crosses.et >= CLOSE]["size"].sum())
open_cross = int(crosses[crosses.et < OPEN + pd.Timedelta("1s")]["size"].sum())
trf = tr[tr.publisher_id.isin([82, 83])].copy()
bxpsx = tr[tr.publisher_id.isin([88, 89])]
trf_all = int(trf["size"].sum())
trf_c = trf[(trf.et >= OPEN) & (trf.et < CLOSE)].copy()
TOTAL = 5152008                                            # 来源：day.csv / FMP quote
day_csv = pd.read_csv(f"{D}/03_SFIX_day.csv", parse_dates=["date"])
prev20 = day_csv[day_csv.date < "2026-10-08"].tail(20)
avg20 = prev20.volume.mean()
OUT["volume"] = dict(lit_mbo_all=lit_all, close_cross=close_cross, open_cross=open_cross,
                     lit_continuous=int(T["size"].sum()), trf_all=trf_all,
                     trf_continuous=int(trf_c["size"].sum()), bx_psx=int(bxpsx["size"].sum()),
                     total=TOTAL, unobserved=TOTAL - lit_all - trf_all - int(bxpsx["size"].sum()),
                     avg20=round(avg20), rvol=round(TOTAL / avg20, 2),
                     coverage_incl_cross=round(lit_all / TOTAL * 100, 1),
                     coverage_continuous=round(T["size"].sum() / (TOTAL - close_cross - open_cross) * 100, 1),
                     close_cross_pct=round(close_cross / TOTAL * 100, 1))

# 价格：高低点及时间（场内连续 + TRF）
allpx = pd.concat([T[["et", "price", "size"]].assign(src="LIT"),
                   trf_c[["et", "price", "size"]].assign(src="TRF")]).sort_values("et")
hi = allpx.loc[allpx.price.idxmax()]; lo = allpx.loc[allpx.price.idxmin()]
lhi = T.loc[T.price.idxmax()]; llo = T.loc[T.price.idxmin()]
OUT["price"] = dict(high=float(hi.price), high_t=str(hi.et.time())[:8], high_src=hi.src,
                    low=float(lo.price), low_t=str(lo.et.time())[:8], low_src=lo.src, low_size=int(lo["size"]),
                    lit_high=float(lhi.price), lit_high_t=str(lhi.et.time())[:8],
                    lit_low=float(llo.price), lit_low_t=str(llo.et.time())[:8],
                    close_cross_px=float(crosses[crosses.et >= CLOSE].price.iloc[0]) if close_cross else None,
                    last_cont_px=float(T.price.iloc[-1]))
vw = (allpx.price * allpx["size"]).sum() / allpx["size"].sum()
OUT["price"]["vwap_cont"] = round(vw, 4)

# ---------------------------------------------------------------- Delta / 恒等式
TB = int(T.loc[T.side == "B", "size"].sum()); TA = int(T.loc[T.side == "A", "size"].sum())
TN = int(T.loc[T.side == "N", "size"].sum())
Fc = d[(d.action == "F") & cont]
FA = int(Fc.loc[Fc.side == "A", "size"].sum()); FB = int(Fc.loc[Fc.side == "B", "size"].sum())
OUT["delta"] = dict(TB=TB, TA=TA, TN_hidden=TN, net=TB - TA, FA=FA, FB=FB,
                    identity_TB_eq_FA=TB == FA, identity_gap=TB - FA, identity_TA_FB_gap=TA - FB)

# 尾盘 30 min
last30 = T[T.et >= CLOSE - pd.Timedelta("30min")]
l30_trf = trf_c[trf_c.et >= CLOSE - pd.Timedelta("30min")]
OUT["last30"] = dict(lit=int(last30["size"].sum()), trf=int(l30_trf["size"].sum()),
                     vol_incl_cross=int(last30["size"].sum() + l30_trf["size"].sum() + close_cross),
                     delta=int(last30.loc[last30.side == "B", "size"].sum() - last30.loc[last30.side == "A", "size"].sum()))
OUT["last30"]["pct_of_day"] = round(OUT["last30"]["vol_incl_cross"] / TOTAL * 100, 1)

# ---------------------------------------------------------------- 关键价位：ORH/ORL/POC/VAH/VAL
orw = allpx[allpx.et < OPEN + pd.Timedelta("30min")]
ORH, ORL = float(orw.price.max()), float(orw.price.min())
prof = allpx.assign(px=(allpx.price / TICK).round().astype(int) * TICK).groupby("px")["size"].sum().sort_index()
poc = float(prof.idxmax())
# 价值区：从 POC 向两侧扩展直到 70%
tot = prof.sum(); idx = list(prof.index); i = j = idx.index(prof.idxmax()); acc = prof.iloc[i]
while acc < 0.7 * tot:
    up = prof.iloc[j + 1] if j + 1 < len(idx) else -1
    dn = prof.iloc[i - 1] if i - 1 >= 0 else -1
    if up >= dn: j += 1; acc += up
    else: i -= 1; acc += dn
OUT["levels"] = dict(ORH=ORH, ORL=ORL, POC=round(poc, 2), VAH=round(idx[j], 2), VAL=round(idx[i], 2),
                     profile={f"{k:.2f}": int(v) for k, v in prof.items()})

# ---------------------------------------------------------------- 父单聚合（同 venue 同 ts 同方向 = 一个主动单）
TT = T[T.side.isin(["B", "A"])].copy()
par = TT.groupby(["publisher_id", "ts_event", "side"], as_index=False).agg(size=("size", "sum"), price=("price", "last"),
                                                                           pmin=("price", "min"), pmax=("price", "max"))
par["et"] = par.ts_event.dt.tz_convert(TZ)
par = par.sort_values("ts_event").reset_index(drop=True)
q = par["size"].quantile([.9, .95, .98, .99]).to_dict()
P90, P95, P98, P99 = [float(q[k]) for k in (.9, .95, .98, .99)]
big = par[par["size"] >= P95]; small = par[par["size"] < P95]
f = lambda x, s: int(x.loc[x.side == s, "size"].sum())
OUT["cvd_layers"] = dict(P90=P90, P95=P95, P98=P98, P99=P99, n_parent=len(par),
                         big_TB=f(big, "B"), big_TA=f(big, "A"), big_net=f(big, "B") - f(big, "A"),
                         small_TB=f(small, "B"), small_TA=f(small, "A"), small_net=f(small, "B") - f(small, "A"))
# 原始逐笔分位（对照）
OUT["cvd_layers"]["raw_print_pctl"] = {str(k): float(v) for k, v in TT["size"].quantile([.9, .95, .98, .99]).items()}

# CVD 曲线
TT["sgn"] = np.where(TT.side == "B", 1, -1) * TT["size"]
TT["cvd"] = TT.sgn.cumsum()
pk = TT.loc[TT.cvd.idxmax()]; tg = TT.loc[TT.cvd.idxmin()]
cvd30 = TT.set_index("et").cvd.resample("30min").last()
d30 = TT.set_index("et").sgn.resample("30min").sum()
last2h = TT[TT.et >= CLOSE - pd.Timedelta("2h")].sgn.sum()
OUT["cvd"] = dict(peak=int(pk.cvd), peak_t=str(pk.et.time())[:8], trough=int(tg.cvd), trough_t=str(tg.et.time())[:8],
                  end=int(TT.cvd.iloc[-1]), last2h=int(last2h),
                  by30=[(str(k.time())[:5], int(v), int(c)) for (k, v), c in zip(d30.items(), cvd30.values)])

# ---------------------------------------------------------------- §3.1 委比快照
def wratio(row):
    b = np.array([row[f"bid_s{k}"] for k in range(1, 6)], float)
    a = np.array([row[f"ask_s{k}"] for k in range(1, 6)], float)
    wb, wa = (b * W).sum(), (a * W).sum()
    return wb, wa, (wb - wa) / (wb + wa) * 100 if wb + wa else np.nan

dc = d[cont & d.bid_p1.notna() & d.ask_p1.notna()]
ts_arr = dc.ts_event.values

def dom_before(ts, lag="0s"):
    k = np.searchsorted(ts_arr, (ts - pd.Timedelta(lag)).to_datetime64(), side="right") - 1
    return dc.iloc[max(k, 0)]

snaps = []
for h in pd.date_range(OPEN, CLOSE - pd.Timedelta("30min"), freq="30min"):
    k = np.searchsorted(ts_arr, h.to_datetime64(), side="left")
    r = dc.iloc[min(k, len(dc) - 1)]
    wb, wa, wr = wratio(r)
    snaps.append(dict(t=str(r.et.time())[:8], trig="30min bar", wb=round(wb), wa=round(wa), wr=round(wr, 1), flow="—"))
for _, p in par[par["size"] >= P99].sort_values("size", ascending=False).head(12).iterrows():
    r = dom_before(p.ts_event, "1s")
    wb, wa, wr = wratio(r)
    flow = "顺" if (wr > 0 and p.side == "B") or (wr < 0 and p.side == "A") else "逆"
    snaps.append(dict(t=str(p.et.time())[:8], trig=f"P99大单 {p.side} {int(p['size'])}@{p.price}", wb=round(wb), wa=round(wa), wr=round(wr, 1), flow=flow))
lv = OUT["levels"]
for name in ["ORH", "ORL", "POC", "VAH", "VAL"]:
    L = lv[name]
    after = allpx[(allpx.et >= OPEN + pd.Timedelta("30min")) & ((allpx.price - L).abs() < 1e-9)]
    if len(after):
        r = dom_before(after.et.iloc[0], "1s"); wb, wa, wr = wratio(r)
        snaps.append(dict(t=str(after.et.iloc[0].time())[:8], trig=f"首触{name}={L}", wb=round(wb), wa=round(wa), wr=round(wr, 1), flow="—"))
OUT["weighted_ratio"] = snaps
# 全天时间加权委比（参考）
dd = dc.copy()
bs = sum(dd[f"bid_s{k}"] * W[k - 1] for k in range(1, 6)); as_ = sum(dd[f"ask_s{k}"] * W[k - 1] for k in range(1, 6))
dd["wr"] = (bs - as_) / (bs + as_) * 100
dd["dur"] = dd.ts_event.diff().shift(-1).dt.total_seconds().fillna(0)
dd["wrd"] = dd.wr * dd.dur
OUT["weighted_ratio_timeavg"] = round(float(dd.wrd.sum() / dd.dur.sum()), 1)
_g = dd.set_index("et")[["wrd", "dur"]].resample("30min").sum()
OUT["weighted_ratio_by30"] = {str(k.time())[:5]: round(float(r.wrd / r.dur), 1) for k, r in _g.iterrows() if r.dur > 0}

# ---------------------------------------------------------------- §3.2 ISO 扫单（5ms 内同向 ≥2 档，量≥P99）
iso = []
for s in ["B", "A"]:
    x = TT[TT.side == s].sort_values("ts_event")
    g = (x.ts_event.diff() > pd.Timedelta("5ms")).cumsum()
    for _, grp in x.groupby(g):
        if grp.price.nunique() >= 2 and grp["size"].sum() >= P99:
            iso.append(dict(t=str(grp.et.iloc[0].time())[:12], side=s, levels=int(grp.price.nunique()),
                            size=int(grp["size"].sum()), pmin=float(grp.price.min()), pmax=float(grp.price.max()),
                            venues=int(grp.publisher_id.nunique())))
iso = pd.DataFrame(iso)
OUT["iso"] = dict(list=iso.sort_values("size", ascending=False).head(15).to_dict("records") if len(iso) else [],
                  buy_n=int((iso.side == "B").sum()) if len(iso) else 0, buy_v=int(iso.loc[iso.side == "B", "size"].sum()) if len(iso) else 0,
                  sell_n=int((iso.side == "A").sum()) if len(iso) else 0, sell_v=int(iso.loc[iso.side == "A", "size"].sum()) if len(iso) else 0)

# ---------------------------------------------------------------- 订单生命周期（键 = publisher, channel, order_id）
o = d[d.action.isin(["A", "C", "F", "M"]) & (d.order_id != 0)].copy()
o["key"] = list(zip(o.publisher_id, o.channel_id, o.order_id))
adds = o[o.action == "A"].drop_duplicates("key", keep="first").set_index("key")
canc = o[o.action == "C"].groupby("key").agg(c_ts=("ts_event", "last"))
fills = o[o.action == "F"].groupby("key").agg(f_size=("size", "sum"), f_ts=("ts_event", "first"))
mods = o[o.action == "M"].groupby("key").agg(m_n=("size", "size"), m_pmin=("price", "min"), m_pmax=("price", "max"), m_smax=("size", "max"))
L = adds.join(canc).join(fills).join(mods)
L["f_size"] = L.f_size.fillna(0)
L["orig"] = np.maximum(L["size"], L.m_smax.fillna(0))
L["fill_ratio"] = L.f_size / L.orig
L["life"] = (L.c_ts - L.ts_event).dt.total_seconds()
L["price_moved"] = L.m_n.notna() & ((L.m_pmin != L.price) | (L.m_pmax != L.price))

def level_of(r):
    if r.side == "B":
        for k in range(1, 6):
            if r[f"bid_p{k}"] == r.price: return k
        return 99 if pd.notna(r.bid_p1) and r.price < r.bid_p1 else 0
    for k in range(1, 6):
        if r[f"ask_p{k}"] == r.price: return k
    return 99 if pd.notna(r.ask_p1) and r.price > r.ask_p1 else 0
L["lvl"] = L.apply(level_of, axis=1)
L["dist"] = np.where(L.side == "B", (L.bid_p1 - L.price) / TICK, (L.price - L.ask_p1) / TICK).round()
L["et"] = L.ts_event.dt.tz_convert(TZ)
asz = L["size"].quantile([.9, .98, .99]).to_dict()
OUT["add_size_pctl"] = {str(k): float(v) for k, v in asz.items()}

# ---------------------------------------------------------------- §3.3 挂单墙（size≥P99 且存活>15s）
P99A = float(asz[.99])
day_lo, day_hi = OUT["price"]["low"], OUT["price"]["high"]
wall = L[(L["size"] >= P99A) & L.c_ts.notna() & (L.life > 15) & (L.price >= day_lo - 5 * TICK) & (L.price <= day_hi + 5 * TICK)
         & (L.et >= OPEN) & (L.et < CLOSE)].copy()
wall["lvl_s"] = wall.apply(lambda r: (("B" if r.side == "B" else "A") + str(r.lvl)) if 1 <= r.lvl <= 5 else "OUTSIDE_L5", axis=1)
OUT["walls"] = dict(P99=P99A, n=len(wall),
                    B_n=int((wall.side == "B").sum()), B_v=int(wall.loc[wall.side == "B", "size"].sum()),
                    A_n=int((wall.side == "A").sum()), A_v=int(wall.loc[wall.side == "A", "size"].sum()),
                    B_n_L1_5=int(((wall.side == "B") & wall.lvl.between(1, 5)).sum()),
                    B_v_L1_5=int(wall.loc[(wall.side == "B") & wall.lvl.between(1, 5), "size"].sum()),
                    A_n_L1_5=int(((wall.side == "A") & wall.lvl.between(1, 5)).sum()),
                    A_v_L1_5=int(wall.loc[(wall.side == "A") & wall.lvl.between(1, 5), "size"].sum()),
                    top=wall.sort_values("size", ascending=False).head(12)[["et", "price", "side", "size", "life", "lvl_s", "f_size"]]
                    .assign(et=lambda x: x.et.dt.strftime("%H:%M:%S")).to_dict("records"),
                    by30=wall.assign(b=wall.et.dt.floor("30min").dt.strftime("%H:%M")).groupby(["b", "side"])["size"].sum().unstack(fill_value=0).astype(int).to_dict("index"))

# ---------------------------------------------------------------- §3.5 谎骗确认（150s 穿越）
lit_tr = d[(d.action == "T") & (d.et >= OPEN) & (d.et <= CLOSE + pd.Timedelta("1s"))][["ts_event", "price"]].sort_values("ts_event")
tts = lit_tr.ts_event.values; tpx = lit_tr.price.values
cand = L[L.lvl.between(1, 5) & L.c_ts.notna() & (L.fill_ratio < 0.15) & (L.et >= OPEN) & (L.c_ts < CLOSE)].copy()

def spoof_eval(r):
    if r.price_moved: return "UNDET"
    if r.c_ts + pd.Timedelta("150s") > CLOSE: return "UNDET"
    a = np.searchsorted(tts, r.c_ts.to_datetime64(), side="right")
    b = np.searchsorted(tts, (r.c_ts + pd.Timedelta("150s")).to_datetime64(), side="right")
    w = tpx[a:b]
    if len(w) == 0: return "PROP"
    if r.side == "B": return "SPOOF" if (w < r.price - 1e-9).any() else "PROP"
    return "SPOOF" if (w > r.price + 1e-9).any() else "PROP"
cand["res"] = cand.apply(spoof_eval, axis=1)

def spoof_table(c):
    t = {}
    for s in ["B", "A"]:
        x = c[c.side == s]; n = len(x)
        t[s] = dict(n=n, spoof=int((x.res == "SPOOF").sum()), prop=int((x.res == "PROP").sum()), undet=int((x.res == "UNDET").sum()),
                    spoof_pct=round((x.res == "SPOOF").mean() * 100, 1) if n else None,
                    spoof_vol=int(x.loc[x.res == "SPOOF", "size"].sum()))
    return t
OUT["spoof_all"] = spoof_table(cand)
OUT["spoof_ge_p90"] = spoof_table(cand[cand["size"] >= asz[.9]])
OUT["spoof_ge_p90_threshold"] = float(asz[.9])
# 极速撤单 ≤1s（撤单 / 全部撤单）
cc = L[L.c_ts.notna() & (L.et >= OPEN) & (L.et < CLOSE)]
OUT["fast_cancel_le1s"] = {s: round(float((cc[cc.side == s].life <= 1).mean() * 100), 1) for s in ["B", "A"]}
# 深预埋撤单
P98A = float(asz[.98])
deep = L[(L["size"] >= P98A) & (L.dist > 10) & (L.life > 120) & (L.f_size == 0) & L.c_ts.notna() & (L.et >= OPEN - pd.Timedelta("2h"))]
OUT["deep_cancel"] = dict(P98=P98A, B_n=int((deep.side == "B").sum()), B_v=int(deep.loc[deep.side == "B", "size"].sum()),
                          A_n=int((deep.side == "A").sum()), A_v=int(deep.loc[deep.side == "A", "size"].sum()))

# ---------------------------------------------------------------- §3.6 TRF 分类
trf_c = trf_c.sort_values("ts_event")
k = np.searchsorted(ts_arr, trf_c.ts_event.values, side="right") - 1
dom = dc.iloc[np.clip(k, 0, None)][["bid_p1", "ask_p1"]].reset_index(drop=True)
trf_c = trf_c.reset_index(drop=True).join(dom)
trf_c["cls"] = np.where(trf_c.price >= trf_c.ask_p1, "near_ask", np.where(trf_c.price <= trf_c.bid_p1, "near_bid", "mid"))
tv = trf_c.groupby("cls")["size"].sum(); ttot = tv.sum()
rng_lo, rng_hi = OUT["price"]["low"], OUT["price"]["high"]
trf_c["pctl"] = (trf_c.price - rng_lo) / (rng_hi - rng_lo) * 100
p995 = trf_c["size"].quantile(.995)
top3 = trf_c.assign(px=trf_c.price.round(3)).groupby("px")["size"].sum().sort_values(ascending=False).head(3)
OUT["trf"] = dict(total=int(ttot), near_ask=int(tv.get("near_ask", 0)), near_bid=int(tv.get("near_bid", 0)), mid=int(tv.get("mid", 0)),
                  near_ask_pct=round(tv.get("near_ask", 0) / ttot * 100, 1), near_bid_pct=round(tv.get("near_bid", 0) / ttot * 100, 1),
                  mid_pct=round(tv.get("mid", 0) / ttot * 100, 1), net=int(tv.get("near_ask", 0) - tv.get("near_bid", 0)),
                  p995=float(p995),
                  top5=trf_c.sort_values("size", ascending=False).head(5)[["et", "price", "size", "cls", "bid_p1", "ask_p1"]]
                  .assign(et=lambda x: x.et.dt.strftime("%H:%M:%S")).to_dict("records"),
                  top3_px=[(float(p), int(v), round((p - rng_lo) / (rng_hi - rng_lo) * 100, 1)) for p, v in top3.items()],
                  vwap_pctl=round(float(((trf_c.pctl * trf_c["size"]).sum()) / ttot), 1),
                  zone_vol={z: int(trf_c.loc[m, "size"].sum()) for z, m in
                            {"<30%": trf_c.pctl < 30, "30-70%": trf_c.pctl.between(30, 70), ">70%": trf_c.pctl > 70}.items()},
                  by30=trf_c.assign(b=trf_c.et.dt.floor("30min").dt.strftime("%H:%M")).groupby(["b", "cls"])["size"].sum().unstack(fill_value=0).astype(int).to_dict("index"))

# ---------------------------------------------------------------- §3.7 预谋强度（size≥P98 的被动成交单）
pm = L[(L["size"] >= P98A) & (L.f_size > 0)].copy()
pm["T_travel"] = (pm.f_ts - pm.ts_event).dt.total_seconds()
pm["cls"] = np.where((pm.dist > 10) & (pm.T_travel > 120), "STRONG", np.where(pm.dist <= 2, "REALTIME", "FAST"))
pmt = {}
for s in ["B", "A"]:
    x = pm[pm.side == s]
    pmt[s] = {c: dict(n=int((x.cls == c).sum()), add_v=int(x.loc[x.cls == c, "size"].sum()), fill_v=int(x.loc[x.cls == c, "f_size"].sum())) for c in ["STRONG", "FAST", "REALTIME"]}
st = pm[pm.cls == "STRONG"]
pub_conc = st.publisher_id.value_counts(normalize=True).to_dict() if len(st) else {}
OUT["premed"] = dict(P98=P98A, table=pmt, pub_conc={int(k): round(v * 100, 1) for k, v in pub_conc.items()},
                     strong_list=st.sort_values("f_size", ascending=False).head(12)[["et", "side", "price", "size", "f_size", "dist", "T_travel", "publisher_id"]]
                     .assign(et=lambda x: x.et.dt.strftime("%H:%M:%S")).to_dict("records"))

# ---------------------------------------------------------------- 低/高价区被动吸收率（F 成交按价区）
Fz = Fc.copy()
Fz["zone"] = np.where(Fz.price <= 2.69, "LOW", np.where(Fz.price >= 2.72, "HIGH", "MID"))
zz = Fz.groupby(["zone", "side"])["size"].sum().unstack(fill_value=0)
OUT["passive_by_zone"] = {z: dict(FB=int(r.get("B", 0)), FA=int(r.get("A", 0)),
                                  bid_absorb_pct=round(r.get("B", 0) / (r.get("B", 0) + r.get("A", 0)) * 100, 1)) for z, r in zz.iterrows()}

# ---------------------------------------------------------------- Liquidity Hole（价差异常扩大 + 快速位移）
dc2 = dc[["et", "bid_p1", "ask_p1"]].copy()
dc2["spr"] = ((dc2.ask_p1 - dc2.bid_p1) / TICK).round()
dc2["dur"] = dc2.et.diff().shift(-1).dt.total_seconds().fillna(0)
OUT["spread"] = dict(time_w_mean=round(float((dc2.spr * dc2.dur).sum() / dc2.dur.sum()), 2),
                     pct_time_ge3=round(float(dc2.loc[dc2.spr >= 3, "dur"].sum() / dc2.dur.sum() * 100), 2),
                     max=int(dc2.spr.max()))
wide = dc2[(dc2.spr >= 3) & (dc2.dur >= 1)]
OUT["spread"]["wide_episodes"] = wide.assign(et=wide.et.dt.strftime("%H:%M:%S"))[["et", "bid_p1", "ask_p1", "spr", "dur"]].head(15).to_dict("records")

# ---------------------------------------------------------------- 关键时段 15:50~16:00 及收盘竞价
pre = T[T.et >= CLOSE - pd.Timedelta("10min")]
OUT["last10"] = dict(TB=f(pre, "B"), TA=f(pre, "A"), TN=f(pre, "N"),
                     px_open=float(pre.price.iloc[0]) if len(pre) else None, px_last=float(pre.price.iloc[-1]) if len(pre) else None)
last_dom = dc.iloc[-1]
OUT["last_dom_before_close"] = {k: (float(last_dom[k]) if "p" in k else int(last_dom[k])) for k in
                                [f"{s}_{t}{i}" for i in range(1, 6) for s in ("bid", "ask") for t in ("p", "s")]}

# 10:55 炉边谈话窗口
fw = allpx[(allpx.et >= day + pd.Timedelta("10:50:00")) & (allpx.et < day + pd.Timedelta("11:50:00"))]
fwT = TT[(TT.et >= day + pd.Timedelta("10:50:00")) & (TT.et < day + pd.Timedelta("11:50:00"))]
OUT["fireside_window"] = dict(px_1050=float(fw.price.iloc[0]), px_min=float(fw.price.min()), px_min_t=str(fw.loc[fw.price.idxmin(), "et"].time())[:8],
                              px_1150=float(fw.price.iloc[-1]), delta=int(fwT.sgn.sum()), vol=int(fw["size"].sum()))

def conv(o_):
    if isinstance(o_, dict): return {str(k): conv(v) for k, v in o_.items()}
    if isinstance(o_, (list, tuple)): return [conv(v) for v in o_]
    if isinstance(o_, (np.integer,)): return int(o_)
    if isinstance(o_, (np.floating,)): return float(o_)
    if isinstance(o_, (np.bool_,)): return bool(o_)
    if isinstance(o_, pd.Timestamp): return str(o_)
    return o_
json.dump(conv(OUT), open(sys.argv[2], "w"), ensure_ascii=False, indent=1, default=str)
print("ok")
