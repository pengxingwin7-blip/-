"""
从 MBO 逐单重建跨4场所(XNAS/BATS/XNYS/ARCX)合并盘口 L1~L5。

原因：证据包 07_depth5 的 ask_p1 在 RTH 全程为 1.27/size=0 的幽灵档（盘口交叉），
不可直接使用。本脚本按 Databento MBO 语义重建：
  A=新增  C=撤单(size=撤掉的量，可部分)  M=改单(price/size为新值)
  F=被动成交(size=成交量)  T=主动成交(不改簿)  R=清空该publisher的簿
订单键 = (publisher_id, channel_id, order_id)
输出：每个事件行附带 事件后(post) 的 L1~L5，以及 事件前(pre) 的 L1 价格。
"""
import sys
import heapq
import pandas as pd
import numpy as np

src, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
# 按事件时间 + 交易所序号稳定排序（不同publisher之间以ts_event为准）
df = df.sort_values(['ts_event', 'publisher_id', 'sequence'], kind='mergesort').reset_index(drop=True)

orders = {}                 # key -> [side, price, size]
levels = {'B': {}, 'A': {}}  # side -> {price: total_size}


def lvl_add(side, px, sz):
    d = levels[side]
    d[px] = d.get(px, 0) + sz
    if d[px] <= 0:
        del d[px]


def top5(side):
    d = levels[side]
    if side == 'B':
        ps = heapq.nlargest(5, d.keys())
    else:
        ps = heapq.nsmallest(5, d.keys())
    ps = ps + [np.nan] * (5 - len(ps))
    ss = [d.get(p, 0) if p == p else 0 for p in ps]
    return ps, ss


rows_post = []
pre_bid1, pre_ask1 = [], []
anomalies = 0
for r in df.itertuples(index=False):
    # 记录事件前 L1
    pb = max(levels['B']) if levels['B'] else np.nan
    pa = min(levels['A']) if levels['A'] else np.nan
    pre_bid1.append(pb)
    pre_ask1.append(pa)

    key = (r.publisher_id, r.channel_id, r.order_id)
    act = r.action
    if act == 'R':
        # 清空该publisher全部订单
        for k in [k for k in orders if k[0] == r.publisher_id]:
            s, p, z = orders.pop(k)
            lvl_add(s, p, -z)
    elif act == 'A' and r.side in ('B', 'A'):
        if key in orders:  # 重复add视为替换
            s, p, z = orders.pop(key)
            lvl_add(s, p, -z)
        orders[key] = [r.side, round(r.price, 4), int(r.size)]
        lvl_add(r.side, round(r.price, 4), int(r.size))
    elif act == 'M' and key in orders:
        s, p, z = orders[key]
        lvl_add(s, p, -z)
        orders[key] = [s, round(r.price, 4), int(r.size)]
        lvl_add(s, round(r.price, 4), int(r.size))
    elif act in ('C', 'F') and key in orders:
        s, p, z = orders[key]
        dz = min(int(r.size), z)
        orders[key][2] = z - dz
        lvl_add(s, p, -dz)
        if orders[key][2] <= 0:
            del orders[key]
    elif act in ('C', 'F', 'M'):
        anomalies += 1  # 找不到订单（可能是开盘前存量或隐藏单）

    bp, bs = top5('B')
    ap, as_ = top5('A')
    rows_post.append(bp + bs + ap + as_)

cols = [f'bid_p{i}' for i in range(1, 6)] + [f'bid_s{i}' for i in range(1, 6)] + \
       [f'ask_p{i}' for i in range(1, 6)] + [f'ask_s{i}' for i in range(1, 6)]
book = pd.DataFrame(rows_post, columns=cols)
base = df[['ts_event', 'publisher_id', 'channel_id', 'order_id', 'action', 'side',
           'price', 'size', 'flags', 'sequence']].copy()
out_df = pd.concat([base, book], axis=1)
out_df['pre_bid1'] = pre_bid1
out_df['pre_ask1'] = pre_ask1
out_df['et'] = out_df.ts_event.dt.tz_convert('America/New_York')
out_df.to_parquet(out)
print('rows', len(out_df), 'orphan C/F/M', anomalies)
rth = out_df[(out_df.et.dt.strftime('%H:%M:%S') >= '09:30:00') & (out_df.et.dt.strftime('%H:%M:%S') < '16:00:00')]
print('RTH crossed/locked ratio', (rth.bid_p1 >= rth.ask_p1).mean().round(4))
print('RTH spread median', (rth.ask_p1 - rth.bid_p1).median())
