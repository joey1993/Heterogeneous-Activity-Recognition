#!/usr/bin/env python3

import sys
from collections import Counter
INF = -1
try:
    import ujson as json
except:
    import json

def parse(line):
    data = line.strip().split(',')
    return {
            'time': int(data[2]),
            'data': data[3:6],
            'user': tuple(data[6:9]),
            'lbl': data[9]
           }


if __name__ == '__main__':
    dev = sys.argv[1]
    sec = int(sys.argv[2])
    S = 1000000000
    
    PATH_A = '../../raw/%s_accelerometer.csv.sort' % dev
    PATH_G = '../../raw/%s_gyroscope.csv.sort' % dev

    print('Start loading data from raw files', file=sys.stderr)
    data_user_a = {}
    data_user_g = {}
    with open(PATH_A, 'r') as fin:
        for line in fin:
            d = parse(line)
            u = d['user']
            if u not in data_user_a:
                data_user_a[u] = []
            data_user_a[u].append(d)

    with open(PATH_G, 'r') as fin:
        for line in fin:
            d = parse(line)
            u = d['user']
            if u not in data_user_g:
                data_user_g[u] = []
            data_user_g[u].append(d)

    user_a = set(data_user_a.keys())
    user_g = set(data_user_g.keys())
    
    users = user_g & user_a
    ulist = sorted(list(users))

    print('Start partitioning data', file=sys.stderr)
    with open('../data_%s_%d.json' % (dev, sec), 'w') as fout:
        for u in ulist:
            print(u, file=sys.stderr)
            dg = data_user_g[u]
            da = data_user_a[u]
            itr_g = itr_a = 0

            beg_time = INF
            beg_time = da[itr_a]['time'] if itr_a < len(da) and (beg_time == INF or da[itr_a]['time'] < beg_time) else beg_time
            beg_time = dg[itr_g]['time'] if itr_g < len(dg) and (beg_time == INF or dg[itr_g]['time'] < beg_time) else beg_time
            end_time = beg_time + S * sec
            while itr_g < len(dg) or itr_a < len(da):
                cnt = Counter()
                glst = []
                alst = []
                while itr_g < len(dg) and dg[itr_g]['time'] < end_time:
                    glst.append({'time': dg[itr_g]['time'], 'data': dg[itr_g]['data']})
                    cnt[dg[itr_g]['lbl']] += 1
                    itr_g += 1
                while itr_a < len(da) and da[itr_a]['time'] < end_time:
                    alst.append({'time': da[itr_a]['time'], 'data': da[itr_a]['data']})
                    cnt[da[itr_a]['lbl']] += 1
                    itr_a += 1
                # output
                opt = {
                        'begin_time': beg_time,
                        'end_time': end_time,
                        'gdata': glst,
                        'adata': alst,
                        'user': u,
                        'label': cnt.most_common(1)[0][0]
                      }
                print(json.dumps(opt), file=fout)
                # re-init
                beg_time = INF
                beg_time = da[itr_a]['time'] if itr_a < len(da) and (beg_time == INF or da[itr_a]['time'] < beg_time) else beg_time
                beg_time = dg[itr_g]['time'] if itr_g < len(dg) and (beg_time == INF or dg[itr_g]['time'] < beg_time) else beg_time
                end_time = beg_time + S * sec
                
