"""Dựng cache snapshot + ma trận kề cho mọi decision date (tháng: đồ thị 60 và 250 phiên; tuần: 60 phiên)."""
import pickle, os, time
from multiprocessing import Pool
import pandas as pd
import common as C

_df = None
def _init():
    global _df
    _df = C.load_df()
def _work(d):
    return C.compute_one((_df, d))

if __name__ == '__main__':
    df = C.load_df()
    sets = {f: set(pd.Timestamp(d) for v in C.quiet_split(df, f).values() for d in v) for f in ('M', 'W')}
    dates = sorted(sets['M'] | sets['W'])
    print(len(dates), 'decision dates'); t = time.time()
    with Pool(4, initializer=_init) as p:
        out = dict(p.imap_unordered(_work, dates, chunksize=4))
    for f, g in (('M', 'A60'), ('M', 'A250'), ('W', 'A60')):
        sl = {d: {'X': s['X'], 'mask': s['mask'], 'symbols': s['symbols'], g: s[g]}
              for d, s in out.items() if d in sets[f]}
        pickle.dump(sl, open(os.path.join(C.CACHE, f'snap_{f}_{g}.pkl'), 'wb'))
    print('xong', round(time.time() - t), 's')
