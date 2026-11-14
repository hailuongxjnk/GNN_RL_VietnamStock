"""Dựng cache snapshot + ma trận kề cho mọi decision date (tuần và tháng). Chạy 1 lần."""
import pickle, sys, time
from multiprocessing import Pool
import common as C

_df = None
def _init():
    global _df
    _df = C.load_df()
def _work(d):
    return C.compute_one((_df, d))

if __name__ == '__main__':
    df = C.load_df()
    dates = set()
    for f in ('W', 'M'):
        for v in C.quiet_split(df, f).values():
            dates |= set(v)
    dates = sorted(dates)
    print(len(dates), 'decision dates'); t = time.time()
    with Pool(4, initializer=_init) as p:
        out = dict(p.imap_unordered(_work, dates, chunksize=4))
    with open(C.snapshot_cache_path(), 'wb') as f:
        pickle.dump(out, f)
    print('xong', round(time.time() - t), 's')
