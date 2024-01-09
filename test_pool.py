import time
import numpy as np


from multiprocessing import Pool, Manager, Value, Array, Process
from functools import partial
import multiprocessing.pool


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc

def rotr(x, c):
    return (x >> c) | ((x << (16 - c)) & 2 ** 16)

def popcount_py(x):
        return bin(x).count("1")

def lbox(x, y):
    a = x ^ rotr(x, 12)
    b = y ^ rotr(y, 12)
    a = a ^ rotr(a, 3)
    b = b ^ rotr(b, 3)
    a = a ^ rotr(x, 17)
    b = b ^ rotr(y, 17)
    c = a ^ rotr(a, 31)
    d = b ^ rotr(b, 31)
    a = a ^ rotr(d, 26)
    b = b ^ rotr(c, 25)
    a = a ^ rotr(c, 15)
    b = b ^ rotr(d, 15)
    b = rotr(b, 1)
    #return a, b
    return popcount_py(x) + popcount_py(a)

def fun(X, Y):

    k = []
    for x in X:
        for y in Y:
            #if x != y:
            k.append(lbox(x, y))
            #k = np.min(k)
           # print(k)
    return k

def FUN(XXYY):
    print(len(XXYY))
    pool = Pool(len(XXYY))
    res = list(pool.apply_async(fun, args=(XY)) for XY in XXYY)
    pool.close()
    pool.join()

    results = [r.get() for r in res]
    return np.array(results).min()


# protect the entry point
if __name__ == '__main__':
    start = time.time()

    X1 = np.arange(1, 200001, dtype=np.uint32)
    Y1 = np.arange(1, 200001, dtype=np.uint32)
    # X2 = np.arange(536870913, 1073741824, dtype=np.uint32)
    # Y2 = np.arange(536870913, 1073741824, dtype=np.uint32)
    # X3 = np.arange(1073741825, 1610612736, dtype=np.uint32)
    # Y3 = np.arange(1073741825, 1610612736, dtype=np.uint32)
    XY1 = np.array([X1, Y1])
    print(XY1.shape)
    # XY2 = np.array([X2, Y2])
    # XY3 = np.array([X3, Y3])
    # X = np.hstack((X1, X2))
    # Y = np.hstack((Y1, Y2))

    XXYY1 = np.split(XY1, 10, axis=1)
    # XXYY2 = [XY1, XY2, XY3]
    # XXYY3 = [XY1, XY2, XY3]
    FUN(XXYY1)
    # C = [[XXYY1], [XXYY1]]

    # pool = NoDaemonProcessPool(len(C))
    # res_ = list(pool.apply_async(FUN, args=(m)) for m in C)
    # pool.close()
    # pool.join()
    # #
    # results = [r.get() for r in res_]
    # print(results)


    # C = [XXYY1, XXYY2]
    #
    # # [print(len(x)) for x in C]
    # res =
    # return np.array(results).min()
    #
    # res = FUN(XXYY)
    # print(res)
    # xy1 = np.column_stack((X1, Y1))
    # xy1 = np.transpose(xy1)
    # xy2 = np.column_stack((X1, Y1))
    # xy2 = np.transpose(xy2)
    # XY = np.hstack((xy1, xy2))
    # FUN(X1, Y1)
    # fun(X1, Y1)

    # XY_ = np.split(XY, 2, axis=0)
    # print(len(XY_))
    # with NoDaemonProcessPool(2) as pool:
    #     RES  = pool.apply_async(FUN, args=(X, Y))
    #     print(RES.get())
    #print(mp.cpu_count())
    # with Pool(2) as pool:  # start 4 worker processes


    # print(results)



        # result1 = pool.apply_async(fun, (X1, Y1))  # evaluate "f(10)" asynchronously in a single process
        # B1 = result1.get()
        # print(B1)
    #     print(np.min(B1))
    #     result2 = pool.apply_async(fun, (X2, Y2))  # evaluate "f(10)" asynchronously in a single process
    #     B2 = result2.get()
    #     #print(B1)
    #     print(np.min(B2))
    elapsed = (time.time() - start)
    print("\n", "time elapsed is :", elapsed)
