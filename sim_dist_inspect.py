import numpy as np
from matplotlib import pyplot as plt
from helpers import *
from lda_projection import *

def combine_L_dist(combif):
    N = 100000
    snr = 100
    sigma = np.sqrt(2.67/snr)
    # fig, axs = plt.subplots(1, 5)
    for i, s in enumerate(S_SET):
        secrets = np.ones((N, 10), dtype=np.int16)*s
        shares = gen_shares(secrets, n_shares=2)
        leakage = gen_leakages(shares, sigma, combif=combif, model=ID)
        # Lc = leakage.mean(axis=1)
        p, x = np.histogram(leakage, bins=100, density=True)
        # print(s, Lc.mean(), np.var(Lc))
        plt.plot(x[:-1], p, label=f"s={s}")
        # axs[i].plot(x[:-1], p)
        # axs[i].set_title(f"s={s}")
    plt.legend()
    plt.title(combif)
    plt.show()
if __name__ == '__main__':
    combine_L_dist("norm_prod")
    combine_L_dist("prod")
    combine_L_dist("abs_diff")
    combine_L_dist("sum")
