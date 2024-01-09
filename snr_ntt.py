import numpy as np
from helpers import get_info_from_log
from tqdm import tqdm, trange
from scalib.metrics import SNR
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
color_names = mcolors.XKCD_COLORS
KYBER_Q = 3329

def get_info_from_log(dir_path):
    f_log = f"{dir_path}/log"
    info = {}
    keys = ["n_files", "poly_per_batch", "n_batchs", "n_shares", "m_flag", "fname", "n_samples"]
    with open(f_log, "r") as f:
        f_ls = f.readlines()
        for line in f_ls:
            for key in keys:
                if key in line:
                    val = line.rstrip().split("=")[1] if key=="fname" else int( line.rstrip().split("=")[1])
                    info[key] = val
                    break
    return info


def snr_ntt(d_name, share, c_idx, n_files):
    dir_path = f"traces/{d_name}"
    print(dir_path)
    info = get_info_from_log(dir_path)
    fname = info["fname"]
    n_shares = info["n_shares"]
    n_files = info["n_files"] if n_files is None else n_files
    n_samples = info["n_samples"]*2
    traces_per_file = info["poly_per_batch"]*info["n_batchs"]
    snr = SNR(KYBER_Q, n_samples, len(c_idx), use_64bit=True)
    for fi in trange(n_files, desc="File"):
        f_t = f"{dir_path}/{fname}_{n_shares}_traces_{fi}.npy"
        f_d = f"{dir_path}/{fname}_{n_shares}_meta_{fi}.npz"
        traces = np.load(f_t)
        print(traces.shape)
        traces_i = traces.copy()
        data = np.load(f_d)["polynoms"]
        coeff = data[:, c_idx+256*share].astype(np.uint16)
        snr.fit_u(traces_i, coeff)
    snr_val = snr.get_snr()
    for i, c in enumerate(c_idx):
        plt.plot(range(n_samples), snr_val[i], label=f"coeff {c}")
    plt.legend(fontsize=8)
    plt.title(f"SNR share {share+1}")
    plt.show()
if __name__ == '__main__':
    c_idx = np.arange(0, 10)
    snr_ntt("190923_1613", 0, c_idx, n_files=5)
