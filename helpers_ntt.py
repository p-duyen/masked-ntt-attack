import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm, trange
import os

from scalib.metrics import SNR
color_names_ = [name for name in  mcolors.XKCD_COLORS]
# color_names_ = [name for name in mcolors.CSS4_COLORS]
color_names = color_names_[10:]

# import sys
# sys.path.append("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements")
# from arg_parsing import *
KYBER_Q=3329
HW_Q = 12
#width=1
width = os.get_terminal_size().columns
#
def CE(resX):
    return np.nansum(-(np.log2(resX) * resX), axis=1).mean()
def centered_pr(str):
    print(str.center(width))
def count_1(x):
    return int(x).bit_count()
fcount = np.vectorize(count_1)
def HW(x):
    return fcount(x)

hw_q = HW(np.arange(KYBER_Q))

def last_nbits(x, nbits):
    bin_r = np.array(list(np.binary_repr(x).zfill(16))).astype(np.int8)
    # print(bin_r[-(nbits):])
    return bin_r[:n_bits].sum()
flastbits = np.vectorize(last_nbits, excluded=['nbits'])
def LSnB(x, n):
    return flastbits(x, nbits=n)

def bin_convert(x, nbits=12):
    return np.array(list(np.binary_repr(x).zfill(nbits))).astype(np.int8)
def LSbyte_convert(x, nbits=12):
    bin_x = np.array(list(np.binary_repr(x).zfill(nbits))).astype(np.int8)
    # print(x, bin_x, bin_x[4:], bin_x[4:].sum())
    return bin_x[:4].sum()
fbin = np.vectorize(LSbyte_convert)
def LSByte(x):
    return fbin(x)
def bin_repr(X):
    res = []
    for x in X:
        res.append(bin_convert(x))
    return res
def ID(x):
    return x
enc_mode = ["SUB", "ADD"]


class Leakage_Handler:
    def __init__(self, d_name, n_samples=None):
        self.d_name = d_name
        info = get_info_from_log(d_name)
        self.n_shares = info["n_shares"]
        # self.n_samples = n_samples*self.n_shares
        self.n_samples = info["n_samples"]*self.n_shares
        self.n_files = info["n_files"]
        self.traces_per_file = info["poly_per_batch"]*info["n_batchs"]
        self.m_flag = info["m_flag"]
        self.fname = info["fname"]
        self.dir_name = f"traces/{d_name}"
        self.file_temp = f"traces/{d_name}/{self.fname}_{self.n_shares}"

    def get_PoI(self, n_pois, mode, model, keep=False, display=False):
        self.n_pois = n_pois
        if mode=="on_shares":
            if model == HW:
                snr = SNR(HW_Q, self.n_samples, self.n_shares, use_64bit=True)
            else:
                snr = SNR(model(KYBER_Q), self.n_samples, self.n_shares, use_64bit=True)
        elif mode=="on_sec":
            if model is VAL:
                snr = SNR(5, self.n_samples, 1, use_64bit=True)
            else:
                snr = SNR(HW_Q, self.n_samples, 1, use_64bit=True)
        else:
            self.n_pois = self.n_samples
            self.PoI = np.arange(self.n_samples)
            return None
        for fi in trange(self.n_files, desc="SNR|File"):
            f_t = f"{self.file_temp}_traces_{fi}.npy"
            f_d = f"{self.file_temp}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            data = np.load(f_d)["polynoms"]
            if mode=="on_shares":
                polys = data[range(self.traces_per_file), :-1].astype(np.uint16)
            elif mode=="on_sec":
                polys = data[range(self.traces_per_file), -1:].astype(np.uint16)%5
            polys = model(polys).astype(np.uint16)
            snr.fit_u(traces, polys)
        snr_val = snr.get_snr()

        if mode=="on_sec":
            self.PoI = np.argsort(snr_val[0])[-n_pois:]
        elif mode=="on_shares":
            self.PoI = np.zeros((self.n_pois*self.n_shares,), dtype=np.int16)
            for c in range(self.n_shares):
                idx = np.argsort(snr_val[c])[-n_pois:]
                self.PoI[c*self.n_pois: c*self.n_pois+self.n_pois] = idx
        if display:
            if mode=="on_sec":
                plt.plot(snr_val.T, label="secret")
            elif mode=="on_shares":
                for c in range(self.n_shares):
                    plt.plot(range(self.n_samples), snr_val[c], label=f"share {c+1}")
            plt.title(f"SNR {enc_mode[self.m_flag%10]}")
            plt.legend()
            plt.show()
        if keep:
            self.snr = snr_val
        return None
    def traces_trim(self, mode):
        fnew = f"{self.file_temp}_traces_0_seconly.npy" if mode=="on_sec" else f"{self.file_temp}_{self.n_pois}traces_0.npy"
        if os.path.exists(fnew):
            centered_pr("DATA IS READY!")
            pass
        else:
            for fi in trange(self.n_files, desc="Trimming|FILE"):
                f_t = f"{self.file_temp}_traces_{fi}.npy"
                traces = np.load(f_t).astype(np.int16)
                fnew = f"{self.file_temp}_traces_{fi}_seconly.npy" if mode=="on_sec"  else f"{self.file_temp}_{self.n_pois}traces_{fi}.npy"
                with open(fnew, "wb") as f:
                    np.save(f, traces[:, self.PoI].copy())

    def get_data(self, mode):
        if mode=="full_trace":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_samples))
        elif mode=="on_sec":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_pois))
        elif mode=="on_shares":
            L = np.zeros((self.traces_per_file*self.n_files, self.n_pois*self.n_shares))
        elif mode=="on_shares_sep":
            #NOTE: fix this back after being done Analysis
            # L = np.zeros((self.n_shares, self.traces_per_file*self.n_files, self.n_pois))
            L = np.zeros((self.n_shares, self.traces_per_file*self.n_files, self.n_samples))
        labels = np.zeros((self.traces_per_file*self.n_files, self.n_shares+1))
        if mode != "full_trace":
            self.traces_trim(mode)
        for fi in trange(self.n_files, desc="GETTING DATA|FILE"):
            i_f = fi*self.traces_per_file
            if mode in ["full_trace", "on_shares_sep"] :
                f_t = f"{self.file_temp}_traces_{fi}.npy"
            else:
                if mode=="on_sec":
                    f_t = f"{self.file_temp}_traces_{fi}_seconly.npy"
                elif mode=="on_shares":
                    f_t = f"{self.file_temp}_{self.n_pois}traces_{fi}.npy"
            f_d = f"{self.file_temp}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            data = np.load(f_d)["polynoms"]
            labels[i_f: i_f+self.traces_per_file] = data[range(self.traces_per_file), :]
            if mode != "on_shares_sep":
                L[i_f: i_f+self.traces_per_file] = traces
            else:
                for share_i in range(self.n_shares):
                    self.n_samples
                    idx = share_i*self.n_pois
                    # L[share_i, i_f: i_f+self.traces_per_file] = traces[:, self.PoI[idx:idx+self.n_pois]]
                    L[share_i, i_f: i_f+self.traces_per_file] = traces
                    # labels[share_i, i_f: i_f+self.traces_per_file] = data[range(self.traces_per_file), share_i]
        return L.astype(np.float32), labels



def gen_data(d_name, n_points, sec_only=False):
    '''Get PoI from traces w/ SNR for each share, prepare data for torch DataLoader
    n_points: number of max SNR points to get
    add_sec: collect data for secret
    '''
    dir_name = f"../traces/{d_name}"
    info = get_info_from_log(d_name)
    fname = info["fname"]
    n_shares = info["n_shares"]
    n_files = info["n_files"]
    traces_per_file = info["poly_per_batch"]*info["n_batchs"]
    if n_points is None: #get full trace
        L = np.zeros((traces_per_file*n_files, 195*n_shares))
    elif sec_only:# gen data for secret classification task
        L = np.zeros((traces_per_file*n_files, n_points))
        labels = np.zeros((traces_per_file*n_files, ))
    else:# gen data for classification tasks on shares
        L = np.zeros((traces_per_file*n_files, n_points*n_shares))
    if not sec_only:#labels for shares classification tasks
        labels = np.zeros((traces_per_file*n_files, n_shares))

    for fi in range(n_files):
        i_f = fi*traces_per_file
        if n_points is None:
            f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}.npy"
        elif sec_only:
            traces_trim(d_name, n_points, sec_only=True)
            f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}_seconly.npy"
        else:
            traces_trim(d_name, n_points)
            f_t = f"{dir_name}/{fname}_{n_shares}_{n_points}traces_{fi}.npy"
        f_d = f"{dir_name}/{fname}_{n_shares}_meta_{fi}.npz"
        traces = np.load(f_t).astype(np.int16)
        data = np.load(f_d)["polynoms"]
        L[i_f: i_f+traces_per_file] = traces
        if sec_only:
            labels[i_f: i_f+traces_per_file] = data[range(traces_per_file), -1]
        else:
            labels[i_f: i_f+traces_per_file] = data[range(traces_per_file), :-1]
    return L, labels

def traces_trim(d_name, n_points, sec_only=False):
    dir_name = f"../traces/{d_name}"
    info = get_info_from_log(d_name)
    fname = info["fname"]
    n_shares = info["n_shares"]
    n_files = info["n_files"]
    n_samples = 195*n_shares
    traces_per_file = info["poly_per_batch"]*info["n_batchs"]
    fnew = f"{dir_name}/{fname}_{n_shares}_traces_0_seconly.npy" if sec_only else f"{dir_name}/{fname}_{n_shares}_{n_points}traces_0.npy"
    if os.path.exists(fnew):
        centered_pr("DATA IS READY!")
        pass
    else:
        snr = SNR(3329, n_samples, n_shares+1, use_64bit=True)
        for fi in trange(n_files, desc="SNR|File"):
            f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}.npy"
            f_d = f"{dir_name}/{fname}_{n_shares}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            data = np.load(f_d)["polynoms"]
            share = data[range(traces_per_file), :].astype(np.uint16)
            snr.fit_u(traces, share)
        snr_val = snr.get_snr()
        if sec_only:
            PoI =  np.argsort(snr_val[n_shares])[-n_points:]
        else:
            PoI = np.zeros((n_points*n_shares,), dtype=np.int16)
            for c in range(n_shares):
                idx = np.argsort(snr_val[c])[-n_points:]
                PoI[c*n_points: c*n_points+n_points] = idx
        for fi in trange(n_files, desc="Trimming|FILE"):
            f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}.npy"
            traces = np.load(f_t).astype(np.int16)
            fnew = f"{dir_name}/{fname}_{n_shares}_traces_{fi}_seconly.npy" if sec_only else f"{dir_name}/{fname}_{n_shares}_{n_points}traces_{fi}.npy"
            with open(fnew, "wb") as f:
                np.save(f, traces[:, PoI].copy())
def enc_snr(d_name, n_PoI=10, c_len=195, mode=None, model=None):
    dir_name = f"../traces/{d_name}"
    info = get_info_from_log(d_name)
    fname = info["fname"]
    n_shares = info["n_shares"]
    n_files = info["n_files"]
    n_samples = c_len*n_shares
    m_flag = info["m_flag"]
    flag_desc = ["SUB", "ADD_ALL", "ADD_ONE"]
    traces_per_file = info["poly_per_batch"]*info["n_batchs"]
    if mode=="on_shares":
        if model is None:
            snr = SNR(KYBER_Q, n_samples, n_shares, use_64bit=True)
        else:
            snr = SNR(HW_Q, n_samples, n_shares, use_64bit=True)
    elif mode=="on_sec":
        if model is None:
            snr = SNR(5, n_samples, 1, use_64bit=True)
        else:
            snr = SNR(HW_Q, n_samples, 1, use_64bit=True)
    for fi in trange(n_files, desc="File"):
        f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}.npy"
        f_d = f"{dir_name}/{fname}_{n_shares}_meta_{fi}.npz"
        traces = np.load(f_t).astype(np.int16)
        data = np.load(f_d)["polynoms"]
        if mode=="on_shares":
            share = data[range(traces_per_file), :-1].astype(np.uint16)
        elif mode=="on_sec":
            share = data[range(traces_per_file), -1:].astype(np.uint16)%5
        if model=="HW":
            share = HW(share).astype(np.uint16)
        snr.fit_u(traces, share)
    snr_val = snr.get_snr()
    # idx = np.argsort(snr_val[0])[-50:]
    if mode=="on_sec":
        plt.plot(snr_val.T, label="secret")

    elif mode=="on_shares":
        PoI = []
        for c in range(n_shares):
            idx = np.argsort(snr_val[c])[-n_PoI:]
            PoI.append(idx)
            print(idx, snr_val[c][idx])
            plt.plot(range(n_samples), snr_val[c], label=f"share {c+1}")
    plt.title(f"{flag_desc[m_flag%10]}")
    plt.legend()
    plt.show()
    if mode=="on_shares":
        return PoI





def get_info_from_log(d_name):
    dir_name = f"traces/{d_name}"
    f_log = f"{dir_name}/log"
    info = {}
    keys = ["n_files", "poly_per_batch", "n_batchs", "n_shares", "m_flag", "n_samples", "fname" ]
    with open(f_log, "r") as f:
        f_ls = f.readlines()
    for line in f_ls:
        for key in keys:
            if key in line:
                val = line.rstrip().split("=")[1] if key=="fname" else int( line.rstrip().split("=")[1])
                info[key] = val
                break
    return info


def snr_ntt(d_name, coeff_idx, n_chunks, n_files, model=ID):
    dir_name = f"traces/{d_name}"
    info = get_info_from_log(d_name)
    fname = info["fname"]
    n_shares = info["n_shares"]
    n_files = info["n_files"] if n_files is None else n_files
    # n_samples = trace_len
    n_samples = info["n_samples"]
    traces_per_file = info["poly_per_batch"]*info["n_batchs"]
    coeff_chunks = np.array_split(coeff_idx, n_chunks)
    f_snr = f"SNR_NTT_{d_name}_{coeff_idx}.npy"
    with open(f_snr, "wb") as f:
        np.save(f, coeff_chunks)
        np.save(f, n_samples//4+1)
    for c_chunk in tqdm(coeff_chunks, total=n_chunks, desc="CHUNK|", position=1):
        if model is ID:
            snr = SNR(KYBER_Q, n_samples//4+1, len(c_chunk), use_64bit=True)
        elif model is HW:
            snr = SNR(HW_Q, n_samples//4+1, len(c_chunk), use_64bit=True)
        for fi in trange(n_files, desc="FILE|", leave=False, position=0):
            f_t = f"{dir_name}/{fname}_{n_shares}_traces_{fi}.npy"
            f_d = f"{dir_name}/{fname}_{n_shares}_meta_{fi}.npz"
            traces = np.load(f_t).astype(np.int16)
            traces = traces[:, range(n_samples*7//4, n_samples*2)].copy()
            # print(traces.shape[1],  n_samples//4, len(range(n_samples*7//4, n_samples*2)))
            # traces_holder = np.load(f_t, mmap_mode='r')
            data_holder = np.load(f_d)["polynoms"]
            # for i in trange(0, traces_per_file, 500, desc="Batch|", leave=False):
            #     traces = traces_holder[i:i+500]
            #     data = data_holder[i:i+500]
            coeff = data_holder[:, c_chunk]
            # for t in range(info["n_batchs"]):
            #     data_holder[, c_chunk]
            coeff = model(coeff)
            coeff = coeff.astype(np.uint16)
            snr.fit_u(traces, coeff)
        snr_val = snr.get_snr()
        with open(f_snr, "ab") as f:
            np.save(f, snr_val)
def snr_ntt_readout(d_name, coeff_idx):
    f_snr = f"SNR_NTT_{d_name}_{coeff_idx}.npy"
    with open(f_snr, "rb") as f:
        coeff_chunks = np.load(f)
        n_samples = np.load(f)
        for c_chunk in coeff_chunks:
            snr_chunk = np.load(f)
            snr_min = 1
            snr_max = 0
            for i, c in enumerate(c_chunk):
                snr_min = np.array([snr_chunk[i].min(), snr_min]).min()
                snr_max = np.array([snr_chunk[i].max(), snr_max]).max()
                plt.plot(range(n_samples), snr_chunk[i], label=f"$c_{{{c}}}$")

    # plt.vlines(x=n_samples//2, ymin=snr_min, ymax=snr_max, colors="tab:red")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    # C1 = np.arange(384, 386)
    # C2 = np.arange(510, 512)
    # C = np.hstack((C1, C2))
    # print(C)
    C = np.array([128, 129, 384, 385])
    snr_ntt("060124_2112", C, 1, 5, model=ID)
    snr_ntt_readout("060124_2112", C)
#     with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/log/mlp_onshares_2shares.npy", "rb") as f:
#         print(np.load(f))
#         pi_sub = np.load(f)[2:]
#         pi_add = np.load(f)[2:]
#         plt.plot(range(len(pi_sub)), pi_sub, color="tab:blue", linestyle="solid", label="2 share SUB")
#         plt.plot(range(len(pi_add)), pi_add, color="tab:blue", linestyle="dashed", label="2 share ADD")
#     with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/log/mlp_onshares_3shares.npy", "rb") as f:
#         print(np.load(f))
#         pi_sub = np.load(f)[2:]
#         pi_add = np.load(f)[2:]
#         plt.plot(range(len(pi_sub)), pi_sub, color="tab:orange", linestyle="solid", label="3 share SUB")
#         plt.plot(range(len(pi_add)), pi_add, color="tab:orange", linestyle="dashed", label="3 share ADD")
#
#     plt.legend(fontsize=12)
#     plt.title("MLP_VALIDATION PI")
#     plt.show()
    # print(HW(KYBER_Q))
    # x = np.array([5, 6])
    # print(np.power(x, 2))
    # ft = "/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/traces/140623-21/NTT_2_traces_0.npy"
    # traces = np.load(ft)
    # print(traces.shape)
    # plt.plot(traces.T)
    # plt.show()
    # get_info_from_log("140623-22")
    # snr_ntt("140623-22")
    # traces_trim("190623_1956", 195, 10)
    # enc_snr("270623_1402",c_len=195, mode="on_sec", model=None)



    # colors = ["tab:blue", "tab:orange"]
    # modes = ["SUB", "ADD"]
    # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/mlp_onshares_2shares.npy", "rb") as f:
    #     farr = np.load(f)%10
    #     for i in farr:
    #         pi = np.load(f)
    #         print(modes[i], pi)
    #         plt.plot(range(len(pi)), pi, label=f"{modes[i]}_2 shares")
    # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/mlp_on_sec_3shares.npy", "rb") as f:
    #     farr = np.load(f)%10
    #     for i in farr:
    #         pi = np.load(f)[3:]
    #         print(pi)
    #         plt.plot(range(len(pi)), pi, label=f"{modes[i]}_on sec", linestyle="-", color=colors[i])
    # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/mlp_wholetrace_3shares_.npy", "rb") as f:
    #     farr = np.load(f)%10
    #     for i in farr:
    #         pi = np.load(f)[3:]
    #         print(pi)
    #         plt.plot(range(len(pi)), pi, label=f"{modes[i]}_full trace", linestyle="-.", color=colors[i])
    # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/mlp_onshares_3shares.npy", "rb") as f:
    #     farr = np.load(f)%10
    #     for i in farr:
    #         pi = np.load(f)[3:]
    #         print(modes[i], pi)
    #         plt.plot(range(len(pi)), pi, label=f"{modes[i]}_on shares", linestyle=":", color=colors[i])
    # # with open("/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/postpro/mlp_onsec_2shares.npy", "rb") as f:
    # #     farr = np.load(f)%10
    # #     for i in farr:
    # #         pi = np.load(f)
    # #         plt.plot(range(len(pi)), pi, label=f"{modes[i]}_2 shares")
    # plt.title("Validation PI 3 SHARES")
    # plt.legend()
    # plt.show()
