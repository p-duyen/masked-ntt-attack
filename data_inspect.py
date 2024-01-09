import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
import pickle
from scalib.modeling import LDAClassifier as lda
from scalib.metrics import SNR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from helpers import *
# from discriminant_analysis_ import LinearDiscriminantAnalysis as lda
from scipy import linalg
import math
from LDA_func import mean_vec



D_2 = [ "021123_1335", "021123_1506"]
D_3 = ["221123_2119", "221123_2328"]
D_4 = ["231123_0138", "231123_0431"]
# D_5 = ["171123_1644", "171123_1727"]
def check_data_dict(data, args):

    tmp = np.zeros((args.poly_per_file, KYBER_N))
    for i in range(args.n_shares):
        Xi = data[f"share_{i}"].copy()
        tmp = (tmp+Xi)%KYBER_Q
    sec = data["secret"]
    return np.all(tmp==sec)
def meta_data_sanity_check(dname):
    lh = Leakage_Handler(dname)
    secret = lh.get_secrets()
    print("SEC")
    print(secret.shape)
    for fi in range(lh.n_files):

        print_centered(f"========={dname} {lh.n_shares} SHARES FILE {fi}=============")
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        data = np.load(f_d, allow_pickle=True)
        data = data.item()
        print_centered(f"{check_data_dict(data, lh)}")
        sec = data.get("secret")
        print_centered(f"{np.all(sec==secret)}")

def traces_sanity_check(dname, share_i, f_start, f_end):
    lh = Leakage_Handler(dname)
    print_centered(f"====INSPECTING {dname} share {share_i} in {lh.n_shares} SHARES {f_start} {f_end}====")
    plt.figure(figsize=(12, 12))
    c_list = np.array([128, 130])
    subtrace = np.arange(200, 5000)
    snr = SNR(KYBER_Q, len(subtrace), len(c_list))
    # snr = SNR(12, len(subtrace), len(c_list))
    for fi in trange(f_start, f_end, desc="FILE|"):
        f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        traces = np.load(f_t).astype(np.int16)
        traces = traces[:, subtrace].copy()
        data = np.load(f_d, allow_pickle=True).item()
        data = data.get(f"share_{share_i}")[:, c_list].astype(np.uint16)
        snr.fit_u(traces, data)
        if fi==f_start:
            data_holder = data.copy()
            traces_holder = traces.copy()
        else:
            data_holder = np.append(data_holder, data.copy(), axis=0)
            traces_holder = np.append(traces_holder, traces.copy(), axis=0)
    print(data_holder.shape)
    snr_val = snr.get_snr()
    n_pois = 2
    pois = np.zeros((len(c_list), n_pois), dtype=np.uint32)
    for i, c in enumerate(c_list):
        pois[i] = np.argsort(snr_val[i])[::-1][:n_pois]
        print(snr_val[i].shape, pois[i], pois[i].dtype)
        plt.plot(subtrace, snr_val[i])
        plt.scatter(subtrace[pois[i]], snr_val[i][pois[i]])
    noise = []
    for i, c in enumerate(c_list):
        mean_traces = np.zeros((KYBER_Q, n_pois))
        var_traces = np.zeros((KYBER_Q, n_pois))
        for q in range(KYBER_Q):
            q_pos = np.where(data_holder[:, i]==q)[0]
            trace_q = traces_holder[q_pos].copy()
            trace_q = trace_q[:, pois[i]]
            mean_traces[q] = trace_q.mean(axis=0)
            var_traces[q] = np.var(trace_q, axis=0)
        noise_c = var_traces.mean(axis=0)
        print(noise_c.shape)
        noise.append(noise_c)
    print(np.corrcoef(noise))
    print(np.cov(noise))

    #
    plt.legend()
    # plt.savefig(f"inspect_{dname}_SHARE_{share_i}_F{f_start}_{f_end}.png")
    plt.show()
def trace_snr(lh, n_pois, c_list, subtrace):
    snr_vec = []
    for share_i in range(lh.n_shares):
        snr_vec.append(SNR(KYBER_Q, len(subtrace), len(c_list)))
    for fi in trange(lh.n_files//2, desc="SNR FILE|"):
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        data = np.load(f_d, allow_pickle=True).item()
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
            traces = np.load(f_t).astype(np.int16)
            traces = traces[:, subtrace].copy()
            share = data.get(f"share_{share_i}")[:, c_list].astype(np.uint16)
            snr_vec[share_i].fit_u(traces, share)
    pois = np.zeros((lh.n_shares, len(c_list), n_pois), dtype=np.uint32)
    fig, axs = plt.subplots(lh.n_shares, figsize=(12, 12))
    for share_i in range(lh.n_shares):
        snr_val = snr_vec[share_i].get_snr()
        for ci, c in enumerate(c_list):
            snr_c = snr_val[ci]
            _ = np.argsort(snr_c)[-n_pois:]
            pois[share_i, ci] = subtrace[_]
            axs[share_i].plot(subtrace, snr_val[ci], label=f"share{share_i} c_{c}")
            axs[share_i].scatter(subtrace[pois[share_i, ci]], snr_val[ci, _])
            axs[share_i].set_title(f"SHARE {share_i+1}")
    plt.legend()
    plt.show()
    return pois
def noise_signal_correlation(dname):
    lh = Leakage_Handler(dname)
    secret = lh.get_secrets()
    c_list = np.array([128])
    subtrace = np.arange(5000)
    n_pois = 1
    # POIS = trace_snr(lh, n_pois, c_list, subtrace)
    DATA = {}
    TRACES = {}
    # lh.n_files = 50
    for fi in trange(lh.n_files, desc="GETDATA FILE|"):
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        data = np.load(f_d, allow_pickle=True).item()
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
            traces = np.load(f_t).astype(np.int16)
            # pois = np.ravel(POIS[share_i])
            # print(pois)
            traces = traces[:, 1234].copy()
            share = data.get(f"share_{share_i}")[:, c_list].astype(np.uint16)
            if fi==0:
                DATA[f"share_{share_i}"] = share.copy()
                TRACES[f"share_{share_i}"] = traces.copy()
            else:
                DATA[f"share_{share_i}"] = np.append(DATA[f"share_{share_i}"], share.copy(), axis=0)
                TRACES[f"share_{share_i}"] = np.append(TRACES[f"share_{share_i}"], traces.copy(), axis=0)
    NOISE = []
    SIGNAL = []
    M = []
    for share_i in range(lh.n_shares):
        print(f"=================={share_i}===================")
        share = DATA[f"share_{share_i}"]
        trace_share = TRACES[f"share_{share_i}"]
        mean_traces = np.zeros((KYBER_Q, n_pois*len(c_list)))
        var_traces = np.zeros((KYBER_Q, n_pois*len(c_list)))
        for q in range(KYBER_Q):
            q_pos = np.where(share==q)[0]
            trace_q = trace_share[q_pos]
            mean_traces[q] = trace_q.mean(axis=0)
            var_traces[q] = np.var(trace_q, axis=0)
            # print(mean_traces)
        _ = mean_traces.squeeze()
        print(_.shape)
        M.append(mean_traces.squeeze())
    for q in range(KYBER_Q):
        mus = [M[i][q] for i in range(lh.n_shares)]
        print(q, HW(q), mus, np.abs(mus[0]-mus[1]))
        # exit()
    #     noise_c = var_traces.mean(axis=0)
    #     # print(noise_c)
    #     signal_c = np.var(mean_traces, axis=0)
    #     # print(signal_c)
    #     snr = signal_c/noise_c
    #     plt.scatter(np.ravel(POIS[share_i]), snr, label=f"share {share_i}")
    #     NOISE.append(var_traces.squeeze())
    #     SIGNAL.append(mean_traces.squeeze())
    #
    #
    # print(c_list)
    # print("======NOISE======")
    # print(np.cov(NOISE))
    # print(np.corrcoef(NOISE))
    # print("======SIGNAL======")
    # print(np.cov(SIGNAL))
    # print(np.corrcoef(SIGNAL))
    # print("======COV======")
    # print(np.cov(COV))
    # print(np.corrcoef(COV))
    # plt.legend()
    # plt.show()

# def TTEST(d1, d2):
def get_data(lh):
    DATA = {}
    TRACES = {}
    for fi in trange(lh.n_files, desc="GETDATA FILE|"):
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        data = np.load(f_d, allow_pickle=True).item()
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
            traces = np.load(f_t).astype(np.int16)
            traces = traces[:, lh.pois[share_i]].copy()
            share = data.get(f"share_{share_i}")[:, right_wing].astype(np.uint16)
            if fi==0:
                DATA[f"share_{share_i}"] = share.copy()
                TRACES[f"share_{share_i}"] = traces.copy()
            else:
                DATA[f"share_{share_i}"] = np.append(DATA[f"share_{share_i}"], share.copy(), axis=0)
                TRACES[f"share_{share_i}"] = np.append(TRACES[f"share_{share_i}"], traces.copy(), axis=0)
    return DATA, TRACES
def combined_L_sanity(dname, combif):
    lh = Leakage_Handler(dname)
    secret = lh.get_secrets()
    n_pois = 1
    right_wing = np.arange(KYBER_N_2, KYBER_N)
    coeff_chunks = np.split(right_wing, 16)
    lh.pois = np.zeros((lh.n_shares, KYBER_N_2))
    for coeff_list in tqdm(coeff_chunks, desc="GET SNR CC|"):
        smp_list = coeff_list%128
        subtrace = np.arange(smp_list[0]*1200, smp_list[0]*1200+len(coeff_list)*1500)
        POIS = trace_snr(lh, n_pois, coeff_list, subtrace)
        for share_i in range(lh.n_shares):
            lh.pois[share_i, smp_list] = np.ravel(POIS[share_i])

    DATA, TRACES = get_data(lh)
    if combif in ["abs_diff", "sum"]:
        combined_L = np.zeros((total_N, n_coeff))
    elif combif in ["prod", "norm_prod"]:
        combined_L = np.ones((total_N, n_coeff))




def meta_check():
    meta_data_sanity_check(D_2[0])
    meta_data_sanity_check(D_2[1])
    meta_data_sanity_check(D_3[0])
    meta_data_sanity_check(D_3[1])
    meta_data_sanity_check(D_4[0])
    meta_data_sanity_check(D_4[1])
def traces_check():
    # for i in range(0, 50, 25):

    # traces_sanity_check(D_2[0], 0, f_start=0, f_end=100)
    # # traces_sanity_check(D_2[0], 1, f_start=0, f_end=50)
    # # traces_sanity_check(D_2[1])
    # traces_sanity_check(D_3[0], 0, f_start=0, f_end=50)
    # traces_sanity_check(D_3[1], 1, f_start=0, f_end=50)
    # traces_sanity_check(D_3[1], 2, f_start=0, f_end=50)
    # traces_sanity_check(D_4[0])
    # traces_sanity_check(D_4[1])
    # noise_signal_correlation(D_2[0])
    noise_signal_correlation(D_3[0])

if __name__ == '__main__':
    traces_check()
    # data_sanity_check("221123_2102")


    # data_sanity_check(D_5[0])
    # data_sanity_check(D_5[1])
