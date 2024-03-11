import numpy as np
from tqdm.auto import trange, tqdm
from scalib.metrics import SNR, Ttest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
from helpers import*
from LDA_func import *

def nonrand_traces(dname, ddes=0):
    lh = Leakage_Handler(dname)
    L = {}
    D = {}
    c_0_spl = np.arange(lh.n_samples-4000, lh.n_samples)
    poi_c0 = 1648
    if os.path.exists(f"inspect_trace_{dname}.npy"):
        print("DATA AVAIL!")
        L_ = np.load(f"inspect_trace_{dname}.npy", allow_pickle=True)
        D_ = np.load(f"inspect_data_{dname}.npy", allow_pickle=True)
        for sharei in range(lh.n_shares):
            L[f"share{sharei}"]=L_.item().get(f"share{sharei}")
            D[f"share{sharei}"]=D_.item().get(f"share{sharei}")
    else:

        for fi in trange(lh.n_files):
            f_d = f"{lh.file_path}_meta_{fi}.npy"
            for share_i in range(lh.n_shares):
                f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"

                traces_i = np.load(f_t).astype(np.int16)
                data = np.load(f_d, allow_pickle=True)
                data = data.item().get(f"share_{share_i}").astype(np.uint16)
                c_255 = data[:, [255]]
            #     # print(c_255.tolist())
                t_255 = traces_i[:, c_0_spl].copy()
                if fi==0:
                    L[f"share{share_i}"] = t_255.copy()
                    D[f"share{share_i}"] = c_255.copy()
                else:
                    L[f"share{share_i}"] = np.append(L[f"share{share_i}"], t_255, axis=0)
                    D[f"share{share_i}"] = np.append(D[f"share{share_i}"], c_255, axis=0)

        np.save(f"inspect_trace_{dname}.npy", L, allow_pickle=True)
        np.save(f"inspect_data_{dname}.npy", D, allow_pickle=True)
    snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    # print(L.dtype, L)
    snr.fit_u(L["share0"], D["share0"])
    snr_val = snr.get_snr()
    poi = np.argsort(snr_val[0])[::-1][:10]
    poi.sort()
    plt.text(c_0_spl[0]-200, 5, f"poi cls{ddes} share1 {poi}", size=18,
         ha="left", va="top",
         bbox=dict(boxstyle="round",
                   ec=(1, 0.5, 0.5),
                   fc=(1, 0.8, 0.8),
                   )
         )
    L0 = L["share0"][:, poi].copy()
    print("share0", poi)
    # print(np.sort(poi))
    plt.plot(c_0_spl, snr_val[0], label=f"share 0 class {ddes}")
    plt.scatter(c_0_spl[poi], snr_val[0][poi])



    #share 2 proc
    snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    snr.fit_u(L["share1"], D["share1"])
    snr_val = snr.get_snr()
    plt.plot(c_0_spl, snr_val[0], label=f"share 1 class {ddes}")
    plt.scatter(c_0_spl[poi], snr_val[0][poi])

    poi = np.argsort(snr_val[0])[::-1][:4]

    # poi.sort()
    plt.text(c_0_spl[0]-200, 4,  f"poi cls{ddes} share2 {poi}", size=18,
         ha="left", va="top",
         bbox=dict(boxstyle="round",
                   ec=(1, 0.5, 0.5),
                   fc=(1, 0.8, 0.8),
                   )
         )
    # print("share1", poi)
    L1 = L["share1"][:, poi].copy()
    # print(np.sort(poi))
    #
    # for i in range(KYBER_Q):
    #     idx_i_0 = np.nonzero(D["share0"]==i)[0]
    #     Li_0 = L["share0"][idx_i_0]
    #     idx_i_1 = np.nonzero(D["share1"]==i)[0]
    #     Li_1 = L["share1"][idx_i_1]
    #     plt.plot(c_0_spl, Li_0.mean(axis=0), label=f"c={i} share0")
    #     plt.plot(c_0_spl, Li_1.mean(axis=0), label=f"c={i} share1")
    #     plt.legend()
    #     plt.show()
    #     # plt.legend()
    #     plt.plot(range(len(c_0_spl)), snr_val[0]*2000, color="red", alpha=0.5)
    #     plt.show()
        # print(i, len(idx_i), end="||")
        # if i in [0, 1, 2, 31, 63, 511, 1023]:
        #     plt.plot(c_0_spl, Li.mean(axis=0), label=f"c={i}", color=CSS_COLORS[i%50+10], alpha=0.4)
    # plt.plot(c_0_spl, common_mean, color="red")

    # print(c_0_spl[poi])
    plt.legend()
    plt.show()
    return L0



def share_check(dname):
    lh = Leakage_Handler(dname)
    L = {}
    D = {}
    lh.n_files=50
    # c_0_spl = np.arange(lh.n_samples-2000, lh.n_samples)
    # c_0_spl = np.arange(2000, 4000)
    c_0_spl = np.arange(lh.n_samples)
    for fi in trange(lh.n_files):
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
            traces_i = np.load(f_t)
            data = np.load(f_d, allow_pickle=True)
            data = data.item().get(f"share_{share_i}").astype(np.uint16)
            if fi==0:
                L[f"share{share_i}"] = traces_i[:, c_0_spl].copy()
                D[f"share{share_i}"] = data.copy()
            else:
                L[f"share{share_i}"] = np.append(L[f"share{share_i}"], traces_i[:, c_0_spl].copy(), axis=0)
                D[f"share{share_i}"] = np.append(D[f"share{share_i}"], data.copy(), axis=0)
    C_0_0 = D["share0"][:, 0]
    C_1_0 = D["share1"][:, 0]
    C_0_128 = D["share0"][:, [1]]
    C_1_128 = D["share1"][:, [1]]
    C_0_129 = D["share0"][:, [129]]
    C_1_129 = D["share1"][:, [129]]
    print((C_1_128+C_0_128)%KYBER_Q)

    snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    snr.fit_u(L["share0"], C_0_128)
    snr_val = snr.get_snr()
    poi_128_0 = np.argsort(snr_val[0])[::-1][:10]
    print(poi_128_0)
    plt.plot(snr_val.T)
    plt.show()
    snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    snr.fit_u(L["share1"], C_1_128)
    snr_val = snr.get_snr()
    poi_128_1 = np.argsort(snr_val[0])[::-1][:10]
    print(poi_128_1)
    plt.plot(snr_val.T)
    plt.show()
    # snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    # snr.fit_u(L["share0"], C_0_129)
    # snr_val = snr.get_snr()
    # poi_129_0 = np.argsort(snr_val)[::-1][0]
    # plt.plot(snr_val.T)
    # plt.show()
    # snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    # snr.fit_u(L["share1"], C_1_129)
    # snr_val = snr.get_snr()
    # poi_129_1 = np.argsort(snr_val)[::-1][0]
    # plt.plot(snr_val.T)
    # plt.show()
    L0 = L["share0"][:, poi_128_0]
    L1 = L["share1"][:, poi_128_1]
    Lc = L0+L1
    # p0, x0 = np.histogram(L0, bins=100, density=True)
    # plt.plot(x0[:-1], p0, label=f"share1 c_128 {dname}", alpha=0.75, linewidth=2)
    # p1, x1 = np.histogram(L1, bins=100, density=True)
    # plt.plot(x1[:-1], p1, label=f"share2 c_128 {dname}", alpha=0.75, linewidth=2)
    # plt.legend()
    # pc, xc = np.histogram(Lc, bins=100, density=True)
    # plt.plot(xc[:-1], pc, label=f"combined c_128 {dname}", alpha=0.75, linewidth=2)
    # L0 = L["share0"][:, poi_129_0]
    # L1 = L["share1"][:, poi_129_1]
    # Lc = L0+L1
    # p0, x0 = np.histogram(L0, bins=100, density=True)
    # plt.plot(x0[:-1], p0, label="share1 c_129", alpha=0.75, linewidth=2)
    # p1, x1 = np.histogram(L1, bins=100, density=True)
    # plt.plot(x1[:-1], p1, label="share2 c_129", alpha=0.75, linewidth=2)
    # plt.legend()
    # pc, xc = np.histogram(Lc, bins=100, density=True)
    # plt.plot(xc[:-1], pc, label="combined c_129", alpha=0.75, linewidth=2)
    # plt.legend()
    # plt.show()
    # traces_idx = np.nonzero(C_1_0==c)[0]
    # print(traces_idx)
    # L1 = L["share1"][traces_idx]
    # for i in range(len(traces_idx)):
    # #     c0 = D["share1"][i][0]
    #     plt.plot(c_0_spl, L1[i])
    # # plt.legend()
    # plt.plot(c_0_spl, L1.mean(axis=0), color="red")
    # plt.title("ON SHARE 2")
    # plt.show()
    return Lc

def extra_f(L1, L2, n_profiling):

    labels_profiling = np.zeros(n_profiling, dtype=np.uint16)
    idx_L1 = np.random.choice(len(L1), n_profiling//2)
    idx_L2 = np.random.choice(len(L2), n_profiling//2)
    L_profiling = L1[idx_L1].copy()
    L_profiling = np.append(L_profiling, L2[idx_L2].copy(), axis=0)

    labels_profiling[range(n_profiling//2, n_profiling)] = 1
    shuffle_idx = np.arange(n_profiling)
    np.random.shuffle(shuffle_idx)
    L_profiling = L_profiling[shuffle_idx]
    labels_profiling = labels_profiling[shuffle_idx]
    print_centered(f"==============profiling shape: {L_profiling.shape} {labels_profiling.shape}==============")
    classifer = LDA(solver="eigen", n_components=1)
    L_transformed = classifer.fit_transform(L_profiling, labels_profiling)
    print(classifer.coef_)

    Y0 = L_transformed[labels_profiling==0]
    Y1 = L_transformed[labels_profiling==1]

    p0, x0 = np.histogram(Y0, bins=100, density=True)
    plt.plot(x0[:-1], p0, label="class 1 LDA projection", alpha=0.75, linewidth=2)
    p1, x1 = np.histogram(Y1, bins=100, density=True)
    plt.plot(x1[:-1], p1, label="class 2 LDA projection", alpha=0.75, linewidth=2)

    # Sb = scatter_between(L_profiling, labels_profiling)
    # Sw = scatter_within(L_profiling, labels_profiling)
    # mus = mean_vec(L_profiling, labels_profiling)
    # inv_Sw = np.linalg.inv(Sw)
    # V = project_vector(Sw, Sb)
    # Y = L_profiling.dot(V)
    # Y0 = Y[labels_profiling==0]
    # Y1 = Y[labels_profiling==1]
    # print(len(np.nonzero(labels_profiling==0)[0]), len(np.nonzero(labels_profiling==1)[0]))
    # print("project direction", V)
    # p0, x0 = np.histogram(Y0, bins=100, density=True)
    # plt.plot(x0[:-1], p0, label="class 1 LDA projection", alpha=0.75, linewidth=2)
    # p1, x1 = np.histogram(Y1, bins=100, density=True)
    # plt.plot(x1[:-1], p1, label="class 2 LDA projection", alpha=0.75, linewidth=2)
    #
    # N, leg = L1.shape
    # #====T TEST=============
    # ttest = Ttest(leg, d=3)
    # ttest.fit_u(L_profiling, labels_profiling)
    # t = ttest.get_ttest()
    # for i in range(3):
    #     plt.plot(np.arange(leg), t[i])
    plt.legend()
    plt.show()
def POI_check(dname):
    lh = Leakage_Handler(dname)
    share_i = 0
    coeff_list = np.arange(256)
    n_chunks = 16
    lh.get_snr_on_share(share_i, coeff_list, n_chunks, f_des=None, add_noise=False, model=ID)
    lh.get_PoI_on_share(share_i, coeff_list, n_chunks, n_poi, f_des=None, add_noise=False, model=ID, display=False, keepsnr=False)

if __name__ == '__main__':
    dname = "021123_1335"
    POI_check(dname)
    # Lcls0 = nonrand_traces("250124_1734", 1)
    # Lcls1 = nonrand_traces("250124_1643", 2)
    # L0m = Lcls0.mean(axis=0)
    # L1m =  Lcls1.mean(axis=0)
    # print(L0m.tolist())
    # print(L1m.tolist())
    # print((L0m-L1m).tolist())
    # # N, leg = Lcls0.shape
    # extra_f(Lcls0, Lcls1, n_profiling=50000)
