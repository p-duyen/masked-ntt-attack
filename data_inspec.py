import numpy as np
from helpers import*
from tqdm.auto import trange, tqdm
from scalib.metrics import SNR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def nonrand_traces(dname):
    lh = Leakage_Handler(dname)
    L = {}
    D = {}
    c_0_spl = np.arange(lh.n_samples//4)
    poi_c0 = 1648
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
    # for i in range(10):
    #     plt.plot(np.arange(lh.n_samples), L["share0"][i])
    # plt.show()
    # for i in range(10):
    #     plt.plot(np.arange(lh.n_samples), L["share1"][i])
    # plt.show()

    # C_0_128 = D["share0"][:, [128]]
    C_0_0 = D["share0"][:, 0]
    print(C_0_0.shape, C_0_0)
    C_0_0_u, idx_u = np.unique(C_0_0, return_index=True)
    print(C_0_0_u, idx_u)

    # print(C_0_0.shape)
    # print(C_0_128.tolist())
    # snr = SNR(KYBER_Q, len(c_0_spl), 1, use_64bit=True)
    # snr.fit_u(L["share0"], C_0_0)
    # snr_val = snr.get_snr()
    # poi_128_0 = np.argsort(snr_val[0])[::-1][:10]
    # print(poi_128_0)
    # plt.plot(snr_val.T)
    # plt.show()
    for i in C_0_0_u:
        # print(C_0_0_u[i], idx_u[i])
        idx_i = np.nonzero(C_0_0==i)[0]
        # print(len(idx_i), idx_i)

        L0i = L["share0"][idx_i]
        # print(L0i)
        if i < 100:
            plt.plot(c_0_spl, L0i.mean(axis=0), label=f"c={i}")

    plt.legend(ncol=16)
    plt.savefig("log.png")
    # plt.show()

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

def extra_f():
    n_profiling = 50000
    L1 = share_check("240124_1424")
    L2 = share_check("240124_1502")
    print(L1.shape, L2.shape)
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

    Y0 = L_transformed[labels_profiling==0]
    Y1 = L_transformed[labels_profiling==1]

    p0, x0 = np.histogram(Y0, bins=100, density=True)
    plt.plot(x0[:-1], p0, label="class 1 LDA projection", alpha=0.75, linewidth=2)
    p1, x1 = np.histogram(Y1, bins=100, density=True)
    plt.plot(x1[:-1], p1, label="class 2 LDA projection", alpha=0.75, linewidth=2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    extra_f()
