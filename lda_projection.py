import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from scalib.modeling import LDAClassifier as lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import linalg
import math
from helpers import *
from LDA_func import *




def share_dist_inspect(d_name, c, share_i, n_coeff=KYBER_N_2):
    lh = Leakage_Handler(d_name)
    n_files_og = lh.n_files
    total_N = lh.poly_per_file*lh.n_files
    right_wing = np.arange(KYBER_N_2, KYBER_N)
    lh.n_files = 10
    n_chunks = 16
    lh.get_snr_on_share(share_i, coeff_list=right_wing, n_chunks=n_chunks, f_des=f"{right_wing[0]}_{right_wing[-1]}", add_noise=0)
    lh.get_PoI_on_share(share_i, right_wing, n_chunks, n_poi=1, f_des=f"{right_wing[0]}_{right_wing[-1]}", add_noise=0, display=False)
    for fi in trange(lh.n_files, desc="GETDATA-FILE|"):
        f_d = f"{lh.file_path}_meta_{fi}.npy"
        data = np.load(f_d, allow_pickle=True)
        data = data.item().get(f"share_{share_i}")[:, right_wing]
        pt = np.where(data==c)
        f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
        traces = np.load(f_t)
        poi =  lh.poi[f"share_{share_i}"]
        poi = poi.astype(np.uint32)
        poi_c = poi[pt[1]].squeeze()
        # selected = np.column_stack((pt[0], poi_c))

        # print(selected.shape)
        # print(selected)
        Lc = traces[pt[0], poi_c]
        if fi==0:
            L = Lc.copy()
        else:
            L = np.append(L, Lc)

    print(L.shape, L.mean())
    p0, x0 = np.histogram(L, bins=100, density=True)
    plt.plot(x0[:-1], p0, label=f"class0 s={c}", alpha=0.75)
    plt.legend()
    plt.show()

def combine_leakage(d_name, combif, n_coeff=KYBER_N_2, wing="right", f_decs=None, add_noise=False, model=ID):
    m_desc="HW" if model is HW else "ID"
    f_n = f"log/combined_L_{d_name}_{combif}_{n_coeff}_{f_decs}_noise_{add_noise}_{m_desc}.npy"
    lh = Leakage_Handler(d_name)
    if os.path.exists(f"{f_n}"):
        print_centered(f"COMBINED LEAKAGE FOR {d_name} {lh.n_shares} {combif} {n_coeff} COEFFS {wing} wing(s) noise {add_noise} {m_desc} EXISTED")
        with open(f_n, "rb") as f:
            combined_L = np.load(f)
        return combined_L
    print_centered(f"GENERATE COMBINED LEAKAGE FOR {d_name} {lh.n_shares} {combif} {n_coeff} COEFFS {wing} wing(s) noise {add_noise} {m_desc}")
    n_files_og = lh.n_files
    total_N = lh.poly_per_file*n_files_og
    print( lh.poly_per_file, total_N)

    n_chunks = 16
    if wing=="left":
        coeff_list = np.arange(0, KYBER_N_2)
    elif wing=="right":
        coeff_list = np.arange(KYBER_N_2, KYBER_N)
    elif wing=="both":
        coeff_list = np.arange(KYBER_N)
        n_coeff = 256
        n_chunks = 32

    if combif=="norm_prod":
        temp = np.zeros((lh.n_shares, total_N, n_coeff))
    if combif in ["abs_diff", "sum"]:
        combined_L = np.zeros((total_N, n_coeff))
    elif combif in ["prod", "norm_prod"]:
        combined_L = np.ones((total_N, n_coeff))

    lh.n_files = 10 if wing=="right" else 50

    m_decs = "HW" if model is HW else "ID"
    lh.n_files = n_files_og if model is HW else lh.n_files
    for i in range(lh.n_shares):
        print_centered(f"====SNR SHARE {i+1}====")
        lh.get_snr_on_share(i, coeff_list=coeff_list, n_chunks=n_chunks, f_des=f"{coeff_list[0]}_{coeff_list[-1]}", add_noise=0, model=model)
        lh.get_PoI_on_share(i, coeff_list, n_chunks, n_poi=1, f_des=f"{coeff_list[0]}_{coeff_list[-1]}", add_noise=0, display=False, model=model)

    lh.n_files = 50

    for fi in trange(lh.n_files, desc="GETDATA-FILE|"):
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"

            traces = np.load(f_t)
            poi =  lh.poi[f"share_{share_i}"]
            poi = poi.astype(np.uint32)
            Li = traces[:, poi].copy()
            Li = Li[:, range(n_coeff)].squeeze()
            if add_noise:
                Li = Li + np.random.normal(0, add_noise, size=Li.shape)
                # Li = Li.astype(np.int16)
            # print(lh.poly_per_file*fi, lh.poly_per_file*fi+lh.poly_per_file)
            if combif=="abs_diff":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] =  Li.copy() - combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]
            elif combif=="sum":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] =  Li.copy() + combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]
            elif combif=="prod":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] = combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]*Li.copy()
            elif combif=="norm_prod":
                temp[share_i, np.arange(lh.poly_per_file*fi, lh.poly_per_file*fi+lh.poly_per_file)] = Li.copy()


    if combif=="norm_prod":
        for ni in range(lh.n_shares):
            Li= temp[ni].copy()
            print("norm_prod process", Li.shape, (Li.mean(axis=0)).shape)
            Li = Li -  Li.mean(axis=0)
            print(Li.mean(axis=0))
            # print( Li.mean(axis=0))
            # print(Li.shape, combined_L.shape)
            combined_L = combined_L*Li


    if combif=="abs_diff":
        with open(f_n, "wb") as f:
            np.save(f, np.abs(combined_L))
        return np.abs(combined_L)
    with open(f_n, "wb") as f:
        np.save(f, combined_L)
    return combined_L

def run_onM(n_profiling, noise, n_shares, combif, n_coeff=128, model=ID):
    m_decs = "ID" if model is ID else "HW"
    wing = "left"
    # 2 shares

    # 3 shares
    if n_shares ==2:
        d1 = "021123_1335"
        d2 = "021123_1506"
    elif n_shares==3:
        d1 = "221123_2119"
        d2 = "221123_2328"

    elif n_shares==4:
        # 4 shares
        d1 = "231123_0138"
        d2 = "231123_0431"
    elif n_shares==5:
        d1 = "231123_1239"
        d2 = "231123_0903"
    elif n_shares==6:
        d1 = "060124_2112"
        d2 = "070124_1718"


    L1 = combine_leakage(d1, combif, n_coeff=128, wing=wing, f_decs=f"{wing}wing", add_noise=noise, model=model)
    # print("combined L")
    # print(L1.mean(axis=0))
    L2 = combine_leakage(d2, combif, n_coeff=128, wing=wing, f_decs=f"{wing}wing", add_noise=noise, model=model)
    # print(L2.mean(axis=0))
    print(L1.shape, L2.shape)

    L1 = L1[:, range(n_coeff)]
    L2 = L2[:, range(n_coeff)]



    lh1 = Leakage_Handler(d1)
    S1 = lh1.get_secrets()
    # S1 = S1[:KYBER_N_2]
    S1 = S1[KYBER_N_2:].astype(np.uint16)
    lh2 = Leakage_Handler(d2)
    S2 = lh2.get_secrets()
    # S2 = S2[:KYBER_N_2]
    S2 = S2[KYBER_N_2:].astype(np.uint16)
    # print(S1.shape, S2.shape)
    # L1 = L1.astype(np.int16)
    # L2 = L2.astype(np.int16)
    # lh = Leakage_Handler("021123_1335")
    # lh = Leakage_Handler("131123_1401")
    # S1 = lh.get_secrets()[KYBER_N_2:]
    # lh = Leakage_Handler("021123_1506")
    # lh = Leakage_Handler("141123_1246")
    # S2 = lh.get_secrets()[KYBER_N_2:]

    plt.figure(figsize=(18, 12))
    total_N = len(L1)
    # print(total_N)

    # L_profiling = np.zeros((n_profiling, n_coeff))
    labels_profiling = np.zeros(n_profiling, dtype=np.uint16)
    idx_L1 = np.random.choice(len(L1), n_profiling//2)
    idx_L2 = np.random.choice(len(L2), n_profiling//2)
    L_profiling = L1[idx_L1].copy()
    print(L_profiling.shape)
    L_profiling = np.append(L_profiling, L2[idx_L2].copy(), axis=0)
    print(L_profiling.shape)

    labels_profiling[range(n_profiling//2, n_profiling)] = 1

    same_c = np.nonzero(S1==S2)[0]
    # print(S1[same_c])
    # print(S2[same_c])
    # print(len(same_c))
    # L_profiling = L_profiling[:, same_c].copy()




    # L_p = np.zeros((n_profiling, L1.shape[1]), dtype=np.int16)
    # S_p = np.zeros((n_profiling), dtype=np.uint16)
    # idx_p1 = np.random.choice(n_profiling, n_profiling//2, replace=False)
    # idx_L1 = np.random.choice(total_N, n_profiling//2, replace=False)
    # L_p[idx_p1] = L1[idx_L1].copy()
    # idx_p2 = np.setdiff1d(np.arange(n_profiling), idx_p1)
    # idx_L2 = np.random.choice(total_N, n_profiling//2, replace=False)
    # L_p[idx_p2] = L2[idx_L2].copy()
    # S_p[idx_p2] = 1

    # cls = lda(2, 1, n_coeff)
    # cls.fit_u(L_profiling, labels_profiling)
    # cls.solve()
    # Sb = cls.get_sb()
    # Sw = cls.get_sw()
    # mus = cls.get_mus()

    Sb = scatter_between(L_profiling, labels_profiling)
    Sw = scatter_within(L_profiling, labels_profiling)
    mus = mean_vec(L_profiling, labels_profiling)


    delta = mus[0] - mus[1]
    # print(mus[0])
    # print(mus[1])
    inv_Sw = np.linalg.inv(Sw)
    m_distance = np.transpose(delta).dot(inv_Sw)
    m_distance = m_distance.dot(delta)
    print_centered(f"Mahalanobi distance: {m_distance}")
    # np.set_printoptions(precision=2)
    # print(delta.tolist())
    # for r in inv_Sw:
    #     print(r.tolist())
    # Sb = scatter_between(L_p, S_p)
    # Sw = scatter_within(L_p, S_p)
    # print(Sb.shape)
    # print(Sw.shape)
    # exit()
    # W = project_vector(Sw, Sb, 2)
    #
    #
    # Y = L_p.dot(W)
    # # print(Y.shape)
    # Y0 = Y[S_p==0]
    # Y1 = Y[S_p==1]
    #
    # plt.scatter(Y0[:, 0], Y0[:, 1], label="class0", s=10)
    # plt.scatter(Y1[:, 0], Y1[:, 1], label="class1", s=10, alpha=0.5)
    V = project_vector(Sw, Sb)
    # for i in range(KYBER_N_2):
    #     print(f"c {i}")
    #     print(f"S1 {S1[i]} mean L {mus[0][i]}")
    #     print(f"S2 {S2[i]} mean L {mus[1][i]}")
    #     print(f"Project coeff {V[i]}")
    #     print("=================================")
    # for i in range(KYBER_N_2):
    #     print("====================")
    #     print(f"c{i}")
    #     print(f"share 1 {S1[i]} mean {mus[0][i]}")
    #     print(f"share 2 {S2[i]} mean {mus[1][i]}")
    #     print(f"Diff {np.abs(mus[0][i] - mus[1][i])}")
    #     print(f"Project coeff {V[i]}")
    #     print("====================")
    # print(mus[0].dot(V), mus[1].dot(V))
    # V = V/V.min()
    # m_1 = 0
    # m_2 = 0


    # for s in S_SET:
    #     print(s)
    #     s = (s+KYBER_Q)%KYBER_Q
    #     s_pos1 = np.nonzero(S1==s)[0]
    #     s_pos2 = np.nonzero(S2==s)[0]
    #     Ls1 = L1[:, s_pos1].copy()
    #     Ls2 = L2[:, s_pos2].copy()
    #     Ls1 = np.ravel(Ls1)
    #     Ls2 = np.ravel(Ls2)
    #     p0, x0 = np.histogram(Ls1, bins=100, density=True)
    #     plt.plot(x0[:-1], p0, label=f"class0 s={s}", alpha=0.75)
    #     p0, x0 = np.histogram(Ls2, bins=100, density=True)
    #     plt.plot(x0[:-1], p0, label=f"class1 s={s}", alpha=0.75)


    #     print(Ls1.mean(), Ls2.mean(), np.abs(Ls1.mean()-Ls2.mean()))
    #     v1 = V[s_pos1].squeeze()
    #     v2 = V[s_pos2].squeeze()
    #     m_1 += (Ls1*v1).sum()
    #     m_2 += (Ls2*v2).sum()
    #     # print(s, Ls1*v1, Ls2*v2, np.abs(Ls1*v1 - Ls2*v2))
    # print(m_1, m_2, np.abs(m_1 - m_2))

    Y = L_profiling.dot(V)
    Y0 = Y[labels_profiling==0]
    Y1 = Y[labels_profiling==1]
    m1 = Y0.mean()
    m2 = Y1.mean()
    print_centered(f"mean 1 {m1} mean 2 {m2} mean_distance = {np.abs(m1-m2)}")
    # print(m1, m2)
    mus = np.array([m1, m2])
    m = (m1+m2)/2



    # LDA projection
    p0, x0 = np.histogram(Y0, bins=100, density=True)
    plt.plot(x0[:-1], p0, label="class 1 LDA projection", alpha=0.75, linewidth=2)
    p1, x1 = np.histogram(Y1, bins=100, density=True)
    plt.plot(x1[:-1], p1, label="class 2 LDA projection", alpha=0.75, linewidth=2)

    p0_mx = p0.max()
    p1_mx = p1.max()


    plt.hlines(y=p1_mx/2, xmin=mus.min(), xmax=mus.max(), colors="tab:red")
    plt.text(x=m, y=p1_mx/2, s=f"{np.abs(m1-m2):0.4f}" )
    #
    # # LDA classification
    pooled_cov = scatter_within(Y, labels_profiling)
    pooled_cov = np.sqrt(pooled_cov)
    # pooled_cov = np.power(pooled_cov, 2)
    mu_projected = mean_vec(Y, labels_profiling)
    print(mu_projected, pooled_cov)
    gp_0 = pdf_normal(np.sort(Y0, axis=0), mu_projected[0], pooled_cov)
    gp_1 = pdf_normal(np.sort(Y1, axis=0), mu_projected[1], pooled_cov)
    plt.vlines(x=m1, ymin=0, ymax=gp_0.max(), color="tab:grey")
    plt.text(x=m1, y=-0.001, s=f"{m1:0.4f}" )
    plt.vlines(x=m2, ymin=0, ymax=gp_1.max(), color="tab:grey")
    plt.text(x=m2, y=-0.001, s=f"{m2:0.4f}" )
    plt.plot(np.sort(Y0, axis=0), gp_0, label="class 1 Gaussian estimation", linewidth=2, color="tab:blue")
    plt.plot(np.sort(Y1, axis=0), gp_1, label="class 2 Gaussian estimation", linewidth=2, color="tab:orange")
    #

    plt.legend(fontsize=14)
    plt.title(f"{lh1.n_shares} share {combif} {wing} wing noise {noise}")
    plt.savefig(f"pic/LDA_projection_{wing}wing_{lh1.n_shares}shares_{combif}_noise_{noise}_{n_profiling}_{m_decs}_{n_coeff}_fulllwing.png", bbox_inches='tight')
    # plt.show()
    return mus


def combined_L_dist(d_name, combif, noise, model=ID):
    # plt.figure(figsize=(12, 12))
    lh = Leakage_Handler(d_name)
    print_centered(f"========================={lh.n_shares} SHARES=======================")
    n_coeff = 128
    wing = "right"
    # f_n = f"log/combined_L_{d_name}_{combif}_{n_coeff}_rightwing_noise_{noise}.npy"
    combined_L = combine_leakage(d_name, combif, n_coeff=n_coeff, wing=wing, f_decs=f"{wing}wing", add_noise=noise, model=model)

    secret = lh.get_secrets()
    secret = fix_s(secret, to_Q=True)
    secret = secret[KYBER_N_2:]
    # print(d_name)
    # print(secret.tolist())
    # plt.figure(figsize=(12, 12))
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    mus = []
    var_S = []
    # print(combined_L.mean(axis=0, keepdims=True))
    fopt = np.zeros(len(S_SET))
    for i, s in enumerate(S_SET):
        s_pos = np.where(secret==s)[0]
        combined_L_s = combined_L[:, s_pos[1]].copy()
        # combined_L_s = np.ravel(combined_L_s)
    #     fopt[s+2] = combined_L_s.mean()
    # s_ord = np.array([-2, -1, 0, 1, 2])
    # plt.plot(s_ord, fopt)
    # plt.scatter(s_ord, fopt)
    # plt.title(f"{combif} {lh.n_shares} E(C|s)")
        # combined_L_s = combined_L_s[:, 0]
        p0, x0 = np.histogram(combined_L_s, bins=100, density=True)
        m_s = combined_L_s.mean()
        var_s = np.var(combined_L_s)
        axs[0].plot(x0[:-1], p0, label=f"s = {s}", color=TAB_COLORS[i])
        axs[0].legend()
        mus.append(m_s)
        var_s = np.sqrt(var_s)
        var_S.append(var_s)
    #     # plt.text()
    #     print(s, s_pos[0], len(s_pos), len(combined_L_s), m_s, var_s)
    axs[0].set_title(f"Combined leakage distribution {combif} {lh.n_shares} shares")
    for i, s in enumerate(S_SET):
        axs[1].errorbar(mus[i], i,  xerr=var_S[i], fmt='o', capsize=6,  color=TAB_COLORS[i])
    plt.savefig(f"pic/Combined leakage_{combif}_{lh.n_shares}_noise_{noise}.png", bbox_inches='tight')
    # plt.savefig(f"pic/fopt_{combif}_{lh.n_shares}_noise_{noise}.png", bbox_inches='tight')
    plt.show()

def leakage_dist(d_name):
    lh = Leakage_Handler(d_name)

    total_N = lh.poly_per_file*lh.n_files
    n_files_og  = lh.n_files
    coeff_list = np.arange(128, 134)
    # lh.n_files = 10
    # for i in range(lh.n_shares):
    #     print_centered(f"====SNR SHARE {i+1}====")
    #     lh.get_snr_on_share(i, coeff_list=coeff_list, n_chunks=1, f_des=f"{coeff_list[0]}_{coeff_list[-1]}", add_noise=0)
    #     lh.get_PoI_on_share(i, coeff_list, 1, n_poi=1, f_des=f"{coeff_list[0]}_{coeff_list[-1]}", add_noise=0, display=False)

    lh.n_files = 20
    L = np.zeros((lh.n_shares, lh.n_files*lh.poly_per_file, lh.n_samples//16), dtype=np.int16)
    SHARES = np.zeros((lh.n_shares, lh.n_files*lh.poly_per_file, len(coeff_list)), dtype=np.uint16)
    for fi in trange(lh.n_files, desc="GETDATA-FILE|"):
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"
            traces = np.load(f_t)
            Li = traces[:, range(lh.n_samples//16)]
            t_idx = np.arange(fi*lh.poly_per_file, fi*lh.poly_per_file+lh.poly_per_file)
            L[share_i, t_idx] = Li.copy()
            f_d = f"{lh.file_path}_meta_{fi}.npy"
            data = np.load(f_d, allow_pickle=True)
            data = data.item().get(f"share_{share_i}")[:, coeff_list]
            SHARES[share_i, t_idx] = data.copy()
    SHARES = ID(SHARES)
    # SHARES = HW(SHARES)
    subtrace = np.arange(lh.n_samples//16)

    fig, axs = plt.subplots(lh.n_shares, 1, figsize=(12, 12))
    for shi in trange(lh.n_shares, desc="SHARE|"):
        Lshare = L[(shi+1)%2].copy()
        Xi = SHARES[shi].copy()
        print(Lshare.shape, Lshare.dtype, Xi.shape, Xi.dtype)
        snr = SNR(HW_Q, len(subtrace), len(coeff_list), use_64bit=True)
        snr.fit_u(Lshare, Xi)
        snr_val = snr.get_snr()
        for c_i, c in enumerate(coeff_list):
            print(np.argmax(snr_val[c_i]))
        #     p_coeff = np.zeros(len(subtrace))
        #     for i in subtrace:
        #         p_coeff[i] = np.abs(np.corrcoef(x=Lshare[:, i], y=Xi[:, c_i])[0, 1])
            axs[shi].plot(subtrace, snr_val[c_i], label=f"$c_{{{c}}}$")
        axs[shi].legend()
        axs[shi].set_title(f"SHARE {shi+1}")
    fig.suptitle("SNR")
    plt.show()


    # secret = lh.get_secrets()[KYBER_N_2:]
    # secret = fix_s(secret, to_Q=True)
    # # fig, axs = plt.subplots(1, 5)
    # for i, s in enumerate(S_SET):
    #     s_pos = np.where(secret==s)[0]
    #     print(len(s_pos))
    #     L_s = []
    #     for share_i in range(lh.n_shares):
    #         _ = L[share_i, :, s_pos].copy()
    #         _ = _.reshape(len(s_pos)*total_N)
    #         L_s.append(_)
    #     plt.hist2d(L_s[0], L_s[1], bins=100, density=True)
    #     print(s,)
    #     # axs[i].set(aspect='equal', adjustable='box')
    #     # axs[i].set_title(f"s = {s}")
    #     plt.show()





if __name__=="__main__":
    # nois_correlation()
    # d1 = "021123_1335"
    # leakage_dist(d1)
    # exit()
    # leakage_dist(d1)
    # correlation_c(d1, combif="sum")
    # d1 = "151123_0938"
    # correlation_c(d1, combif="sum")
    # d1 = "161123_1113"
    # correlation_c(d1, combif="sum")
    # d2 = "151123_0938"


    # d2 = "161123_1113"
    # d1 = "021123_1335"
    # d2 = "021123_1506"
    # leakage_dist("021123_1335")
    # leakage_dist("021123_1506")
    # correlation_c(d2, combif="sum")
    # data_sanity_check("021123_1335")
    # shares_sanity_check("021123_1335")
    # shares_sanity_check("021123_1506")
    # shares_sanity_check("131123_1401")
    # shares_sanity_check("141123_1246")
    # share_dist_inspect("131123_1401", share_i=0, c=2, n_coeff=KYBER_N_2)

    noise = 0
    combifs = ["sum", "prod", "norm_prod"]
    # run_onM(n_profiling=40000, noise=500, n_shares=2, n_coeff=128, combif="sum", model=ID)
    # run_onM(n_profiling=40000, noise=0, n_shares=2, n_coeff=32, combif="sum", model=ID)


    # run_onM(noise=noise, n_shares=2, combif=combif)
    # run_onM(noise=noise, n_shares=3, combif=combif)
    # run_onM(noise=noise, n_shares=4, combif=combif)
    # combif = "norm_prod"
    # run_onM(noise=noise, n_shares=2, combif=combif)
    # run_onM(noise=0, n_shares=3)
    # run_onM(noise=0, n_shares=4)
    # for combif in combifs:
    #     d = "021123_1335"
    #     combined_L_dist(d, combif, noise=noise)
    #     d = "161123_0940"
    #     combined_L_dist(d, combif, noise=noise)
    #     d = "161123_1832"
    #     combined_L_dist(d, combif, noise=noise)
    # d = "171123_1644"
    # combined_L_dist(d, combif, noise=noise)
    # combif = "sum"
    # run_onM(n_profiling=40000, noise=0, n_shares=3, combif="sum", model=ID)
    # run_onM(n_profiling=40000, noise=0, n_shares=3, combif="abs_diff", model=ID)
    # run_onM(n_profiling=40000, noise=0, n_shares=4, combif="norm_prod", model=ID)
    # run_onM(n_profiling=80000, noise=0, n_shares=2, combif="abs_diff", model=ID)
    # run_onM(n_profiling=100000, noise=500, n_shares=2, combif=combif, model=ID)
    # run_onM(n_profiling=100000, noise=1000, n_shares=2, combif=combif, model=ID)
    # run_onM(n_profiling=100000, noise=2000, n_shares=2, combif=combif, model=ID)
    for combif in combifs:
    #     run_onM(n_profiling=40000, noise=noise, n_shares=2, combif=combif, model=ID)
        run_onM(n_profiling=40000, noise=noise, n_shares=3, combif=combif, model=ID)
        run_onM(n_profiling=40000, noise=noise, n_shares=4, combif=combif, model=ID)
    #     run_onM(n_profiling=40000, noise=noise, n_shares=5, combif=combif, model=ID)
    #     run_onM(n_profiling=40000, noise=noise, n_shares=6, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=500, n_shares=6, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=1000, n_shares=6, combif=combif, model=ID)
        # run_onM(n_profiling=20000, noise=noise, n_shares=5, combif=combif, model=ID)
    # for i in range(2, 6):
    #     print_centered(f"========={i} SHARES========")
    #     mus = run_onM(noise=noise, n_shares=i, combif=combif)
    #     np.set_printoptions(precision=2)
    #     print((mus[0]-mus[1]).tolist())


    # run_onM(noise=500)
    # run_onM(noise=1000)
    # run_onM(noise=2000)
    # run_onSim("abs_diff")
    # run_onSim("sum")
    # run_onSim("norm_prod")
