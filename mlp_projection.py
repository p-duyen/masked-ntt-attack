import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
import pickle
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange

import math
from helpers import *
from mlp_arch import *


def combine_leakage_full(d_name, combif, n_poi=1, f_desc=None, add_noise=False, model=ID):
    m_desc="HW" if model is HW else "ID"
    f_n = f"log/combined_L_{d_name}_{combif}_{n_poi}_noise_{add_noise}_{m_desc}_fullp.npy"
    lh = Leakage_Handler(d_name)
    if os.path.exists(f"{f_n}"):
        print_centered(f"COMBINED LEAKAGE FOR {d_name} {lh.n_shares} {combif} FULL COEFFS {n_poi} POIs noise {add_noise} {m_desc} EXISTED")
        with open(f_n, "rb") as f:
            combined_L = np.load(f)
        print_centered(f"COMBINED LEAKAGE FOR {d_name} {lh.n_shares} {combif} FULL COEFFS {n_poi} POIs noise {add_noise} {m_desc} LOADED")
        return combined_L.astype(np.float32)
    print_centered(f"GENERATE COMBINED LEAKAGE FOR {d_name} {lh.n_shares} {combif} FULL COEFFS {n_poi} POIs noise {add_noise} {m_desc}")

    n_files_og = lh.n_files
    total_N = lh.poly_per_file*n_files_og
    if combif=="norm_prod":
        temp = np.zeros((lh.n_shares, total_N, KYBER_N*n_poi))
    if combif in ["abs_diff", "sum"]:
        combined_L = np.zeros((total_N, KYBER_N*n_poi), dtype=np.float32)
    elif combif in ["prod", "norm_prod"]:
        combined_L = np.ones((total_N, KYBER_N*n_poi), dtype=np.float32)
    POI = {}
    for share_i in range(lh.n_shares):
        poi_file = f"{lh.dir_path}/POI_on_share_{n_poi}_poi_{share_i}_ID_{f_desc}_noise_{add_noise}_fullp.npy"
        if os.path.exists(poi_file):
            print_centered(f"POI FILE {d_name} {lh.n_shares} share {share_i+1} full COEFFS {n_poi} {add_noise} {m_desc} EXISTED")
            with open(poi_file, "rb") as f:
                POI[f"share_{share_i}"] = np.load(f)
        else:
            print_centered(f"GENERATING POI FILE {d_name} {lh.n_shares} share {share_i+1} full COEFFS {n_poi} {add_noise} {m_desc} ...")
            lh.get_PoI_on_share_fullp(share_i, n_chunks=16, n_poi=n_poi, f_des=None, add_noise=add_noise, model=ID, display=None, savepoi=True)
            POI[f"share_{share_i}"] = lh.poi_full[f"share_{share_i}"].copy()
        for t in [0, 1, 2, 128, 129, 130]:
            print(t, POI[f"share_{share_i}"][t])
    lh.n_files = n_files_og
    for fi in trange(lh.n_files, desc="GETDATA-FILE|"):
        for share_i in range(lh.n_shares):
            f_t = f"{lh.file_path}_traces_{fi}_share_{share_i}.npy"

            traces = np.load(f_t)
            poi =  POI[f"share_{share_i}"].astype(np.uint32)
            Li = traces[:, poi.flatten()].copy()
            if add_noise:
                Li = Li + np.random.normal(0, add_noise, size=Li.shape)
                # Li = Li.astype(np.int16)
            # print(lh.poly_per_file*fi, lh.poly_per_file*fi+lh.poly_per_file)
            if combif=="abs_diff":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] =  Li.copy() - combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]
            elif combif=="sum":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] =  Li.copy() + combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]
            elif combif=="prod":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] = combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]*(Li.copy()/100)
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
        # combined_L*=1000


    if combif=="abs_diff":
        with open(f_n, "wb") as f:
            np.save(f, np.abs(combined_L))
        return np.abs(combined_L)
    with open(f_n, "wb") as f:
        np.save(f, combined_L)
    return combined_L
def classify_(d1, d2, L1, L2, n_profiling):

    # DATA PREPROCESS
    lh1 = Leakage_Handler(d1)

    lh2 = Leakage_Handler(d2)

    labels_profiling = np.zeros(n_profiling, dtype=np.int8)
    print("OG size", L1.shape)
    idx_L1 = np.random.choice(len(L1), n_profiling//2, replace=False)
    idx_L2 = np.random.choice(len(L2), n_profiling//2, replace=False)
    L_profiling = L1[idx_L1].copy()
    L_profiling = np.append(L_profiling, L2[idx_L2].copy(), axis=0)
    labels_profiling[range(n_profiling//2, n_profiling)] = 1
    shuffle_idx = np.arange(n_profiling)
    np.random.shuffle(shuffle_idx)
    L_profiling = L_profiling[shuffle_idx]
    labels_profiling = labels_profiling[shuffle_idx]


    idx_a_L1 = np.setdiff1d(np.arange(len(L1)), idx_L1)
    print(len(idx_a_L1), len(idx_a_L1))
    idx_a_L2 = np.setdiff1d(np.arange(len(L2)), idx_L2)
    L_attack = L1[idx_a_L1].copy()
    L_attack = np.append(L_attack, L2[idx_a_L2].copy(), axis=0)

    labels_attack = np.zeros(len(L_attack), dtype=np.int8)
    labels_attack[len(idx_a_L1):] = 1

    labels_attack = np.expand_dims(labels_attack, 1)
    labels_profiling = np.expand_dims(labels_profiling, 1)

    print("train size", L_profiling.shape)
    print("val size", L_attack.shape)
    # DATASET
    train_batch = L_profiling.shape[0]//20
    val_batch = labels_attack.shape[0]//5
    trainset = LeakageData(L_profiling, labels_profiling)
    traindata = DataLoader(trainset, batch_size=train_batch, num_workers=20, shuffle=True)
    valset = LeakageData(L_attack, labels_attack)
    print(valset[0][0].dtype, valset[0][1].dtype)
    print(valset[0][0].shape, valset[0][1].shape)
    valdata = DataLoader(valset, batch_size=val_batch, num_workers=10, shuffle=False)


    early_stop_callback = EarlyStopping_(monitor="val_loss", patience=4, mode="min")

    #TRAINER SETUP
    mlp = MLP_BIN_BCE(in_dim=L_profiling.shape[1])
    print("indim", L_profiling.shape[1])
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                            max_epochs=200,
                            check_val_every_n_epoch=1,
                            callbacks=[early_stop_callback],
                            detect_anomaly=True)

    #TRAINING
    trainer.fit(mlp, traindata, valdata)

    #Transform (on attack data)
    pred_data = DataLoader(valset, batch_size=L_attack.shape[0], num_workers=10, shuffle=False)
    res = trainer.predict(mlp, pred_data)

    Y_TRANS = res[0][0].numpy(force=True)
    Y = res[0][1].numpy(force=True)
    PRED =  res[0][2].numpy(force=True)
    Y = Y.squeeze()
    PRED = PRED.squeeze()
    print(PRED[:10])
    PRED[PRED>0.5] = 1
    PRED[PRED<0.5] = 0

    correc_guess = np.nonzero(Y==PRED)
    print(correc_guess)
    correc_guess = correc_guess[0]
    avg_score = len(correc_guess)/len(Y)
    print_centered(f"===========AVG SCORE {avg_score}============", "yellow")


    #Display transformed

    Y0 = Y_TRANS[Y==0]
    Y1 = Y_TRANS[Y==1]
    m1 = Y0.mean()
    m2 = Y1.mean()
    mus = np.array([m1, m2])
    m = (m1+m2)/2

    # plt.scatter(Y0[:, 0], Y0[:, 1], label="class 1")
    # plt.scatter(Y1[:, 0], Y1[:, 1], label="class 2")



    p0, x0 = np.histogram(Y0, bins=100, density=True)
    plt.plot(x0[:-1], p0, label="class 1 MLP projection", alpha=0.75, linewidth=2)
    p1, x1 = np.histogram(Y1, bins=100, density=True)
    plt.plot(x1[:-1], p1, label="class 2 MLP projection", alpha=0.75, linewidth=2)

    p0_mx = p0.max()
    p1_mx = p1.max()

    plt.hlines(y=p1_mx/2, xmin=mus.min(), xmax=mus.max(), colors="tab:red")
    plt.text(x=m, y=p1_mx/2, s=f"{np.abs(m1-m2):0.4f}" )
    print_centered(f"Mean distace: {np.abs(m1-m2)}")
    plt.vlines(x=m1, ymin=0, ymax=p0_mx, color="tab:grey")
    plt.vlines(x=m2, ymin=0, ymax=p1_mx, color="tab:grey")

    plt.hlines(y=p1_mx/2, xmin=mus.min(), xmax=mus.max(), colors="tab:red")
    plt.text(x=m, y=p1_mx/2, s=f"{np.abs(m1-m2):0.4f}" )
    return avg_score



def attack_sr(n_profiling, n_shares, combif, n_coeff=128, wing="right", n_poi=1, model=ID, c_i=0, rep=1):

    m_decs = "ID" if model is ID else "HW"
    # 2 shares

    # 3 shares
    if n_shares==2:
        d1 = "021123_1335"
        # d2 = "021123_1335"
        d2 = "021123_1506"
        # d1 = "240124_1424" #s=0
        # d2 = "240124_1856" #s=0
        # d2 = "240124_1502" #s=10000....
        # d2 = "240124_1654" #s=0
        # d2 = "240124_1606" #s=00000....000001
    elif n_shares==3:
        d1 = "221123_2119"
        d2 = "221123_2328"
    elif n_shares==4:
        # 4 shares
        d1 = "231123_0138"
        d2 = "231123_0431"
    elif n_shares==5:
        d1 = "231123_1239"
        # d2 = "231123_1239"
        d2 = "231123_0903"
    LH1 = Leakage_Handler(d1)
    LH2 = Leakage_Handler(d2)
    # L1 = combine_leakage(d1, combif, n_coeff=128, n_poi=n_poi, wing=wing, desc=f"{wing}wing", add_noise=False, model=model)
    # L2 = combine_leakage(d2, combif, n_coeff=128, n_poi=n_poi, wing=wing, desc=f"{wing}wing", add_noise=False, model=model)
    L1 = combine_leakage_full(d1, combif, n_poi=n_poi, f_desc=None, add_noise=False, model=ID)
    for shi in range(n_shares):
        os.system(f"cp {LH1.dir_path}/POI_on_share_{n_poi}_poi_{shi}_ID_None_noise_False_fullp.npy {LH2.dir_path}/POI_on_share_{n_poi}_poi_{shi}_ID_None_noise_False_fullp.npy")
    L2 = combine_leakage_full(d2, combif, n_poi=n_poi, f_desc=None, add_noise=False, model=ID)

    print_centered(f"=====Combines leakage share {L1.shape} {L2.shape}===========")
    if n_coeff == 1:
        L1 = L1[:, np.arange(c_i*n_poi, c_i*n_poi+n_poi)]
        L2 = L2[:, np.arange(c_i*n_poi, c_i*n_poi+n_poi)]
    else:
        poi_idx = L1.shape[1]
        print(L1.shape)
        print("wing", wing)
        if wing=="left":
            L1 = L1[:, np.arange(poi_idx//2)].copy()
            L2 = L2[:, np.arange(poi_idx//2)].copy()
            print_centered(f"+====Combines leakage share {L1.shape} {L2.shape}===========")
        elif wing=="right":
            L1 = L1[:, np.arange(poi_idx//2, poi_idx)].copy()
            L2 = L2[:, np.arange(poi_idx//2, poi_idx)].copy()
            print_centered(f"-====Combines leakage share {L1.shape} {L2.shape}===========")
    print_centered(f"=====Combines leakage share {L1.shape} {L2.shape}===========")


        # c_list = np.random.choice(KYBER_N, n_coeff, replace=False)
        # for ci in c_list:
        #     poi_c =
    #     # c_list.sort()
    #     c_list = np.arange()
    if rep == 2:
        L1_ = np.concatenate((L1, L2), axis=1)
        L2_ = np.concatenate((L2, L1), axis=1)
    elif rep == 3:
        L1_ = np.concatenate((L1, L1, L1), axis=1)
        L2_ = np.concatenate((L2, L2, L2), axis=1)
    else:
        L1_ = L1.copy()
        L2_ = L2.copy()
    L1_ = L1_.astype(np.float32)
    L2_ = L2_.astype(np.float32)
    print_centered(f"==========={combif} POI {n_poi} k={rep}==============", "yellow")
    plt.figure(figsize=(12, 12))
    acc = classify_(d1, d2, L1_, L2_, n_profiling)


    plt.title(f"{n_shares} shares #coeff={n_coeff} #POI={n_poi} combif={combif} acc={acc:0.4f} k={rep}")

    if rep <= 1:
        plt.savefig(f"pic/MLP_{n_shares}_shares_{wing}_wing_{n_poi}_poi_{combif}.png")
    else:
        plt.savefig(f"pic/MLP_{n_shares}_shares_{wing}_wing_{n_poi}_poi_{combif}_k={rep}.png")




def run_onM(n_profiling, noise, n_shares, combif, n_coeff=128, n_poi=1, model=ID, c_i=0):
    m_decs = "ID" if model is ID else "HW"
    wing = "right"
    # 2 shares

    # 3 shares
    if n_shares ==2:
        d1 = "021123_1335"
        # d2 = "021123_1335"
        d2 = "021123_1506"
    elif n_shares==3:
        d1 = "221123_2119"
        # d2 = "221123_2119"
        d2 = "221123_2328"

    elif n_shares==4:
        # 4 shares
        d1 = "231123_0138"
        d2 = "231123_0431"
        # d2 = "231123_0138"
    elif n_shares==5:
        d1 = "231123_1239"
        # d2 = "231123_1239"
        d2 = "231123_0903"
    elif n_shares==6:
        d1 = "060124_2112"
        # d2 = "060124_2112"
        d2 = "070124_1718"


    L1 = combine_leakage(d1, combif, n_coeff=128, n_poi=n_poi, wing=wing, f_decs=f"{wing}wing", add_noise=noise, model=model)
    L2 = combine_leakage(d2, combif, n_coeff=128, n_poi=n_poi, wing=wing, f_decs=f"{wing}wing", add_noise=noise, model=model)

    lh1 = Leakage_Handler(d1)
    S1 = lh1.get_secrets()
    S1 = S1[KYBER_N_2:].astype(np.uint16)
    lh2 = Leakage_Handler(d2)
    S2 = lh2.get_secrets()
    S2 = S2[KYBER_N_2:].astype(np.uint16)

    if n_coeff == 1:
        L1 = L1[:, np.arange(c_i*n_poi, c_i*n_poi+n_poi)]
        L2 = L2[:, np.arange(c_i*n_poi, c_i*n_poi+n_poi)]
    else:
        L1 = L1[:, np.arange(n_coeff*n_poi)]
        L2 = L2[:, np.arange(n_coeff*n_poi)]


    plt.figure(figsize=(18, 12))
    total_N = len(L1)
    # print(total_N)

    # L_profiling = np.zeros((n_profiling, n_coeff))
    labels_profiling = np.zeros(n_profiling, dtype=np.uint16)
    idx_L1 = np.random.choice(len(L1), n_profiling//2)
    idx_L2 = np.random.choice(len(L2), n_profiling//2)
    L_profiling = L1[idx_L1].copy()
    # print(L_profiling.shape)
    L_profiling = np.append(L_profiling, L2[idx_L2].copy(), axis=0)
    # print(L_profiling.shape)

    labels_profiling[range(n_profiling//2, n_profiling)] = 1

    shuffle_idx = np.arange(n_profiling)
    np.random.shuffle(shuffle_idx)
    L_profiling = L_profiling[shuffle_idx]
    labels_profiling = labels_profiling[shuffle_idx]

    print_centered("=============BEFORE TRANSFORM=============")
    print("MEAN CLASS 1", L_profiling[labels_profiling==0].mean(axis=0))
    print("MEAN CLASS 2", L_profiling[labels_profiling==1].mean(axis=0))
    print("to projection", L_profiling.shape)



    Sb = scatter_between(L_profiling, labels_profiling)
    Sw = scatter_within(L_profiling, labels_profiling)
    mus = mean_vec(L_profiling, labels_profiling)


    delta = mus[0] - mus[1]
    inv_Sw = np.linalg.inv(Sw)
    m_distance = np.transpose(delta).dot(inv_Sw)
    m_distance = m_distance.dot(delta)
    print_centered(f"Mahalanobi distance: {m_distance}")
    V = project_vector(Sw, Sb)
    print_centered("===================PROJECT VEC===============")
    print(V)

    Y = L_profiling.dot(V)
    Y0 = Y[labels_profiling==0]
    Y1 = Y[labels_profiling==1]
    m1 = Y0.mean()
    m2 = Y1.mean()
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

    # LDA projection
    pooled_cov = scatter_within(Y, labels_profiling)
    pooled_cov = np.sqrt(pooled_cov)
    mu_projected = mean_vec(Y, labels_profiling)

    # LDA pdf
    gp_0 = pdf_normal(np.sort(Y0, axis=0), mu_projected[0], pooled_cov)
    gp_1 = pdf_normal(np.sort(Y1, axis=0), mu_projected[1], pooled_cov)
    plt.vlines(x=m1, ymin=0, ymax=gp_0.max(), color="tab:grey")
    plt.text(x=m1, y=-0.001, s=f"{m1:0.4f}" )
    plt.vlines(x=m2, ymin=0, ymax=gp_1.max(), color="tab:grey")
    plt.text(x=m2, y=-0.001, s=f"{m2:0.4f}" )
    plt.plot(np.sort(Y0, axis=0), gp_0, label="class 1 Gaussian estimation", linewidth=2, color="tab:blue", linestyle="dashed")
    plt.plot(np.sort(Y1, axis=0), gp_1, label="class 2 Gaussian estimation", linewidth=2, color="tab:orange", linestyle="dashed")


    plt.legend(fontsize=14)
    if n_coeff==1:
        plt.title(f"{lh1.n_shares} shares {combif} {wing} wing ncoeff {n_coeff} nPOI {n_poi} noise {noise} \n r_0 = {S1[c_i]}\n r_1 = {S2[c_i]}")
    else:
        plt.title(f"{lh1.n_shares} shares {combif} {wing} wing ncoeff {n_coeff} nPOI {n_poi} noise {noise}")

    # plt.savefig(f"pic/LDA_projection_{wing}wing_{lh1.n_shares}shares_{combif}_noise_{noise}_{n_profiling}_{m_decs}_{n_coeff}_{n_poi}.png", bbox_inches='tight')

    plt.show()
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


def poi_check(d_name, wing, share_i, n_poi):
    lh = Leakage_Handler(d_name)
    lh.n_files = 15 if wing=="right" else 40
    coeff_list = np.arange(KYBER_N_2) if wing=="left" else np.arange(KYBER_N_2, KYBER_N)
    n_chunks = 4
    n_c = len(coeff_list)/n_chunks
    lh.get_snr_on_share(share_i, coeff_list=coeff_list, n_chunks=n_chunks, f_des=f"test", add_noise=0, model=ID)
    TRACES, SNR_VAL = lh.get_PoI_on_share(share_i, coeff_list, n_chunks, n_poi=n_poi, f_des=f"test", add_noise=0, display=False, model=ID, keepsnr=True)
    print_centered(f"==================={d_name} share {share_i+1} {wing} wing ============")
    coeff_chunks = np.split(coeff_list, n_chunks)
    for chunk_i, c_chunk in enumerate(coeff_chunks):
        traces_chunk = TRACES[chunk_i]
        snr_chunk = SNR_VAL[chunk_i]
        for ci, c in enumerate(c_chunk):
            poi_c = lh.poi[f"share_{share_i}"][chunk_i*len(c_chunk)+ci]
            poi_c = poi_c.astype(np.uint32)

            print(f"chunk {chunk_i} c_{c},  poi={poi_c} {poi_c.shape}")
            if chunk_i > 2:
                if ci < 5:
                    print((traces_chunk).shape, (snr_chunk[ci]).shape)
                    print((traces_chunk[poi_c]).shape, (snr_chunk[ci][poi_c]).shape)
                    plt.plot(traces_chunk, snr_chunk[ci], label=f"$c_{{{c}}}$")
                    plt.scatter(poi_c, snr_chunk[ci][poi_c])
        if chunk_i > 2:
            plt.legend()
            plt.savefig("pic_i/{d_name}_snr_chunk{chunk_i}.png")


if __name__=="__main__":
    # attack_sr(n_profiling=40000, n_shares=2, combif="abs_diff", n_coeff=256, wing="both", n_poi=20, model=ID, c_i=0)
    # attack_sr(n_profiling=40000, n_shares=2, combif="norm_prod", n_coeff=256, wing="both", n_poi=20, model=ID, c_i=0)
    # attack_sr(n_profiling=80000, n_shares=3, combif="abs_diff", n_coeff=256, wing="both", n_poi=100, model=ID, c_i=0)
    # attack_sr(n_profiling=80000, n_shares=3, combif="norm_prod", n_coeff=256, wing="both", n_poi=100, model=ID, c_i=0)
    # attack_sr(n_profiling=80000, n_shares=3, combif="sum", n_coeff=256, wing="both", n_poi=100, model=ID, c_i=0)
    # attack_sr(n_profiling=80000, n_shares=3, combif="sum", n_coeff=256, wing="both", n_poi=20, model=ID, c_i=0)
    # attack_sr(n_profiling=80000, n_shares=3, combif="sum", n_coeff=256, wing="both", n_poi=50, model=ID, c_i=0)
    # poi_check(d_name="021123_1335", wing="left", share_i=0, n_poi=20)
    # noise = 0
    # combifs = ["prod", "norm_prod"]
    # for combif in combifs:
    #     run_onM(n_profiling=40000, noise=noise, n_shares=3, combif=combif, model=ID)
    #     run_onM(n_profiling=40000, noise=noise, n_shares=6, combif=combif, model=ID)
    # run_onM(n_profiling=40000, noise=500, n_shares=2, n_coeff=128, combif="sum", model=ID)
    # run_onM(n_profiling=40000, noise=0, n_shares=2, n_coeff=32, combif="sum", model=ID)
    N_POI = [50]
    Fs = ["norm_prod", "abs_diff", "sum"]
    for n_poi in N_POI:
        for combif in Fs:
            attack_sr(n_profiling=70000, n_shares=4, combif=combif, wing="both", n_coeff=256, n_poi=n_poi, model=ID, c_i=1, rep=1)
            attack_sr(n_profiling=70000, n_shares=4, combif=combif, wing="both", n_coeff=256, n_poi=n_poi, model=ID, c_i=1, rep=2)
            attack_sr(n_profiling=70000, n_shares=5, combif=combif, wing="both", n_coeff=256, n_poi=n_poi, model=ID, c_i=1, rep=1)
            attack_sr(n_profiling=70000, n_shares=5, combif=combif, wing="both", n_coeff=256, n_poi=n_poi, model=ID, c_i=1, rep=2)

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
    # for combif in combifs:
    #     attack_sr(n_profiling=50000, n_shares=2, combif=combif, n_coeff=128, n_poi=1, model=ID)
    # for k in range(128):
    #     run_onM(n_profiling=50000, noise=noise, n_shares=2, n_coeff=1, n_poi=4, combif="sum", model=ID, c_i=k)
        # attack_sr(n_profiling=50000, n_shares=2, combif="sum", n_coeff=1, n_poi=4, model=ID, c_i=k)
    # for combif in combifs:
    #     attack_sr(n_profiling=40000, n_shares=3, combif=combif, n_coeff=128, n_poi=50, model=ID, c_i=0)
    # attack_sr(n_profiling=50000, n_shares=2, combif="norm_prod", n_coeff=256, n_poi=10, model=ID, c_i=1)
    # attack_sr(n_profiling=50000, n_shares=2, combif="abs_diff", n_coeff=256, n_poi=51, model=ID, c_i=14)

    # attack_sr(n_profiling=40000, n_shares=2, combif="abs_diff", wing="both", n_coeff=256, n_poi=20, model=ID, c_i=14)
    # attack_sr(n_profiling=40000, n_shares=2, combif="sum", n_coeff=1, n_poi=10, model=ID, c_i=15)
    # attack_sr(n_profiling=40000, n_shares=2, combif="sum", n_coeff=1, n_poi=10, model=ID, c_i=16)
        # run_onM(n_profiling=50000, noise=noise, n_shares=2, n_coeff=2, combif=combif, model=ID)
        # run_onM(n_profiling=50000, noise=noise, n_shares=2, n_coeff=16, combif=combif, model=ID)
        # run_onM(n_profiling=50000, noise=noise, n_shares=2, n_coeff=32, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=noise, n_shares=3, n_coeff=128, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=noise, n_shares=4, n_coeff=128, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=noise, n_shares=5, n_coeff=128, combif=combif, model=ID)
        # run_onM(n_profiling=40000, noise=noise, n_shares=6, n_coeff=128, combif=combif, model=ID)
        # run_onM(n_profiling=30000, noise=noise, n_shares=, combif=combif, model=ID)
        # run_onM(n_profiling=30000, noise=noise, n_shares=4, combif=combif, model=ID)
        # run_onM(n_profiling=30000, noise=noise, n_shares=5, combif=combif, model=ID)
        # run_onM(n_profiling=30000, noise=noise, n_shares=6, combif=combif, model=ID)
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
