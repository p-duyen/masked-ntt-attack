import numpy as np
from matplotlib import pyplot as plt
from scalib.modeling import LDAClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import trange, tqdm
import os
from helpers import Leakage_Handler, print_centered

# TODO
"""
1. Combine leakage
    - Get poi for each share
    - Get L_i[poi]
    - Combine L_i[poi]
2. Classify
    - Get secrets
    - LDA
3. Attack
"""
KYBER_Q = 3329
KYBER_N = 256
KYBER_N_2 = 128

def combine_leakage(d_name, combif, f_decs=None):
    f_n = f"log/combined_L_{d_name}_{combif}_{f_decs}.npy"
    if os.path.exists(f"{f_n}"):
        print_centered(f"COMBINED LEAKAGE FOR {combif} EXISTED")
        with open(f_n, "rb") as f:
            combined_L = np.load(f)
        return combined_L
    print_centered(f"GENERATE COMBINED LEAKAGE FOR {combif}")
    lh = Leakage_Handler(d_name)
    n_files_og = lh.n_files
    total_N = lh.poly_per_file*lh.n_files
    if combif=="abs_diff":
        combined_L = np.zeros((total_N, 10))
    elif combif=="prod":
        combined_L = np.ones((total_N, 10))
    poi_shares = np.zeros((2, KYBER_N_2))
    for i in range(lh.n_shares):
        share_i_coeff = np.arange(KYBER_N*i, KYBER_N*i + KYBER_N)
        wings = np.split(share_i_coeff, 2)
        for wi, wing in enumerate(wings):
            if wi ==1:
                print(wing.tolist())
                lh.n_files = 10 if wi==1 else 50
                print_centered(f"==SNR on wing {wi}==")
                lh.get_snr(wing, 16, f_des=f"share_{i}_wing{wi}")
                print_centered("==GET POI==")
                lh.get_PoI(wing, 16, 1, f_des=f"share_{i}_wing{wi}", display=False)
                poi_shares[i] = lh.poi.squeeze()
    lh.n_files = n_files_og
    for fi in trange(lh.n_files, desc="GETDATA-FILE|"):
        ft = f"{lh.file_path}_traces_{fi}.npy"
        traces = np.load(ft).astype(np.int16)
        for i in range(lh.n_shares):
            trace_i = np.arange(i, lh.traces_per_file, 2)
            Li = traces[trace_i]
            poi =  poi_shares[i, range(10)] - i*lh.n_samples
            poi = poi.astype(np.uint32)
            Li = Li[:, poi].copy()
            Li = Li + np.random.normal(0, 300, size=Li.shape)
            if combif=="abs_diff":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] =  Li - combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]
            elif combif=="prod":
                combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file] = combined_L[lh.poly_per_file*fi:lh.poly_per_file*fi+lh.poly_per_file]*Li.copy()
    if combif=="abs_diff":
        with open(f_n, "wb") as f:
            np.save(f, np.abs(combined_L))
        return np.abs(combined_L)
    with open(f_n, "wb") as f:
        np.save(f, np.abs(combined_L))
    return combined_L

def classify(L1, L2, n_profiling, n_attack):
    total_N = len(L1)
    # print(total_N)
    L_p = np.zeros((n_profiling, L1.shape[1]), dtype=np.int16)
    S_p = np.zeros((n_profiling), dtype=np.uint16)
    idx_p1 = np.random.choice(n_profiling, n_profiling//2, replace=False)
    idx_L1 = np.random.choice(total_N, n_profiling//2, replace=False)
    L_p[idx_p1] = L1[idx_L1].copy()
    idx_p2 = np.setdiff1d(np.arange(n_profiling), idx_p1)
    idx_L2 = np.random.choice(total_N, n_profiling//2, replace=False)
    L_p[idx_p2] = L2[idx_L2].copy()
    S_p[idx_p2] = 1



    # cls = LDA(solver="eigen", n_components=1)
    # cls.fit(L_p, S_p)
    cls = LDAClassifier(2, 1, L_p.shape[1])
    cls.fit_u(L_p, S_p)
    cls.solve()

    chosen_state = np.random.randint(0, 2, 1)[0]
    if chosen_state==0:
        idx_a = np.setdiff1d(np.arange(total_N), idx_p1)
        idx_a = np.random.choice(idx_a, n_attack)
        L_a = L1[idx_a].copy()
    elif chosen_state==1:
        idx_a = np.setdiff1d(np.arange(total_N), idx_p2)
        idx_a = np.random.choice(idx_a, n_attack)
        L_a = L2[idx_a].copy()

    # print()
    L_a = L_a.astype(np.int16)
    preds = cls.predict_proba(L_a)
    # print(preds.shape)
    # print(preds[:10])
    # print(chosen_state)
    # exit()
    # preds = cls.predict_log_proba(L_a)
    # preds = cls.predict_proba(L_a.astype(np.int16))
    preds = np.log10(preds)
    # print(preds.shape)
    # print(preds.sum(axis=0))
    guess = np.argmax(preds.sum(axis=0))
    return 1 if guess==chosen_state else 0






if __name__ == '__main__':
    combif = "abs_diff"
    L1_combined = combine_leakage("021123_1335", combif)
    L2_combined = combine_leakage("021123_1506", combif)

    NP = np.array([1000])
    Na = np.arange(1, 20, 2)

    sr_curve = np.zeros(len(Na))
    for n_profiling in NP:
        for ni, n_attack in tqdm(enumerate(Na), total=len(Na)):
            res = 0
            for i in trange(100, desc="REP|"):
                print_centered(f"====={n_profiling} {n_attack} rep {i}=====")
                res +=classify(L1_combined, L2_combined, n_profiling, n_attack)
            print_centered(f"====={n_profiling} {n_attack} SR {res}=====")
            sr_curve[ni] = res
        with open(f"SR_{combif}_{n_profiling}.npy", "wb") as f:
            np.save(f, Na)
            np.save(f, sr_curve)
        plt.plot(Na, sr_curve, label=f"{combif}")


    # combif = "prod"
    # L1_combined = combine_leakage("021123_1335", combif)
    # L2_combined = combine_leakage("021123_1506", combif)
    #
    # sr_curve = np.zeros(len(Na))
    # for n_profiling in NP:
    #     for ni, n_attack in tqdm(enumerate(Na), total=len(Na)):
    #         res = 0
    #         for i in trange(100, desc="REP|"):
    #             print_centered(f"====={n_profiling} {n_attack} rep {i}=====")
    #             res +=classify(L1_combined, L2_combined, n_profiling, n_attack)
    #         print_centered(f"====={n_profiling} {n_attack} SR {res}=====")
    #         sr_curve[ni] = res
    #     with open(f"SR_{combif}_{n_profiling}.npy", "wb") as f:
    #         np.save(f, Na)
    #         np.save(f, sr_curve)
    #     plt.plot(Na, sr_curve, label=f"{combif}")
