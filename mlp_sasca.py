import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from scalib.metrics import SNR
from scalib.attacks import SASCAGraph, FactorGraph, BPState
import gc
from time import time
from tqdm import trange
from mlp_arch import *
from helpers import *

log_dir = "mlp_sasca_log"

S_SET = np.array([0, 1, -1, 2, -2])
PS = np.array([ 0.375,  0.25, 0.25, 0.0625, 0.0625])
def ce__(pSL, labels, s_set, ps):
    ce = 0
    for s_i, s in enumerate(s_set):
        Ls = np.where(labels==s)[0]
        # print_centered(f"==MI EST {s} {len(Ls)}===")
        if len(Ls)!=0:
            nLs = len(Ls)
            s_idx = np.where(s_set==s)[0][0]
            # print(pSL[Ls, s_i])
            psLs = np.log2(pSL[Ls, s_i])
            ce += ps[s_i]*np.nanmean(psLs)
    return -ce
def share_model(d_name, L_train, L_val, s_train, s_val, share_i, attack_model=ID, get_grad=False):
    """Build share's model (p(x_i|l_i)) from L_train traces and get predictions for L_val, s_val is used to estimate PI for p(x_i|l_i)
    First train to get the best model then use the best model to predict
    If pre-trained model existed then used it to predict
    Best model: min NLL on evaluation set.

    """

    if attack_model is HW:
        s_train = HW(s_train)
        s_val = HW(s_val)

    # set batch size for training and evaluating
    train_batch = L_train.shape[0]//10
    val_batch = s_val.shape[0]//5

    # prepare validation set
    valset = LeakageData(L_val, s_val)
    valdata = DataLoader(valset, batch_size=val_batch, num_workers=20, shuffle=False)

    m_desc = "HW" if attack_model is HW else "ID"
    n_train, trace_len = L_train.shape
    if attack_model is HW:
        chkpt_dir_path = f"{log_dir}/ENC_{m_desc}_{d_name}_{n_train}_{trace_len}_checkpoint"
    else:
        chkpt_dir_path = f"{log_dir}/ENC_{d_name}_{n_train}_{trace_len}_checkpoint"

    filename=f"share_{share_i}"
    out_dim = KYBER_Q if attack_model is ID else HW_Q
    mlp = MLP(in_dim=L_train.shape[-1], out_dim=out_dim, get_grad=get_grad)


    if os.path.exists(f"{chkpt_dir_path}/{filename}.ckpt"): #pre-trained model existed
        print_centered(f"=======================PRE-TRAINED MODEL FOR {d_name} {len(L_train)} share {share_i} IS AVAIL================")
        print_centered("PREDICT RIGHTAWAY")
        saved_model = mlp.load_from_checkpoint(f"{chkpt_dir_path}/{filename}.ckpt", in_dim=L_train.shape[-1], out_dim=out_dim)
        saved_model.eval()
        pred_data = DataLoader(valset, batch_size=L_val.shape[0], num_workers=10, shuffle=False)
        trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                max_epochs=1000,
                                detect_anomaly=True)
        pdf = trainer.predict(saved_model, pred_data)

        return pdf
    else: #train from scratch to find best model then use best model to predict
        print_centered(f"=======================PRE-TRAINED MODEL FOR {d_name} {len(L_train)} share {share_i} IS NOT AVAIL=================")
        print_centered("TRAINING START SOON")
        trainset = LeakageData(L_train, s_train)
        traindata = DataLoader(trainset, batch_size=train_batch, num_workers=50, shuffle=True)
        if not get_grad:
            early_stop_callback = EarlyStopping_(monitor="val_loss", patience=10, mode="min")
            pi_log_callback = PICallback()
            chkpt_callback = ModelCheckpoint_(save_top_k=1, monitor="val_loss", mode="min", dirpath=chkpt_dir_path, filename=filename)
            trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                    max_epochs=200,
                                    check_val_every_n_epoch=2,
                                    callbacks=[early_stop_callback, pi_log_callback, chkpt_callback],
                                    detect_anomaly=True)
            trainer.fit(mlp, traindata, valdata)
        else:
            early_stop_callback = EarlyStopping_(monitor="val_loss", patience=10, mode="min")
            pi_log_callback = PICallback()
            grad_log_callback = GetGrad()
            chkpt_callback = ModelCheckpoint_(save_top_k=1, monitor="val_loss", mode="min", dirpath=chkpt_dir_path, filename=filename)
            trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                max_epochs=1000,
                                check_val_every_n_epoch=2,
                                callbacks=[early_stop_callback, pi_log_callback, chkpt_callback, grad_log_callback],
                                detect_anomaly=True)
            trainer.fit(mlp, traindata, valdata)
            grad_log = trainer.callbacks[-1].GRAD
            with open(f"{log_dir}/ENC_{d_name}_{n_train}_{trace_len}_GRAD.npy", "wb") as f_g:
                np.save(f_g, grad_log)
            plt.plot(grad_log)
            plt.show()
            exit()
        print_centered("====================TRAINING DONE!====================")
        print_centered("PREDICT USING BEST MODEL")
        # predict w the best model
        saved_model = mlp.load_from_checkpoint(f"{chkpt_dir_path}/{filename}.ckpt", in_dim=L_train.shape[-1], out_dim=out_dim)
        saved_model.eval()
        pred_data = DataLoader(valset, batch_size=L_val.shape[0],num_workers=10, shuffle=False)
        trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto",
                                max_epochs=1000,
                                detect_anomaly=True)
        pdf = trainer.predict(saved_model, pred_data) # predict faster on several batchs

        return pdf
def mlp_sasca(leakage_handler, traces, labels, n_profiling, n_val, attack_model):
    """Estimate PI of model built from n_profiling profiling trace
    traces: total tracec available
    labels: data (shares values, secret values)
    n_profiling: number of profiling traces
    n_val: number of traces that secret model is estimated

    =========
    Return: PI of p(s|l) resulted by SASCA where p(x_i|l_i) is built by MLP


    """
    total_n_traces = leakage_handler.n_files*leakage_handler.traces_per_file
    n_pois = leakage_handler.n_pois
    PoI = leakage_handler.PoI.reshape(leakage_handler.n_shares, n_pois)
    pdfs = []


    # shares' models built on different sets of traces (secret model built on the same traces)
    # train_idx, val_idx = under_sample(total_n_traces, n_profiling, labels, leakage_handler.n_shares, n_val=n_val)

    # shares' models built on first n_profiling traces, and secret model is built on last n_val traces
    train_idx = [np.arange(0, n_profiling), np.arange(0, n_profiling)]
    val_idx = np.arange(total_n_traces - n_val, total_n_traces)

    pi_shares = []

    for share_i in trange(leakage_handler.n_shares, desc="PDFSHARE", leave=False):
        traces_i = traces[:, PoI[share_i]]
        Li_train = traces_i[train_idx[share_i]]
        si_train = labels[train_idx[share_i], share_i]
        L_val = traces_i[val_idx]
        si_val = labels[val_idx, share_i]
        res = share_model(leakage_handler.d_name, Li_train, L_val, si_train, si_val, share_i, attack_model)

        pdf_share = res[0].numpy(force=True)
        for i in range(1, len(res)): # accumulate results from several batchs
            pdf_share = np.vstack((pdf_share, res[i].numpy(force=True)))
        # pi_share = np.log2(KYBER_Q) - CE_pi(pdf_share, si_val)
        if attack_model is HW:
            pdf_share = pdf_share[:, HW(Zq)]
            pdf_share = pdf_share/pdf_share.sum(axis=1, keepdims=True)
        pi_share = np.log2(KYBER_Q) - CE_pi(pdf_share, si_val)
        print_centered(f"PI SHARE {pi_share} {np.log2(KYBER_Q) - CE_hi(pdf_share)}")
        pi_shares.append(pi_share)
        pdfs.append(pdf_share.copy())

    # integrate to graph
    # graph_desc = gen_graph(n_shares=leakage_handler.n_shares, q=KYBER_Q)
    # graph = FactorGraph(graph_desc)
    # bp = BPState(graph, n_val)
    # for i in range(leakage_handler.n_shares):
    #     if leakage_handler.m_flag==10:
    #         bp.set_evidence(f"S{i}", pdfs[i].astype(np.float64))
    #     else:
    #         reverse_idx = (KYBER_Q - np.arange(KYBER_Q))%KYBER_Q
    #         pdf_si = pdfs[i] if i== leakage_handler.n_shares-1 else pdfs[i][:, reverse_idx]
    #         bp.set_evidence(f"S{i}", pdf_si.astype(np.float64))
    #
    # bp.bp_acyclic("S")
    # resS = bp.get_distribution("S")
    #
    # # CBD and normalize
    # pr_s = resS[:, s_range]*prior_s
    # pr_s = pr_s/pr_s.sum(axis=1, keepdims=True)


    # SASCA on small range
    # pr_s = sasca(pdfs, m_flag=leakage_handler.m_flag)
    m_flag = leakage_handler.m_flag
    p_s_l = np.zeros((n_val, 5))
    pdfs = np.array(pdfs)
    if n_val >= 500000:
        _chunks = np.split(pdfs, 2, axis=1)
        chunk_len = n_val//2
        for chki, chunk in tqdm(enumerate(_chunks), total=2, desc="SASCA CHUNK|"):
            p_s_l[chki*chunk_len:(chki*chunk_len+chunk_len)] = convo(chunk, m_flag, S_SET, PS, KYBER_Q)
    else:

        p_s_l = convo(pdfs, m_flag, S_SET, PS, KYBER_Q)

    ce_hi = CE_hi(p_s_l)
    # Prepare secret labels
    s_val = labels[val_idx, 2]
    s_val = fix_s(s_val)
    # s_val = (s_val%5).astype(np.int8)
    # ce_pi = CE_pi(p_s_l, s_val)
    ce_pi = ce__(p_s_l, s_val, S_SET, PS)
    entS = ent_s()
    return entS - ce_pi, entS - ce_hi, pi_shares


def pi_curve_wn(d_name, N_TRAIN, n_pois=50, attack_model=ID):
    leakage_handler = Leakage_Handler(d_name)
    leakage_handler.n_files = 200
    leakage_handler.get_PoI(n_pois=n_pois, mode="on_shares", model=ID)
    n_shares = leakage_handler.n_shares
    traces, labels = leakage_handler.get_data("full_trace")
    labels = labels.astype(np.int16)
    pi_holder = np.zeros(len(N_TRAIN))
    hi_holder = np.zeros(len(N_TRAIN))
    pi_shares_holder = np.zeros((2, len(N_TRAIN)))
    ns_pbar = tqdm(enumerate(N_TRAIN), total=len(N_TRAIN))
    n_val = 200000
    for i_n, n_p in ns_pbar:
        ns_pbar.set_description_str(f"{d_name} N_TRACES: {n_p}")
        pi, hi, pi_shares = mlp_sasca(leakage_handler, traces, labels, n_p, n_val, attack_model=attack_model)
        ns_pbar.set_postfix_str(f"PI {pi:0.4f} HI {hi:0.4f}")
        print_centered(f"===============PI {pi:0.4f} HI {hi:0.4f} PI SHARES {pi_shares}================")
        pi_holder[i_n] = pi
        hi_holder[i_n] = hi
        pi_shares_holder[0, i_n] = pi_shares[0]
        pi_shares_holder[1, i_n] = pi_shares[1]
    return pi_holder, hi_holder, pi_shares_holder
def exp_run(attack_model):
    N_PROFILING = [3000, 10000, 50000, 100000, 500000, 700000, 1000000, 1500000]
    # N_PROFILING = [1000000, 1500000]
    # N_VAL = [100000, 250000, 300000, 500000]
    DNAMES = ["260923_1752", "260923_1840"]
    NPOIS = [50]
    model_desc = "HW" if attack_model is HW else "ID"
    for dname in DNAMES:
        for t, npois in enumerate(NPOIS):
            print_centered(f"========{dname} {npois} {model_desc}=======")
            pi_curve,  hi_curve, pi_shares_curve= pi_curve_wn(dname, N_PROFILING, n_pois=npois, attack_model=attack_model)
            print_centered(f"{pi_curve}")
            print_centered(f"{hi_curve}")
            print_centered(f"{pi_shares_curve}")
            plt.plot(N_PROFILING, pi_curve, label=f"{dname} {npois} {model_desc}")
    #         with open(f"{log_dir}/{dname}_{npois}_{model_desc}_rerun.npy", "wb") as f:
    #             np.save(f, N_PROFILING)
    #             np.save(f, pi_curve)
    # #             np.save(f, hi_curve)
    #             np.save(f, pi_shares_curve)
    # plt.legend()
    # plt.show()
if __name__ == '__main__':
    exp_run(attack_model=ID)
    # exp_run(attack_model=HW)
    # N_PROFILING = [5000, 10000, 50000, 100000, 300000, 500000, 700000, 1000000]
    # # N_PROFILING = [50000]
    # folder = "260923_1752"
    # lh = Leakage_Handler(folder)
    # lh.n_files = 100
    # traces, data = lh.get_data("full_trace")
    # share_i = 0
    # labels = data[:, [share_i]].squeeze()
    # L_train, L_val, s_train, s_val = train_test_split(traces, labels, test_size=0.3)
    # _ = share_model(folder, L_train, L_val, s_train, s_val, share_i, attack_model=HW, get_grad=True)
    # pi_curve,  hi_curve, pi_shares_curve = pi_curve_wn(folder, N_PROFILING, n_pois=20, attack_model=ID)
    # with open(f"{log_dir}/pi_hi_{folder}_ID.npy", "wb") as f:
    #     np.save(f, N_PROFILING)
    #     np.save(f, pi_curve)
    #     np.save(f, hi_curve)
    # plt.plot(N_PROFILING, pi_curve, label="pi SUB", color=COLOR_NAMES["hotpink"])
    # plt.plot(N_PROFILING, hi_curve, label="hi SUB", color=COLOR_NAMES["hotpink"], linestyle="dashed")
    # with open(f"{log_dir}/pi_shares_{folder}_ID.npy", "wb") as f:
    #     np.save(f, pi_shares_curve)
    #
    #
    # #====================================================
    # folder = "260923_1752"
    # pi_curve,  hi_curve, pi_shares_curve_ = pi_curve_wn(folder, N_PROFILING, n_pois=20, attack_model=ID)
    # with open(f"{log_dir}/pi_hi_{folder}_ID.npy", "wb") as f:
    #     np.save(f, N_PROFILING)
    #     np.save(f, pi_curve)
    #     np.save(f, hi_curve)
    # with open(f"{log_dir}/pi_shares_{folder}_ID.npy", "wb") as f:
    #     np.save(f, pi_shares_curve_)
    # plt.plot(N_PROFILING, pi_curve, label="pi ADD", color=COLOR_NAMES["salmon"])
    # plt.plot(N_PROFILING, hi_curve, label="hi ADD", color=COLOR_NAMES["salmon"], linestyle="dashed")
    #
    # plt.legend()
    # plt.show()
    #
    # plt.plot(pi_shares_curve.T, label="SUB")
    # plt.plot(pi_shares_curve_.T, label="ADD")
    # plt.legend()
    # plt.show()
    # N_TRAIN = [4000, 10000, 25000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 750000, 1000000, 1500000, 1600000, 1750000]
    #
    # pi = exp_run("280623_1200")
    # plt.plot(N_TRAIN, pi, label="sub")
    # pi = exp_run("280623_1226")
    # plt.plot(N_TRAIN, pi, label="add")
    # plt.show()
    # N_TRAIN = [800000, 900000, 1000000, 1250000]
    # pi = pi_curve_mlp_wn("280623_1200", N_TRAIN, n_pois=50, model=ID, mode="on_shares")
    # with open(f"mlp_shares_280623_1200_poi50.npy", "wb") as f:
    #     np.save(f, N_TRAIN)
    #     np.save(f, pi)
    # plt.plot(N_TRAIN, pi, label=f"S=X1+X2")
    # pi = pi_curve_mlp_wn("280623_1226", N_TRAIN, n_pois=50, model=ID, mode="on_shares")
    # with open(f"mlp_shares_280623_1226_poi50.npy", "wb") as f:
    #     np.save(f, N_TRAIN)
    #     np.save(f, pi)
    # plt.plot(N_TRAIN, pi, label=f"S=X2-X1")
    # plt.legend()
    # plt.show()
    # x = np.array([1, 2, 3328, 3327, 0])
    # y = (x - 3329)
    # print(y)
    # print_centered(f"==========MLP_SASCA PROCESS ID MODEL 2 SHARES===============")
    # pi_sub = mlp_shares("280623_1200", model=ID)
    # pi_add = mlp_shares("280623_1226", model=ID)
    # print_centered("===============MLPSASCA 2 shares share_val model==============")
    # print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # print_centered("==================================")
    #pi_sub = mlp_sasca("200623_1211")
    #pi_add = mlp_sasca("200623_1225")
    #print_centered("===============MLPSASCA 3 shares share_val model==============")
    #print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    #print_centered("==================================")
    # print_centered(f"==========MLP_SASCA PROCESS ID MODEL 2 SHARES===============")
    # pi_sub = mlp_shares("280623_1200", model=HW)
    # pi_add = mlp_shares("280623_1226", model=HW)
    # print_centered("===============MLPSASCA 2 shares HW model==============")
    # print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    # print_centered("==================================")
    #pi_sub = mlp_sasca("200623_1211", model="HW")
    #pi_add = mlp_sasca("200623_1225", model="HW")
    #print_centered("===============MLPSASCA 3 shares HW model==============")
    #print_centered(f"= SUB: {pi_sub}|ADD: {pi_add}|GAP: {pi_add/pi_sub} =")
    #print_centered("==================================")

    # y_leakage = np.random.rand(2, 256) # this might come from an LDA
    # print(y_leakage.shape)
    # y_leakage = y_leakage / y_leakage.sum(axis=1, keepdims=True)
    # print(y_leakage.shape, y_leakage.sum(axis=1))

    # share_model("190623_1600", 10, 0)
