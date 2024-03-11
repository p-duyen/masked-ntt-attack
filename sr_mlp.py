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
def H_matrix(p, labels):
    S = np.array([0, 1, 2, 3, 4])
    P_S = np.array([0.375, 0.25, 0.0625, 0.0625, 0.25])
    H = np.zeros((5, 5))
    for i, s in enumerate(S):
        correct_key_traces = (labels==s)
        ns = len(correct_key_traces)
        PIM_s = []
        for j, s in enumerate(S):
            second_term = p[correct_key_traces, j]
            PIM_s.append(second_term.mean())
        H[i] = PIM_s.copy()
    return H

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
    train_batch = L_train.shape[0]//5 if len(L_train)<10000 else L_train.shape[0]//50
    val_batch = s_val.shape[0]//20

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
        pred_data = DataLoader(valset, batch_size=L_val.shape[0],num_workers=10, shuffle=False)
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
                                    max_epochs=1000,
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
def mlp_sasca_sr(leakage_handler, traces, labels, n_profiling, n_val, attack_model):
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
            p_s_l[chki*chunk_len:(chki*chunk_len+chunk_len)] = sasca(chunk, m_flag)
    else:

        p_s_l = sasca(pdfs, m_flag)

    ce_hi = CE_hi(p_s_l)
    # Prepare secret labels
    s_val = labels[val_idx, 2]
    s_val = fix_s(s_val)
    s_val = (s_val%5).astype(np.int8)
    ce_pi = CE_pi(p_s_l, s_val)
    entS = ent_s()
    print_centered(f"Model  PI {ent_s() - CE_pi(p_s_l, s_val)}")
    H = H_matrix(p_s_l, s_val)
    for h in H:
        print(h.tolist())

if __name__ == '__main__':
    leakage_handler = Leakage_Handler("260923_1840")
    leakage_handler.n_files = 200
    leakage_handler.get_PoI(n_pois=50, mode="on_shares", model=ID)
    n_shares = leakage_handler.n_shares
    traces, labels = leakage_handler.get_data("full_trace")
    labels = labels.astype(np.int16)

    n_val = 300000

    mlp_sasca_sr(leakage_handler, traces, labels, 1000000, n_val, attack_model=ID)
    leakage_handler = Leakage_Handler("260923_1752")
    leakage_handler.n_files = 200
    leakage_handler.get_PoI(n_pois=50, mode="on_shares", model=ID)
    n_shares = leakage_handler.n_shares
    traces, labels = leakage_handler.get_data("full_trace")
    labels = labels.astype(np.int16)

    n_val = 300000

    mlp_sasca_sr(leakage_handler, traces, labels, 1000000, n_val, attack_model=ID)
