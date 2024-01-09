import pickle5 as pickle
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 22})
from scalib.metrics import SNR
from tqdm.auto import trange, tqdm
import os

import matplotlib.colors as mcolors
TAB_COLORS =list(mcolors.TABLEAU_COLORS.keys())
CSS_COLORS = list(mcolors.CSS4_COLORS.keys())

XKCD_COLORS = list(mcolors.XKCD_COLORS.keys())

S_SET = np.array([0, 1, -1, 2, -2])
PS = np.array([ 0.375,  0.25, 0.25, 0.0625, 0.0625])
KYBER_Q = 3329
KYBER_N = 256
KYBER_N_2 = 128
HW_Q = 12

try:
    width = os.get_terminal_size().columns
except:
    width = 1
def print_centered(str):
    print(str.center(width))
def count_1(x):
    return int(x).bit_count()
fcount = np.vectorize(count_1)
def HW(x):
    return fcount(x).astype(np.uint16)

def ID(x):
    return x
def pdf_normal(x, mu, sigma):
    ep = (x-mu)/sigma
    ep = -ep**2/2
    return np.exp(ep) / (sigma * np.sqrt(2*np.pi))


def get_info_from_log(dir_path):
    f_log = f"{dir_path}/log"
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

# def fix_s(secrets):
#     sec_ = secrets[0]
#     res = secrets.copy()
#     _ = np.where(sec_==3328)[0]
#     res[:, _] = 4
#     _ = np.where(sec_==3327)[0]
#     res[:, _] = 3
#     return res
class Leakage_Handler:
    def __init__(self, d_name):
        self.d_name = d_name
        if os.path.exists(f"/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/traces/{d_name}"):
            self.dir_path = f"/home/tpay/Desktop/WS/kyber_masked_measurements/Kyber_F415/Measurements/traces/{d_name}"

        else:
            self.dir_path = f"traces/{d_name}"

        info = get_info_from_log(self.dir_path)
        self.n_shares = info["n_shares"]
        self.n_samples = info["n_samples"]
        self.n_files = info["n_files"]
        self.poly_per_file = info["poly_per_batch"]*info["n_batchs"]
        self.traces_per_file = info["poly_per_batch"]*info["n_batchs"]*self.n_shares
        self.m_flag = info["m_flag"]
        self.fname = info["fname"]
        self.file_path = f"{self.dir_path}/{self.fname}_{self.n_shares}"
        self.poi = {}
    def get_snr_on_share(self, share_i, coeff_list, n_chunks, f_des=None, add_noise=False, model=ID):
        """ Get SNR where each share's trace is saved separately
        """
        model_d = "ID" if model is ID else "HW"
        snr_file = f"{self.dir_path}/SNR_on_share_{share_i}_{model_d}_{f_des}_noise_{add_noise}.pkl"
        if os.path.exists(snr_file):
            print_centered(f"SNR {share_i}_{model_d}_{f_des}_noise_{add_noise} FILE EXIST!")
        else:
            with open(snr_file, "wb") as f:
                coeff_chunks = np.split(coeff_list, n_chunks)
                coeff_proc_len = 1273
                for chunk_i, c_chunk in tqdm(enumerate(coeff_chunks), total=len(coeff_chunks), desc="COEFF CHUNK", colour="yellow"):
                    spl_chunk = c_chunk[0]%128
                    start_sample = 500 + coeff_proc_len*spl_chunk
                    subtrace = np.arange(start_sample, start_sample + coeff_proc_len*len(c_chunk))
                    if model is HW:
                        snr = SNR(HW_Q, len(subtrace), len(c_chunk), use_64bit=True)
                    elif model is ID:
                        snr = SNR(KYBER_Q, len(subtrace), len(c_chunk), use_64bit=True)
                    for fi in trange(self.n_files, desc="FILE|", colour="green", leave=False):
                        # print_centered(f"==SNR share {share_i} traces {(share_i+1)%2}")
                        f_t = f"{self.file_path}_traces_{fi}_share_{share_i}.npy"
                        f_d = f"{self.file_path}_meta_{fi}.npy"
                        traces = np.load(f_t).astype(np.int16)
                        traces = traces[:, subtrace].copy()
                        if add_noise:
                            traces = traces + np.random.normal(0, add_noise, size=traces.shape)
                            traces = traces.astype(np.int16)
                        data = np.load(f_d, allow_pickle=True)
                        data = data.item().get(f"share_{share_i}")[:, c_chunk]
                        # print(data)
                        data = model(data)
                        data = data.astype(np.uint16)
                        snr.fit_u(traces, data)
                    snr_val = snr.get_snr()
                    c_name = f"c_chunk_share_{share_i}_{chunk_i}"
                    st_name = f"subtrace_share_{share_i}_{chunk_i}"
                    snr_name = f"snr_{share_i}_{chunk_i}"
                    pickle.dump({c_name:c_chunk, st_name:subtrace, snr_name:snr_val}, f)

    def get_PoI_on_share(self, share_i, coeff_list, n_chunks, n_poi, f_des=None, add_noise=False, model=ID, display=False):
        model_d = "ID" if model is ID else "HW"
        snr_file = f"{self.dir_path}/SNR_on_share_{share_i}_{model_d}_{f_des}_noise_{add_noise}.pkl"
        self.poi[f"share_{share_i}"] = np.zeros((len(coeff_list), n_poi))
        if display:
            plt.figure(figsize=(12, 12))
        if os.path.exists(snr_file):
            print_centered(f"SNR {share_i}_{model_d}_{f_des}_noise_{add_noise} FILE EXIST!")
            with open(snr_file, "rb") as f:
                coeff_chunks = np.split(coeff_list, n_chunks)
                for chunk_i, c_chunk in tqdm(enumerate(coeff_chunks), total=len(coeff_chunks), desc="COEFF CHUNK", colour="blue"):
                    data = pickle.load(f)

                    c_name = f"c_chunk_share_{share_i}_{chunk_i}"
                    st_name = f"subtrace_share_{share_i}_{chunk_i}"
                    snr_name = f"snr_{share_i}_{chunk_i}"
                    _ = data[c_name]
                    subtrace = data[st_name]
                    snr = data[snr_name]
                    for i, c in enumerate(c_chunk):
                        poi = np.argsort(snr[i])[::-1][:n_poi]
                        self.poi[f"share_{share_i}"][chunk_i*len(c_chunk)+i] = subtrace[poi]

                        if display:
                            # print(c, subtrace[poi], snr[i][poi])
                            plt.plot(subtrace, snr[i], label=f"$c_{{{c}}}$")
                            plt.scatter(subtrace[poi], snr[i][poi])
        if display:
            plt.title(f"{f_des}")
            plt.legend(fontsize=18)
            plt.savefig(f"pic/SNR_{f_des}.png", bbox_inches="tight")
            plt.show()

    def get_snr_on_sec(self, share_i, model=ID, display=False):
        model_d = "ID" if model is ID else "HW"
        snr_file = f"{self.dir_path}/SNR_on_sec_share_{share_i}_traces_{model_d}.pkl"
        coeff_list = np.arange(KYBER_N)
        n_chunks = 16
        coeff_chunks = np.split(coeff_list, n_chunks)
        if os.path.exists(snr_file):
            print(snr_file)
            print_centered(f"SNR FILE EXIST!")
            if display:
                with open(snr_file, "rb") as f:
                    for chunk_i, c_chunk in tqdm(enumerate(coeff_chunks), total=len(coeff_chunks), desc="COEFF CHUNK", colour="blue"):
                        data = pickle.load(f)

                        c_name = f"c_chunk_share_{share_i}_{chunk_i}"
                        st_name = f"subtrace_share_{share_i}_{chunk_i}"
                        snr_name = f"snr_{share_i}_{chunk_i}"
                        _ = data[c_name]
                        subtrace = data[st_name]
                        snr = data[snr_name]
                        for i, c in enumerate(c_chunk):
                            plt.plot(subtrace, snr[i], label=f"$c_{{{c}}}$")
                        plt.title(f"SNR ON SEC from {c_chunk[0]} to {c_chunk[-1]} on traces of share {share_i+1}")
                        plt.legend()
                        plt.show()
        else:
            with open(snr_file, "wb") as f:
                coeff_proc_len = 1272
                for chunk_i, c_chunk in tqdm(enumerate(coeff_chunks), total=len(coeff_chunks), desc="COEFF CHUNK", colour="yellow"):
                    spl_chunk = c_chunk[0]%128
                    start_sample = 700 + coeff_proc_len*spl_chunk
                    subtrace = np.arange(start_sample, start_sample + coeff_proc_len*len(c_chunk))
                    snr = SNR(5, len(subtrace), len(c_chunk), use_64bit=True)
                    for fi in trange(self.n_files, desc="FILE|", leave=None, colour="green"):
                        f_t = f"{self.file_path}_traces_{fi+50}_share_{share_i}.npy"
                        f_d = f"{self.file_path}_meta_{fi+50}.npy"
                        traces = np.load(f_t).astype(np.int16)
                        traces = traces[:, subtrace].copy()

                        data = np.load(f_d, allow_pickle=True)
                        data = data.item().get("secret")[:, c_chunk]
                        data = fix_s(data)
                        data = model(data)
                        data = data.astype(np.uint16)
                        snr.fit_u(traces, data)
                    snr_val = snr.get_snr()
                    c_name = f"c_chunk_share_{share_i}_{chunk_i}"
                    st_name = f"subtrace_share_{share_i}_{chunk_i}"
                    snr_name = f"snr_{share_i}_{chunk_i}"
                    pickle.dump({c_name:c_chunk, st_name:subtrace, snr_name:snr_val}, f)
    def get_secrets(self):
        f_d = f"{self.file_path}_meta_0.npy"
        data = np.load(f_d, allow_pickle=True)
        if self.n_shares == 2:
            secrets = data.item().get("secret")[0]
        else:
            secrets = data.item().get("secret")[0]
        return secrets
def fix_s(sec, to_Q=False):
    sec_n = sec.copy()
    if to_Q:
        sec_n[sec_n==3327] = -2
        sec_n[sec_n==3328] = -1
    else:
        sec_n[sec_n==3328] = 4
        sec_n[sec_n==3327] = 3
    return sec_n

if __name__ == '__main__':


    # with open(f"{lh.dir_path}/{lh.fname}_meta_0.npy", "rb") as f:
    #
    #     data = np.load(f, allow_pickle=True)
    #     for i in range(lh.n_shares):
    #         share_i = data.item().get(f"share_{i}")
    #         print(i, share_i)
    # exit()
    #
    # lh.n_files=5
    # lh = Leakage_Handler("021123_1506")
    # with open(f"{lh.dir_path}/{lh.fname}_{lh.n_shares}_meta_0.npz", "rb") as f:
    #     data = np.load(f)["polynoms"]
    #     for i in range(lh.n_shares):
    #         idx_i = np.arange(i*KYBER_N, i*KYBER_N+KYBER_N)
    #         share_i = data[:, idx_i]
    #         print(i, share_i)
        # for i in range(lh.n_shares):
        #     share_i = data.item().get(f"share_{i}")
        #     print(i, share_i)
    lh = Leakage_Handler("021123_1335")
    lh.n_files=10


    # lh.get_snr_on_sec(share_i=0)
    # lh.get_snr_on_sec(share_i=0, display=True)
    # lh.get_snr_on_sec(share_i=1)
    # lh.get_snr_on_sec(share_i=1, display=True)
    # exit()
    # c_l = np.array([0, 1, 2, 3])
    c_l = np.array([128, 129, 130, 131])
    # c_l = np.array([384, 385, 386, 387, 388])
    n_c = 1
    lh.get_snr_on_share(0, coeff_list=c_l, n_chunks=n_c, f_des="left_wing_share1", add_noise=0, model=ID)
    lh.get_PoI_on_share(0, c_l, n_c, n_poi=1, f_des="left_wing_share1", add_noise=0, display=True, model=ID)
    lh.get_snr_on_share(1, coeff_list=c_l, n_chunks=n_c, f_des="left_wing_share2", add_noise=0, model=ID)
    lh.get_PoI_on_share(1, c_l, n_c, n_poi=1, f_des="left_wing_share2", add_noise=0, display=True, model=ID)

    # lh.n_files = 50
    # c_l = np.array([0, 1, 2, 3])
    # lh.get_snr_on_share(0, coeff_list=c_l, n_chunks=n_c, f_des="left_wing", add_noise=0, model=ID)
    # lh.get_PoI_on_share(0, c_l, n_c, n_poi=1, f_des="left_wing", add_noise=0, display=True, model=ID)
    # lh.get_PoI_on_share(0, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True, model=ID)
    #
    # lh.get_snr_on_share(1, coeff_list=c_l, n_chunks=n_c, f_des=None, add_noise=0, model=ID)
    # lh.get_PoI_on_share(1, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True, model=ID)
    # lh.get_PoI_on_share(1, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True, model=ID)
    # for i in range(6):
    #     lh.get_snr_on_share(i, coeff_list=c_l, n_chunks=n_c, f_des=None, add_noise=0, model=ID)
    #     lh.get_PoI_on_share(i, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True, model=ID)
    # lh.get_snr_on_share(2, coeff_list=c_l, n_chunks=n_c, f_des=None, add_noise=0)
    # lh.get_PoI_on_share(2, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True)
    #
    # lh.get_snr_on_share(3, coeff_list=c_l, n_chunks=n_c, f_des=None, add_noise=0)
    # lh.get_PoI_on_share(3, c_l, n_c, n_poi=1, f_des=None, add_noise=0, display=True)
    # lh.get_snr_on_shares(coeff_list=c_l, n_chunks=n_c, f_des=None, add_noise=None)
    # lh.get_PoI(coeff_list=c_l, n_chunks=n_c, n_poi=1, f_des="None_noise_None", display=True)

    # lh.get_snr_on_shares(coeff_list=c_l, n_chunks=n_c, f_des=f"{c_l[0]}_{c_l[-1]}", add_noise=500)
    # lh.get_PoI(coeff_list=c_l, n_chunks=n_c, n_poi=1, f_des=f"{c_l[0]}_{c_l[-1]}_noise_{500}", display=True)

    # lh.get_snr_on_shares(coeff_list=c_l, n_chunks=n_c, f_des=f"{c_l[0]}_{c_l[-1]}", add_noise=1000)
    # lh.get_PoI(coeff_list=c_l, n_chunks=n_c, n_poi=1, f_des=f"{c_l[0]}_{c_l[-1]}_noise_{1000}", display=True)

    # lh.get_snr_on_shares(coeff_list=c_l, n_chunks=n_c, f_des=f"{c_l[0]}_{c_l[-1]}", add_noise=2000)
    # lh.get_PoI(coeff_list=c_l, n_chunks=n_c, n_poi=1, f_des=f"{c_l[0]}_{c_l[-1]}_noise_{2000}", display=True)
