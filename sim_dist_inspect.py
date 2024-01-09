import numpy as np
from matplotlib import pyplot as plt
from Crypto.Cipher import AES
from helpers import *
from LDA_func import *
AES_N_BLOCKS = 16

def gen_inputs_fixkey(secret_seed):

    # Key gen
    n_coeffs = KYBER_N
    secrets = np.zeros((KYBER_N, ), dtype=np.int16)
    prg = AES.new(key=secret_seed.tobytes(), mode=AES.MODE_ECB)
    dummy_msg = np.zeros(AES_N_BLOCKS, dtype=np.uint8).tobytes()
    i_count = 0
    while i_count < KYBER_N:
        cipher_bytes = prg.encrypt(dummy_msg)
        dummy_msg = cipher_bytes
        for cbyte in cipher_bytes:
            bin_cbyte = np.array(list(np.binary_repr(cbyte, 8)), dtype=np.int8)
            cbyte_chunks = np.array_split(bin_cbyte, 2)
            for chunk in cbyte_chunks:
                secrets[i_count] = int(chunk[0]-chunk[1]+chunk[2]-chunk[3])
                i_count += 1
                if i_count == KYBER_N:
                    break
            if i_count == KYBER_N:
                break
    return secrets
def gen_shares(secrets, n_shares=2, q=KYBER_Q, mode="sub"):
    shares = {}
    masked_state = secrets.copy()
    for i in range(n_shares-1):
        tmp = np.random.randint(0, q, size=secrets.shape, dtype=np.int32)
        if mode=="sub":
            masked_state = (masked_state-tmp)%q
        elif mode=="add":
            masked_state = (masked_state+tmp)%q
        elif mode=="add_one":
            masked_state = (masked_state+tmp)%q if i==0 else (masked_state-tmp)%q
        shares[f"S{i}"] = tmp.copy()
    shares[f"S{n_shares-1}"] = masked_state.copy()
    return shares

def gen_leakages(shares, sigma, model):
    L = {}
    c = 0
    for share, share_val in shares.items():
        noise = np.random.normal(0, sigma, size=shares["S0"].shape)
        if isinstance(sigma, float):
            L[share] = model(share_val) + noise*(c+1)
            noise = -noise
            c += 1
            # noise = -noise +  np.random.normal(0, sigma, size=shares["S0"].shape)
        else:
            L[share] = model(share_val) + np.random.normal(0, sigma[c], size=share_val.shape)
            c += 1
    return L



def combine_L(combif, leakage):
    total_N, n_coeff = leakage["S0"].shape
    n_shares = len(leakage.keys())
    if combif in ["abs_diff", "sum"]:
        combined_L = np.zeros((total_N, n_coeff))
    elif combif in ["prod", "norm_prod"]:
        combined_L = np.ones((total_N, n_coeff))
    for share_i in range(n_shares):
        Li = leakage[f"S{share_i}"]
        if combif=="abs_diff":
            combined_L =  Li - combined_L
        elif combif=="sum":
            combined_L =  Li + combined_L
        elif combif=="prod":
            combined_L = combined_L*Li
        elif combif=="norm_prod":
            Li = Li - Li.mean(axis=0)
            combined_L = combined_L*Li
    if combif=="abs_diff":
        return np.abs(combined_L)
    return combined_L

def sim_test(combif):
    plt.figure(figsize=(16, 12))
    N = 50000
    sigma = 0.1
    n_shares = 2
    secret_seed = np.array([118, 141, 240,  66,  13,  35, 177, 141, 119, 181, 191, 217, 182, 190, 167, 114], dtype=np.uint8) #021123_1335
    sec1 = gen_inputs_fixkey(secret_seed)
    sec1 = np.repeat([sec1], N, axis=0)

    secret_seed = np.array([ 53, 140, 144, 103,  26, 193, 106,  32, 249,  53,  39,  38, 199, 173, 154, 140], dtype=np.uint8) #021123_1506
    sec2 = gen_inputs_fixkey(secret_seed)
    sec2 = np.repeat([sec2], N, axis=0)
    shares_s1 = gen_shares(sec1, n_shares)
    shares_s2 = gen_shares(sec2, n_shares)
    L_s1 = gen_leakages(shares_s1, sigma=sigma, model=HW)
    L_s2 = gen_leakages(shares_s2, sigma=sigma, model=HW)
    combined_L_s1 = combine_L(combif, L_s1)
    combined_L_s2 = combine_L(combif, L_s2)


    L = np.append(combined_L_s1, combined_L_s2, axis=0)
    label = np.zeros(2*N, dtype=np.int8)
    label[:N] = 1
    Sb = scatter_between(L, label)
    Sw = scatter_within(L, label)
    mus = mean_vec(L, label)


    delta = mus[0] - mus[1]
    inv_Sw = np.linalg.inv(Sw)
    m_distance = np.transpose(delta).dot(inv_Sw)
    m_distance = m_distance.dot(delta)
    print_centered(f"Mahalanobi distance: {m_distance}")

    V = project_vector(Sw, Sb)
    Y = L.dot(V)
    Y0 = Y[label==0]
    Y1 = Y[label==1]
    m1 = Y0.mean()
    m2 = Y1.mean()
    print_centered(f"mean 1 {m1} mean 2 {m2} mean_distance = {np.abs(m1-m2)}")
    # print(m1, m2)
    mus = np.array([m1, m2])
    m = (m1+m2)/2

    p0, x0 = np.histogram(Y0, bins=100, density=True)
    plt.plot(x0[:-1], p0, label="class 1 LDA projection", alpha=0.75, linewidth=2)
    p1, x1 = np.histogram(Y1, bins=100, density=True)
    plt.plot(x1[:-1], p1, label="class 2 LDA projection", alpha=0.75, linewidth=2)


    p0_mx = p0.max()
    p1_mx = p1.max()


    plt.hlines(y=p1_mx/2, xmin=mus.min(), xmax=mus.max(), colors="tab:red")
    plt.text(x=m, y=p1_mx/2, s=f"{np.abs(m1-m2):0.4f}" )
    pooled_cov = scatter_within(Y, label)
    pooled_cov = np.sqrt(pooled_cov)
    # pooled_cov = np.power(pooled_cov, 2)
    mu_projected = mean_vec(Y, label)
    print(mu_projected, pooled_cov)
    gp_0 = pdf_normal(np.sort(Y0, axis=0), mu_projected[0], pooled_cov)
    gp_1 = pdf_normal(np.sort(Y1, axis=0), mu_projected[1], pooled_cov)
    plt.vlines(x=m1, ymin=0, ymax=gp_0.max(), color="tab:grey")
    plt.text(x=m1, y=-0.001, s=f"{m1:0.4f}" , fontsize=10)
    plt.vlines(x=m2, ymin=0, ymax=gp_1.max(), color="tab:grey")
    plt.text(x=m2, y=-0.001, s=f"{m2:0.4f}" , fontsize=10)
    plt.plot(np.sort(Y0, axis=0), gp_0, label="class 1 Gaussian estimation", linewidth=2, color="tab:blue")
    plt.plot(np.sort(Y1, axis=0), gp_1, label="class 2 Gaussian estimation", linewidth=2, color="tab:orange")
    #
    plt.legend(fontsize=14)
    plt.title(f"{n_shares} share {combif} both wings sigma {sigma}")
    plt.savefig(f"figures/LDA_projection_sim_fullwings_{n_shares}shares_{combif}_sigma_{sigma}_{N}_.png", bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    combifs = ["sum", "abs_diff", "prod", "norm_prod"]
    for cf in combifs:
        sim_test(cf)
