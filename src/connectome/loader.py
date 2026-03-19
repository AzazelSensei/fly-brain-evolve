import numpy as np


def generate_synthetic_connectome(config, seed=42):
    rng = np.random.default_rng(seed)

    num_pn = config["num_pn"]
    num_kc = config["num_kc"]
    num_mbon = config["num_mbon"]
    kc_pn_k = config["kc_pn_connections"]

    pn_kc = np.zeros((num_pn, num_kc), dtype=np.float64)
    for kc_idx in range(num_kc):
        chosen_pns = rng.choice(num_pn, size=kc_pn_k, replace=False)
        pn_kc[chosen_pns, kc_idx] = rng.uniform(0.3, 1.0, size=kc_pn_k)

    kc_mbon = rng.uniform(0.1, 0.5, size=(num_kc, num_mbon))

    kc_apl = np.ones(num_kc, dtype=np.float64) * 0.5
    apl_kc = np.ones(num_kc, dtype=np.float64) * -1.0

    return {
        "pn_kc": pn_kc,
        "kc_mbon": kc_mbon,
        "kc_apl": kc_apl,
        "apl_kc": apl_kc,
        "num_pn": num_pn,
        "num_kc": num_kc,
        "num_mbon": num_mbon,
    }


def save_connectome(connectome, path):
    np.savez(path, **connectome)


def load_connectome(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}
