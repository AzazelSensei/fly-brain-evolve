import numpy as np
from src.evolution.genome import ConnectomeGenome


def crossover(parent_a, parent_b, seed=None):
    rng = np.random.default_rng(seed)

    mask_kc = rng.random(parent_a.num_kc) < 0.5

    pn_kc = np.where(mask_kc[np.newaxis, :], parent_a.pn_kc, parent_b.pn_kc)

    kc_mbon = np.where(mask_kc[:, np.newaxis], parent_a.kc_mbon, parent_b.kc_mbon)

    kc_apl = np.where(mask_kc, parent_a.kc_apl, parent_b.kc_apl)
    apl_kc = np.where(mask_kc, parent_a.apl_kc, parent_b.apl_kc)
    kc_thresholds = np.where(mask_kc, parent_a.kc_thresholds, parent_b.kc_thresholds)
    kc_tau_m = np.where(mask_kc, parent_a.kc_tau_m, parent_b.kc_tau_m)

    return ConnectomeGenome(
        pn_kc=pn_kc.copy(),
        kc_mbon=kc_mbon.copy(),
        kc_apl=kc_apl.copy(),
        apl_kc=apl_kc.copy(),
        kc_thresholds=kc_thresholds.copy(),
        kc_tau_m=kc_tau_m.copy(),
    )
