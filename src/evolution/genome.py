import numpy as np
from dataclasses import dataclass, field


@dataclass
class ConnectomeGenome:
    pn_kc: np.ndarray
    kc_mbon: np.ndarray
    kc_apl: np.ndarray
    apl_kc: np.ndarray
    kc_thresholds: np.ndarray
    kc_tau_m: np.ndarray

    @staticmethod
    def random(num_pn, num_kc, num_mbon, kc_pn_k=6, seed=42):
        rng = np.random.default_rng(seed)

        pn_kc = np.zeros((num_pn, num_kc), dtype=np.float64)
        for kc_idx in range(num_kc):
            chosen = rng.choice(num_pn, size=kc_pn_k, replace=False)
            pn_kc[chosen, kc_idx] = rng.uniform(0.3, 1.0, size=kc_pn_k)

        kc_mbon = rng.uniform(0.1, 0.5, size=(num_kc, num_mbon))
        kc_apl = np.ones(num_kc) * 0.5
        apl_kc = np.ones(num_kc) * -1.0
        kc_thresholds = np.full(num_kc, -0.050)
        kc_tau_m = np.full(num_kc, 0.020)

        return ConnectomeGenome(
            pn_kc=pn_kc,
            kc_mbon=kc_mbon,
            kc_apl=kc_apl,
            apl_kc=apl_kc,
            kc_thresholds=kc_thresholds,
            kc_tau_m=kc_tau_m,
        )

    @property
    def num_pn(self):
        return self.pn_kc.shape[0]

    @property
    def num_kc(self):
        return self.pn_kc.shape[1]

    @property
    def num_mbon(self):
        return self.kc_mbon.shape[1]

    @property
    def num_synapses(self):
        return int(np.count_nonzero(self.pn_kc) + np.count_nonzero(self.kc_mbon))

    def copy(self):
        return ConnectomeGenome(
            pn_kc=self.pn_kc.copy(),
            kc_mbon=self.kc_mbon.copy(),
            kc_apl=self.kc_apl.copy(),
            apl_kc=self.apl_kc.copy(),
            kc_thresholds=self.kc_thresholds.copy(),
            kc_tau_m=self.kc_tau_m.copy(),
        )

    def to_connectome_dict(self):
        return {
            "pn_kc": self.pn_kc,
            "kc_mbon": self.kc_mbon,
            "kc_apl": self.kc_apl,
            "apl_kc": self.apl_kc,
            "num_pn": self.num_pn,
            "num_kc": self.num_kc,
            "num_mbon": self.num_mbon,
        }
