import numpy as np
from numba import njit, prange


@njit(cache=True)
def _run_network(
    v, g_exc, g_inh, refr,
    W_exc, W_inh,
    input_spikes,
    tau_m, V_rest, V_thresh, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, mbon_start, mbon_end, kc_start, kc_end,
    input_weight,
    num_pn,
):
    num_mbon = mbon_end - mbon_start
    num_kc = kc_end - kc_start
    mbon_counts = np.zeros(num_mbon, dtype=np.int32)
    kc_spiked = np.zeros(num_kc, dtype=np.int32)
    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)

    E_exc = np.full(num_neurons, E_exc_val)
    E_inh = np.full(num_neurons, E_inh_val)

    for t in range(num_steps):
        for pn in range(num_pn):
            if input_spikes[t, pn]:
                g_exc[pn] += input_weight

        I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v)
        dv = (-(v - V_rest) + I_syn / g_L) / tau_m * dt

        for i in range(num_neurons):
            if refr[i] > 0:
                refr[i] -= 1
            else:
                v[i] += dv[i]

        for i in range(num_neurons):
            if v[i] > V_thresh[i] and refr[i] == 0:
                v[i] = V_reset[i]
                refr[i] = refr_steps

                for j in range(num_neurons):
                    if W_exc[i, j] > 0:
                        g_exc[j] += W_exc[i, j]
                    if W_inh[i, j] > 0:
                        g_inh[j] += W_inh[i, j]

                if mbon_start <= i < mbon_end:
                    mbon_counts[i - mbon_start] += 1
                if kc_start <= i < kc_end:
                    kc_spiked[i - kc_start] = 1

        g_exc *= decay_exc
        g_inh *= decay_inh

    return mbon_counts, kc_spiked


@njit(parallel=True, cache=True)
def simulate_growing_batch(
    batch_W_exc, batch_W_inh, batch_input_spikes, batch_V_thresh,
    V_rest_arr, V_reset_arr, g_L_arr,
    E_exc_val, E_inh_val,
    tau_m_arr, tau_exc_val, tau_inh_val,
    dt, num_steps, refr_steps,
    num_neurons, mbon_start, mbon_end, kc_start, kc_end,
    input_weight, num_pn,
):
    batch_size = batch_W_exc.shape[0]
    num_mbon = mbon_end - mbon_start
    num_kc = kc_end - kc_start
    all_mbon = np.zeros((batch_size, num_mbon), dtype=np.int32)
    all_kc = np.zeros((batch_size, num_kc), dtype=np.int32)

    for b in prange(batch_size):
        v = V_rest_arr.copy()
        g_exc = np.zeros(num_neurons)
        g_inh = np.zeros(num_neurons)
        refr = np.zeros(num_neurons, dtype=np.int32)

        mc, ks = _run_network(
            v, g_exc, g_inh, refr,
            batch_W_exc[b], batch_W_inh[b],
            batch_input_spikes[b],
            tau_m_arr, V_rest_arr, batch_V_thresh[b], V_reset_arr, g_L_arr,
            E_exc_val, E_inh_val, tau_exc_val, tau_inh_val,
            dt, num_steps, refr_steps,
            num_neurons, mbon_start, mbon_end, kc_start, kc_end,
            input_weight, num_pn,
        )
        all_mbon[b] = mc
        all_kc[b] = ks

    return all_mbon, all_kc


class BrainConfig:
    def __init__(self, num_pn, num_kc, num_mbon, num_apl=1):
        self.num_pn = num_pn
        self.num_kc = num_kc
        self.num_mbon = num_mbon
        self.num_apl = num_apl

        self.pn_start = 0
        self.pn_end = num_pn
        self.kc_start = num_pn
        self.kc_end = num_pn + num_kc
        self.mbon_start = num_pn + num_kc
        self.mbon_end = num_pn + num_kc + num_mbon
        self.apl_start = num_pn + num_kc + num_mbon
        self.apl_end = num_pn + num_kc + num_mbon + num_apl
        self.num_neurons = num_pn + num_kc + num_mbon + num_apl

    def build_params(self):
        n = self.num_neurons
        tau_m = np.zeros(n)
        V_rest = np.full(n, -0.070)
        V_reset = np.full(n, -0.070)
        g_L = np.full(n, 25e-9)

        tau_m[self.pn_start:self.pn_end] = 0.010
        tau_m[self.kc_start:self.kc_end] = 0.020
        tau_m[self.mbon_start:self.mbon_end] = 0.015
        tau_m[self.apl_start:self.apl_end] = 0.010

        return tau_m, V_rest, V_reset, g_L


class GrowingGenome:
    def __init__(self, brain_config, pn_kc, kc_mbon, kc_thresh, kc_apl_w=2.0, apl_kc_w=200.0):
        self.config = brain_config
        self.pn_kc = pn_kc
        self.kc_mbon = kc_mbon
        self.kc_thresh = kc_thresh
        self.kc_apl_w = kc_apl_w
        self.apl_kc_w = apl_kc_w

    @staticmethod
    def random(brain_config, rng, kc_pn_k=6):
        pn_kc = np.zeros((brain_config.num_pn, brain_config.num_kc))
        for kc in range(brain_config.num_kc):
            chosen = rng.choice(brain_config.num_pn, size=min(kc_pn_k, brain_config.num_pn), replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=len(chosen))

        kc_mbon = rng.uniform(2.0, 10.0, size=(brain_config.num_kc, brain_config.num_mbon))
        kc_thresh = rng.uniform(-48, -42, size=brain_config.num_kc)
        return GrowingGenome(brain_config, pn_kc, kc_mbon, kc_thresh)

    @staticmethod
    def from_parent(parent_genome, new_config, rng):
        old = parent_genome
        new_pn = new_config.num_pn
        new_kc = new_config.num_kc
        new_mbon = new_config.num_mbon

        pn_kc = np.zeros((new_pn, new_kc))
        min_pn = min(old.pn_kc.shape[0], new_pn)
        min_kc = min(old.pn_kc.shape[1], new_kc)
        pn_kc[:min_pn, :min_kc] = old.pn_kc[:min_pn, :min_kc]
        for kc in range(min_kc, new_kc):
            chosen = rng.choice(new_pn, size=6, replace=False)
            pn_kc[chosen, kc] = rng.uniform(1.0, 5.0, size=6)

        kc_mbon = np.zeros((new_kc, new_mbon))
        min_mbon = min(old.kc_mbon.shape[1], new_mbon)
        kc_mbon[:min_kc, :min_mbon] = old.kc_mbon[:min_kc, :min_mbon]
        for kc in range(new_kc):
            for m in range(min_mbon, new_mbon):
                kc_mbon[kc, m] = rng.uniform(2.0, 10.0)
        for kc in range(min_kc, new_kc):
            kc_mbon[kc, :] = rng.uniform(2.0, 10.0, size=new_mbon)

        kc_thresh = np.full(new_kc, -45.0)
        kc_thresh[:min_kc] = old.kc_thresh[:min_kc]
        kc_thresh[min_kc:] = rng.uniform(-48, -42, size=new_kc - min_kc)

        return GrowingGenome(new_config, pn_kc, kc_mbon, kc_thresh,
                            old.kc_apl_w, old.apl_kc_w)

    def copy(self):
        return GrowingGenome(self.config, self.pn_kc.copy(), self.kc_mbon.copy(),
                            self.kc_thresh.copy(), self.kc_apl_w, self.apl_kc_w)

    def build_weight_matrices(self):
        n = self.config.num_neurons
        W_exc = np.zeros((n, n))
        W_inh = np.zeros((n, n))
        cfg = self.config

        pi, ki = np.nonzero(self.pn_kc)
        for idx in range(len(pi)):
            W_exc[cfg.pn_start + pi[idx], cfg.kc_start + ki[idx]] = self.pn_kc[pi[idx], ki[idx]] * 1e-9

        for kc_i in range(cfg.num_kc):
            for m in range(cfg.num_mbon):
                W_exc[cfg.kc_start + kc_i, cfg.mbon_start + m] = self.kc_mbon[kc_i, m] * 1e-9

        for kc_i in range(cfg.num_kc):
            W_exc[cfg.kc_start + kc_i, cfg.apl_start] = self.kc_apl_w * 1e-9

        for kc_i in range(cfg.num_kc):
            W_inh[cfg.apl_start, cfg.kc_start + kc_i] = self.apl_kc_w * 1e-9

        return W_exc, W_inh

    def build_threshold_vector(self):
        n = self.config.num_neurons
        cfg = self.config
        V_thresh = np.full(n, -0.055)
        V_thresh[cfg.kc_start:cfg.kc_end] = self.kc_thresh * 1e-3
        V_thresh[cfg.mbon_start:cfg.mbon_end] = -0.050
        V_thresh[cfg.apl_start:cfg.apl_end] = -0.045
        return V_thresh
