import numpy as np
from numba import njit, prange


PN_START = 0
PN_END = 64
KC_START = 64
KC_END = 264
MBON_START = 264
MBON_END = 266
APL_IDX = 266
NUM_NEURONS = 267


@njit(cache=True)
def _run_single(
    v, g_exc, g_inh, refr,
    W_exc, W_inh,
    input_spikes,
    tau_m, V_rest, V_thresh, V_reset, g_L, E_exc, E_inh, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
):
    mbon_counts = np.zeros(2, dtype=np.int32)
    kc_spiked = np.zeros(200, dtype=np.int32)
    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)

    for t in range(num_steps):
        for pn in range(64):
            if input_spikes[t, pn]:
                g_exc[pn] += 50e-9

        I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v)
        dv = (-(v - V_rest) + I_syn / g_L) / tau_m * dt
        for i in range(NUM_NEURONS):
            if refr[i] > 0:
                refr[i] -= 1
            else:
                v[i] += dv[i]

        for i in range(NUM_NEURONS):
            if v[i] > V_thresh[i] and refr[i] == 0:
                v[i] = V_reset[i]
                refr[i] = refr_steps

                for j in range(NUM_NEURONS):
                    if W_exc[i, j] > 0:
                        g_exc[j] += W_exc[i, j]
                    if W_inh[i, j] > 0:
                        g_inh[j] += W_inh[i, j]

                if MBON_START <= i < MBON_END:
                    mbon_counts[i - MBON_START] += 1
                if KC_START <= i < KC_END:
                    kc_spiked[i - KC_START] = 1

        g_exc *= decay_exc
        g_inh *= decay_inh

    return mbon_counts, kc_spiked


@njit(parallel=True, cache=True)
def simulate_batch(
    batch_W_exc, batch_W_inh, batch_input_spikes, batch_V_thresh,
    V_rest_arr, V_reset_arr, g_L_arr, E_exc_val, E_inh_val,
    tau_m_arr, tau_exc_val, tau_inh_val,
    dt, num_steps, refr_steps,
):
    batch_size = batch_W_exc.shape[0]
    all_mbon_counts = np.zeros((batch_size, 2), dtype=np.int32)
    all_kc_spiked = np.zeros((batch_size, 200), dtype=np.int32)

    for b in prange(batch_size):
        v = np.full(NUM_NEURONS, -0.070)
        g_exc = np.zeros(NUM_NEURONS)
        g_inh = np.zeros(NUM_NEURONS)
        refr = np.zeros(NUM_NEURONS, dtype=np.int32)

        V_thresh = batch_V_thresh[b]
        E_exc = np.full(NUM_NEURONS, E_exc_val)
        E_inh = np.full(NUM_NEURONS, E_inh_val)

        mc, ks = _run_single(
            v, g_exc, g_inh, refr,
            batch_W_exc[b], batch_W_inh[b],
            batch_input_spikes[b],
            tau_m_arr, V_rest_arr, V_thresh, V_reset_arr,
            g_L_arr, E_exc, E_inh, tau_exc_val, tau_inh_val,
            dt, num_steps, refr_steps,
        )
        all_mbon_counts[b] = mc
        all_kc_spiked[b] = ks

    return all_mbon_counts, all_kc_spiked


def build_neuron_params():
    tau_m = np.zeros(NUM_NEURONS)
    V_rest = np.full(NUM_NEURONS, -0.070)
    V_reset = np.full(NUM_NEURONS, -0.070)
    g_L = np.full(NUM_NEURONS, 25e-9)

    tau_m[PN_START:PN_END] = 0.010
    tau_m[KC_START:KC_END] = 0.020
    tau_m[MBON_START:MBON_END] = 0.015
    tau_m[APL_IDX] = 0.010

    return tau_m, V_rest, V_reset, g_L


def build_weight_matrix(genome):
    W_exc = np.zeros((NUM_NEURONS, NUM_NEURONS))
    W_inh = np.zeros((NUM_NEURONS, NUM_NEURONS))

    pn_kc = genome.pn_kc[:64, :200] if genome.pn_kc.shape[0] >= 64 else genome.pn_kc
    pi, ki = np.nonzero(pn_kc)
    for idx in range(len(pi)):
        W_exc[PN_START + pi[idx], KC_START + ki[idx]] = pn_kc[pi[idx], ki[idx]] * 1e-9

    kc_mbon = genome.kc_mbon[:200, :2] if genome.kc_mbon.shape[0] >= 200 else genome.kc_mbon
    for kc_i in range(kc_mbon.shape[0]):
        for mbon_j in range(kc_mbon.shape[1]):
            W_exc[KC_START + kc_i, MBON_START + mbon_j] = kc_mbon[kc_i, mbon_j] * 1e-9

    for kc_i in range(min(200, genome.kc_mbon.shape[0])):
        W_exc[KC_START + kc_i, APL_IDX] = 2.0 * 1e-9

    for kc_i in range(min(200, genome.kc_mbon.shape[0])):
        W_inh[APL_IDX, KC_START + kc_i] = 200.0 * 1e-9

    return W_exc, W_inh


def build_threshold_vector(genome):
    V_thresh = np.full(NUM_NEURONS, -0.055)
    kc_thresh = genome.kc_thresh[:200] if hasattr(genome, "kc_thresh") else np.full(200, -0.045)
    V_thresh[KC_START:KC_END] = kc_thresh * 1e-3
    V_thresh[MBON_START:MBON_END] = -0.050
    V_thresh[APL_IDX] = -0.045
    return V_thresh


def generate_input_spikes_batch(patterns, labels, n_trials, num_pn, duration, dt, seed=42):
    rng = np.random.default_rng(seed)
    num_steps = int(duration / dt)
    batch_size = len(patterns) * n_trials
    all_spikes = np.zeros((batch_size, num_steps, num_pn), dtype=np.bool_)
    batch_labels = np.zeros(batch_size, dtype=np.int32)

    idx = 0
    for p_idx in range(len(patterns)):
        pat = patterns[p_idx].flatten()
        rates = np.clip(pat, 0, 1) * 100.0
        for trial in range(n_trials):
            for pn in range(min(len(rates), num_pn)):
                prob = rates[pn] * dt
                all_spikes[idx, :, pn] = rng.random(num_steps) < prob
            batch_labels[idx] = labels[p_idx]
            idx += 1

    return all_spikes, batch_labels
