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
NUM_KC = 200
NUM_MBON = 2


@njit(cache=True)
def _run_with_dopamine(
    v, g_exc, g_inh, refr,
    W_exc, W_inh,
    kc_mbon_w,
    input_spikes,
    tau_m, V_rest, V_thresh, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    dopamine_signal,
    tau_eligibility, learning_rate,
):
    mbon_counts = np.zeros(NUM_MBON, dtype=np.int32)
    kc_spiked_ever = np.zeros(NUM_KC, dtype=np.int32)

    eligibility = np.zeros((NUM_KC, NUM_MBON))
    kc_trace = np.zeros(NUM_KC)
    mbon_trace = np.zeros(NUM_MBON)

    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)
    decay_elig = np.exp(-dt / tau_eligibility)
    decay_trace = np.exp(-dt / 0.020)

    E_exc = np.full(NUM_NEURONS, E_exc_val)
    E_inh = np.full(NUM_NEURONS, E_inh_val)

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

                if KC_START <= i < KC_END:
                    kc_idx = i - KC_START
                    kc_spiked_ever[kc_idx] = 1
                    kc_trace[kc_idx] += 1.0

                    for m in range(NUM_MBON):
                        g_exc[MBON_START + m] += kc_mbon_w[kc_idx, m] * 1e-9

                if MBON_START <= i < MBON_END:
                    m_idx = i - MBON_START
                    mbon_counts[m_idx] += 1
                    mbon_trace[m_idx] += 1.0

        for kc_idx in range(NUM_KC):
            for m in range(NUM_MBON):
                eligibility[kc_idx, m] += kc_trace[kc_idx] * mbon_trace[m]
                eligibility[kc_idx, m] *= decay_elig

        kc_trace *= decay_trace
        mbon_trace *= decay_trace
        g_exc *= decay_exc
        g_inh *= decay_inh

    dw = eligibility * dopamine_signal * learning_rate
    kc_mbon_w_new = kc_mbon_w + dw
    kc_mbon_w_new = np.clip(kc_mbon_w_new, 0.0, 15.0)

    return mbon_counts, kc_spiked_ever, kc_mbon_w_new


@njit(parallel=True, cache=True)
def train_with_dopamine_batch(
    batch_W_exc, batch_W_inh, batch_kc_mbon_w,
    batch_input_spikes, batch_V_thresh,
    batch_labels,
    V_rest_arr, V_reset_arr, g_L_arr,
    E_exc_val, E_inh_val,
    tau_m_arr, tau_exc_val, tau_inh_val,
    dt, num_steps, refr_steps,
    tau_eligibility, learning_rate,
):
    batch_size = batch_W_exc.shape[0]
    all_mbon = np.zeros((batch_size, NUM_MBON), dtype=np.int32)
    all_kc = np.zeros((batch_size, NUM_KC), dtype=np.int32)
    all_new_w = np.zeros((batch_size, NUM_KC, NUM_MBON))

    for b in prange(batch_size):
        v = np.full(NUM_NEURONS, -0.070)
        g_exc = np.zeros(NUM_NEURONS)
        g_inh = np.zeros(NUM_NEURONS)
        refr = np.zeros(NUM_NEURONS, dtype=np.int32)

        predicted_label = -1

        mc, ks, new_w = _run_with_dopamine(
            v, g_exc, g_inh, refr,
            batch_W_exc[b], batch_W_inh[b],
            batch_kc_mbon_w[b],
            batch_input_spikes[b],
            tau_m_arr, V_rest_arr, batch_V_thresh[b], V_reset_arr, g_L_arr,
            E_exc_val, E_inh_val, tau_exc_val, tau_inh_val,
            dt, num_steps, refr_steps,
            0.0, tau_eligibility, learning_rate,
        )

        if mc[0] + mc[1] > 0:
            if mc[0] > mc[1]:
                predicted_label = 0
            else:
                predicted_label = 1
        else:
            predicted_label = -1

        true_label = batch_labels[b]
        if predicted_label == true_label:
            dopamine = 1.0
        elif predicted_label == -1:
            dopamine = 0.0
        else:
            dopamine = -1.0

        _, _, new_w2 = _run_with_dopamine(
            np.full(NUM_NEURONS, -0.070),
            np.zeros(NUM_NEURONS),
            np.zeros(NUM_NEURONS),
            np.zeros(NUM_NEURONS, dtype=np.int32),
            batch_W_exc[b], batch_W_inh[b],
            batch_kc_mbon_w[b],
            batch_input_spikes[b],
            tau_m_arr, V_rest_arr, batch_V_thresh[b], V_reset_arr, g_L_arr,
            E_exc_val, E_inh_val, tau_exc_val, tau_inh_val,
            dt, num_steps, refr_steps,
            dopamine, tau_eligibility, learning_rate,
        )

        all_mbon[b] = mc
        all_kc[b] = ks
        all_new_w[b] = new_w2

    return all_mbon, all_kc, all_new_w


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
