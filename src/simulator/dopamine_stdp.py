import numpy as np
from numba import njit, prange


@njit(cache=True)
def _simulate_with_eligibility(
    W_exc, W_inh, kc_mbon_w, input_spikes, V_thresh,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    tau_kc_trace, tau_mbon_trace, tau_elig,
):
    num_mbon = mbon_end - mbon_start
    num_kc = kc_end - kc_start
    mbon_counts = np.zeros(num_mbon, dtype=np.int32)
    kc_spiked = np.zeros(num_kc, dtype=np.int32)

    v = V_rest.copy()
    g_exc = np.zeros(num_neurons)
    g_inh = np.zeros(num_neurons)
    refr = np.zeros(num_neurons, dtype=np.int32)

    kc_trace = np.zeros(num_kc)
    mbon_trace = np.zeros(num_mbon)
    eligibility = np.zeros((num_kc, num_mbon))

    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)
    decay_kc = np.exp(-dt / tau_kc_trace)
    decay_mbon = np.exp(-dt / tau_mbon_trace)
    decay_elig = np.exp(-dt / tau_elig)

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

                if kc_start <= i < kc_end:
                    ki = i - kc_start
                    kc_spiked[ki] = 1
                    kc_trace[ki] += 1.0
                    for m in range(num_mbon):
                        g_exc[mbon_start + m] += kc_mbon_w[ki, m] * 1e-9

                if mbon_start <= i < mbon_end:
                    mi = i - mbon_start
                    mbon_counts[mi] += 1
                    mbon_trace[mi] += 1.0

        for ki in range(num_kc):
            if kc_trace[ki] > 0.001:
                for m in range(num_mbon):
                    eligibility[ki, m] += kc_trace[ki] * mbon_trace[m]
        eligibility *= decay_elig

        kc_trace *= decay_kc
        mbon_trace *= decay_mbon
        g_exc *= decay_exc
        g_inh *= decay_inh

    return mbon_counts, kc_spiked, eligibility


@njit(cache=True)
def _simulate_forward(
    W_exc, W_inh, kc_mbon_w, input_spikes, V_thresh,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
):
    num_mbon = mbon_end - mbon_start
    num_kc = kc_end - kc_start
    mbon_counts = np.zeros(num_mbon, dtype=np.int32)
    kc_spiked = np.zeros(num_kc, dtype=np.int32)

    v = V_rest.copy()
    g_exc = np.zeros(num_neurons)
    g_inh = np.zeros(num_neurons)
    refr = np.zeros(num_neurons, dtype=np.int32)

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

                if kc_start <= i < kc_end:
                    ki = i - kc_start
                    kc_spiked[ki] = 1
                    for m in range(num_mbon):
                        g_exc[mbon_start + m] += kc_mbon_w[ki, m] * 1e-9

                if mbon_start <= i < mbon_end:
                    mbon_counts[i - mbon_start] += 1

        g_exc *= decay_exc
        g_inh *= decay_inh

    return mbon_counts, kc_spiked


@njit(parallel=True, cache=True)
def train_and_evaluate(
    pop_W_exc, pop_W_inh, pop_kc_mbon_init, pop_V_thresh,
    train_spikes, train_labels,
    test_spikes, test_labels,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, num_mbon, num_kc,
    kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    tau_kc_trace, tau_mbon_trace, tau_elig,
    lr, w_min, w_max,
    reward_signal, punishment_signal,
):
    pop_size = pop_W_exc.shape[0]
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    fitnesses = np.zeros(pop_size)

    for gi in prange(pop_size):
        kc_mbon = pop_kc_mbon_init[gi].copy()

        for ti in range(n_train):
            mc, ks, elig = _simulate_with_eligibility(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                train_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight,
                tau_kc_trace, tau_mbon_trace, tau_elig,
            )

            dopamine = np.full(num_mbon, punishment_signal)
            dopamine[train_labels[ti]] = reward_signal

            for ki in range(num_kc):
                for m in range(num_mbon):
                    kc_mbon[ki, m] += lr * elig[ki, m] * dopamine[m]
                    if kc_mbon[ki, m] < w_min:
                        kc_mbon[ki, m] = w_min
                    if kc_mbon[ki, m] > w_max:
                        kc_mbon[ki, m] = w_max

        correct = 0
        total_kc = 0
        for ti in range(n_test):
            mc, ks = _simulate_forward(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                test_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight,
            )

            best_m = -1
            best_count = 0
            for m in range(num_mbon):
                if mc[m] > best_count:
                    best_count = mc[m]
                    best_m = m

            if best_m >= 0 and best_m == test_labels[ti]:
                correct += 1
            total_kc += ks.sum()

        acc = correct / n_test
        sp = total_kc / (n_test * num_kc)
        sp_bonus = 0.0
        if abs(sp - 0.1) < 0.1:
            sp_bonus = (1.0 - abs(sp - 0.1) / 0.1) * 0.1
        fitnesses[gi] = acc + sp_bonus

    return fitnesses
