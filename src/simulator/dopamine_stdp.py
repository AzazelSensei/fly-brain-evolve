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


@njit(cache=True)
def _simulate_with_eligibility_attention(
    W_exc, W_inh, kc_mbon_w, input_spikes, V_thresh,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    tau_kc_trace, tau_mbon_trace, tau_elig,
    feedback_strength, feedback_inhibition, w_max_for_norm,
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
                        fb = kc_mbon_w[ki, mi] / w_max_for_norm
                        if fb > 0.1:
                            g_exc[kc_start + ki] += fb * feedback_strength * 1e-9
                        else:
                            g_inh[kc_start + ki] += feedback_inhibition * 1e-9

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
def _simulate_forward_attention(
    W_exc, W_inh, kc_mbon_w, input_spikes, V_thresh,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    feedback_strength, feedback_inhibition, w_max_for_norm,
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
                    mi = i - mbon_start
                    mbon_counts[mi] += 1
                    for ki in range(num_kc):
                        fb = kc_mbon_w[ki, mi] / w_max_for_norm
                        if fb > 0.1:
                            g_exc[kc_start + ki] += fb * feedback_strength * 1e-9
                        else:
                            g_inh[kc_start + ki] += feedback_inhibition * 1e-9

        g_exc *= decay_exc
        g_inh *= decay_inh

    return mbon_counts, kc_spiked


@njit(parallel=True, cache=True)
def train_and_evaluate_attention(
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
    pop_fb_strength, pop_fb_inhibition,
):
    pop_size = pop_W_exc.shape[0]
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    fitnesses = np.zeros(pop_size)

    for gi in prange(pop_size):
        kc_mbon = pop_kc_mbon_init[gi].copy()
        fb_str = pop_fb_strength[gi]
        fb_inh = pop_fb_inhibition[gi]

        for ti in range(n_train):
            mc, ks, elig = _simulate_with_eligibility_attention(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                train_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight,
                tau_kc_trace, tau_mbon_trace, tau_elig,
                fb_str, fb_inh, w_max,
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
            mc, ks = _simulate_forward_attention(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                test_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight,
                fb_str, fb_inh, w_max,
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


@njit(cache=True)
def _simulate_multipass(
    W_exc, W_inh, kc_mbon_w, input_spikes, V_thresh,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    feedback_strength, feedback_inhibition, w_max_for_norm,
    max_passes, confidence_threshold, kc_carry_decay,
    training_mode,
    tau_kc_trace, tau_mbon_trace, tau_elig,
):
    num_mbon = mbon_end - mbon_start
    num_kc = kc_end - kc_start
    total_mbon = np.zeros(num_mbon, dtype=np.int32)
    kc_spiked_ever = np.zeros(num_kc, dtype=np.int32)
    total_elig = np.zeros((num_kc, num_mbon))
    kc_v_carry = np.zeros(num_kc)
    kc_ge_carry = np.zeros(num_kc)
    kc_gi_carry = np.zeros(num_kc)
    decay_exc = np.exp(-dt / tau_exc)
    decay_inh = np.exp(-dt / tau_inh)
    decay_kc_t = np.exp(-dt / tau_kc_trace)
    decay_mbon_t = np.exp(-dt / tau_mbon_trace)
    decay_el = np.exp(-dt / tau_elig)
    E_exc = np.full(num_neurons, E_exc_val)
    E_inh = np.full(num_neurons, E_inh_val)

    for p in range(max_passes):
        v = V_rest.copy()
        g_exc = np.zeros(num_neurons)
        g_inh = np.zeros(num_neurons)
        refr = np.zeros(num_neurons, dtype=np.int32)
        if p > 0:
            for ki in range(num_kc):
                v[kc_start + ki] = V_rest[kc_start + ki] + kc_v_carry[ki] * kc_carry_decay
                g_exc[kc_start + ki] = kc_ge_carry[ki] * kc_carry_decay
                g_inh[kc_start + ki] = kc_gi_carry[ki] * kc_carry_decay
        mbon_counts = np.zeros(num_mbon, dtype=np.int32)
        kc_trace = np.zeros(num_kc)
        mbon_trace = np.zeros(num_mbon)
        elig = np.zeros((num_kc, num_mbon))

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
                        kc_spiked_ever[ki] = 1
                        kc_trace[ki] += 1.0
                        for m in range(num_mbon):
                            g_exc[mbon_start + m] += kc_mbon_w[ki, m] * 1e-9
                    if mbon_start <= i < mbon_end:
                        mi = i - mbon_start
                        mbon_counts[mi] += 1
                        mbon_trace[mi] += 1.0
                        for ki in range(num_kc):
                            fb = kc_mbon_w[ki, mi] / w_max_for_norm
                            if fb > 0.1:
                                g_exc[kc_start + ki] += fb * feedback_strength * 1e-9
                            else:
                                g_inh[kc_start + ki] += feedback_inhibition * 1e-9
            if training_mode == 1:
                for ki in range(num_kc):
                    if kc_trace[ki] > 0.001:
                        for m in range(num_mbon):
                            elig[ki, m] += kc_trace[ki] * mbon_trace[m]
                elig *= decay_el
            kc_trace *= decay_kc_t
            mbon_trace *= decay_mbon_t
            g_exc *= decay_exc
            g_inh *= decay_inh

        for ki in range(num_kc):
            kc_v_carry[ki] = v[kc_start + ki] - V_rest[kc_start + ki]
            kc_ge_carry[ki] = g_exc[kc_start + ki]
            kc_gi_carry[ki] = g_inh[kc_start + ki]
        total_mbon += mbon_counts
        if training_mode == 1:
            total_elig += elig
        if p < max_passes - 1:
            mx1 = 0
            mx2 = 0
            for m in range(num_mbon):
                if total_mbon[m] > mx1:
                    mx2 = mx1
                    mx1 = total_mbon[m]
                elif total_mbon[m] > mx2:
                    mx2 = total_mbon[m]
            if mx1 > 0 and (mx1 - mx2) / mx1 > confidence_threshold:
                break

    return total_mbon, kc_spiked_ever, total_elig


@njit(parallel=True, cache=True)
def train_and_evaluate_multipass(
    pop_W_exc, pop_W_inh, pop_kc_mbon_init, pop_V_thresh,
    train_spikes, train_labels, test_spikes, test_labels,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, num_mbon, num_kc,
    kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    tau_kc_trace, tau_mbon_trace, tau_elig,
    lr, w_min, w_max,
    reward_signal, punishment_signal,
    pop_fb_str, pop_fb_inh,
    pop_max_passes, pop_conf_thresh, pop_kc_decay,
):
    pop_size = pop_W_exc.shape[0]
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    fitnesses = np.zeros(pop_size)

    for gi in prange(pop_size):
        kc_mbon = pop_kc_mbon_init[gi].copy()
        fbs = pop_fb_str[gi]
        fbi = pop_fb_inh[gi]
        mp = int(pop_max_passes[gi])
        ct = pop_conf_thresh[gi]
        cd = pop_kc_decay[gi]

        for ti in range(n_train):
            mc, ks, el = _simulate_multipass(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                train_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight, fbs, fbi, w_max,
                mp, ct, cd, 1,
                tau_kc_trace, tau_mbon_trace, tau_elig)
            dopamine = np.full(num_mbon, punishment_signal)
            dopamine[train_labels[ti]] = reward_signal
            for ki in range(num_kc):
                for m in range(num_mbon):
                    kc_mbon[ki, m] += lr * el[ki, m] * dopamine[m]
                    if kc_mbon[ki, m] < w_min:
                        kc_mbon[ki, m] = w_min
                    if kc_mbon[ki, m] > w_max:
                        kc_mbon[ki, m] = w_max

        correct = 0
        total_kc = 0
        for ti in range(n_test):
            mc, ks, _ = _simulate_multipass(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                test_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight, fbs, fbi, w_max,
                mp, ct, cd, 0,
                tau_kc_trace, tau_mbon_trace, tau_elig)
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


@njit(parallel=True, cache=True)
def train_and_evaluate_meta(
    pop_W_exc, pop_W_inh, pop_kc_mbon_init, pop_V_thresh,
    train_spikes, train_labels, test_spikes, test_labels,
    tau_m, V_rest, V_reset, g_L,
    E_exc_val, E_inh_val, tau_exc, tau_inh,
    dt, num_steps, refr_steps,
    num_neurons, num_pn, num_mbon, num_kc,
    kc_start, kc_end, mbon_start, mbon_end,
    input_weight,
    pop_tau_kc, pop_tau_mbon, pop_tau_elig,
    pop_lr, pop_w_min, pop_w_max,
    pop_reward, pop_punish,
    pop_fb_str, pop_fb_inh,
    pop_max_passes, pop_conf_thresh, pop_kc_decay,
):
    pop_size = pop_W_exc.shape[0]
    n_train = train_labels.shape[0]
    n_test = test_labels.shape[0]
    fitnesses = np.zeros(pop_size)

    for gi in prange(pop_size):
        kc_mbon = pop_kc_mbon_init[gi].copy()
        fbs = pop_fb_str[gi]
        fbi = pop_fb_inh[gi]
        mp = int(pop_max_passes[gi])
        ct = pop_conf_thresh[gi]
        cd = pop_kc_decay[gi]
        lr = pop_lr[gi]
        wmin = pop_w_min[gi]
        wmax = pop_w_max[gi]
        rw = pop_reward[gi]
        pu = pop_punish[gi]
        tk = pop_tau_kc[gi]
        tm = pop_tau_mbon[gi]
        te = pop_tau_elig[gi]

        for ti in range(n_train):
            mc, ks, el = _simulate_multipass(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                train_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight, fbs, fbi, wmax,
                mp, ct, cd, 1, tk, tm, te)
            dopamine = np.full(num_mbon, pu)
            dopamine[train_labels[ti]] = rw
            for ki in range(num_kc):
                for m in range(num_mbon):
                    kc_mbon[ki, m] += lr * el[ki, m] * dopamine[m]
                    if kc_mbon[ki, m] < wmin:
                        kc_mbon[ki, m] = wmin
                    if kc_mbon[ki, m] > wmax:
                        kc_mbon[ki, m] = wmax

        correct = 0
        total_kc = 0
        for ti in range(n_test):
            mc, ks, _ = _simulate_multipass(
                pop_W_exc[gi], pop_W_inh[gi], kc_mbon,
                test_spikes[ti], pop_V_thresh[gi],
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                dt, num_steps, refr_steps,
                num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                input_weight, fbs, fbi, wmax,
                mp, ct, cd, 0, tk, tm, te)
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
