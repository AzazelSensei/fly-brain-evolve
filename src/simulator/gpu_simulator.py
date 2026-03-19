import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def check_gpu():
    if not GPU_AVAILABLE:
        return False, "CuPy not installed"
    try:
        device = cp.cuda.Device(0)
        mem_free, mem_total = device.mem_info
        name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return True, f"{name} ({mem_total // (1024**3)} GB, {mem_free // (1024**3)} GB free)"
    except Exception as e:
        return False, str(e)


def _build_spike_propagation_kernel():
    return cp.RawKernel(r'''
    extern "C" __global__
    void propagate_spikes(
        const float* W_exc, const float* W_inh,
        float* g_exc, float* g_inh,
        const int* spiked, const int num_neurons, const int n_spiked
    ) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= num_neurons) return;

        float exc_sum = 0.0f;
        float inh_sum = 0.0f;
        for (int s = 0; s < n_spiked; s++) {
            int i = spiked[s];
            float we = W_exc[i * num_neurons + j];
            float wi = W_inh[i * num_neurons + j];
            if (we > 0.0f) exc_sum += we;
            if (wi > 0.0f) inh_sum += wi;
        }
        g_exc[j] += exc_sum;
        g_inh[j] += inh_sum;
    }
    ''', 'propagate_spikes')


def _build_kc_mbon_kernel():
    return cp.RawKernel(r'''
    extern "C" __global__
    void kc_mbon_propagate(
        const float* kc_mbon_w, float* g_exc,
        const int* kc_spiked_ids, const int n_kc_spiked,
        const int mbon_start, const int num_mbon
    ) {
        int m = blockIdx.x * blockDim.x + threadIdx.x;
        if (m >= num_mbon) return;

        float sum = 0.0f;
        for (int s = 0; s < n_kc_spiked; s++) {
            int ki = kc_spiked_ids[s];
            sum += kc_mbon_w[ki * num_mbon + m] * 1e-9f;
        }
        g_exc[mbon_start + m] += sum;
    }
    ''', 'kc_mbon_propagate')


def _build_eligibility_kernel():
    return cp.RawKernel(r'''
    extern "C" __global__
    void update_eligibility(
        float* eligibility, const float* kc_trace, const float* mbon_trace,
        const float decay_elig, const int num_kc, const int num_mbon
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_kc * num_mbon) return;

        int ki = idx / num_mbon;
        int m = idx % num_mbon;

        float elig = eligibility[idx] * decay_elig;
        if (kc_trace[ki] > 0.001f) {
            elig += kc_trace[ki] * mbon_trace[m];
        }
        eligibility[idx] = elig;
    }
    ''', 'update_eligibility')


def _build_weight_update_kernel():
    return cp.RawKernel(r'''
    extern "C" __global__
    void apply_stdp(
        float* kc_mbon_w, const float* eligibility, const float* dopamine,
        const float lr, const float w_min, const float w_max,
        const int num_kc, const int num_mbon
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_kc * num_mbon) return;

        int m = idx % num_mbon;
        float dw = lr * eligibility[idx] * dopamine[m];
        float w = kc_mbon_w[idx] + dw;
        if (w < w_min) w = w_min;
        if (w > w_max) w = w_max;
        kc_mbon_w[idx] = w;
    }
    ''', 'apply_stdp')


class GPUSimulator:
    def __init__(self, num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                 dt=0.00005, num_steps=2000, refr_steps=40):
        self.N = num_neurons
        self.num_pn = num_pn
        self.kc_start = kc_start
        self.kc_end = kc_end
        self.mbon_start = mbon_start
        self.mbon_end = mbon_end
        self.num_kc = kc_end - kc_start
        self.num_mbon = mbon_end - mbon_start
        self.dt = dt
        self.num_steps = num_steps
        self.refr_steps = refr_steps

        self.propagate_kernel = _build_spike_propagation_kernel()
        self.kc_mbon_kernel = _build_kc_mbon_kernel()
        self.eligibility_kernel = _build_eligibility_kernel()
        self.weight_update_kernel = _build_weight_update_kernel()

        self.threads = 256
        self.blocks_N = (self.N + self.threads - 1) // self.threads
        self.blocks_kc_mbon = (self.num_kc * self.num_mbon + self.threads - 1) // self.threads

    def simulate_with_eligibility(self, W_exc, W_inh, kc_mbon_w, features, V_thresh,
                                   tau_m, V_rest, V_reset, g_L,
                                   E_exc_val, E_inh_val, tau_exc, tau_inh,
                                   input_weight, max_rate,
                                   tau_kc_trace, tau_mbon_trace, tau_elig):
        N = self.N

        v = cp.array(V_rest, dtype=cp.float32)
        g_exc = cp.zeros(N, dtype=cp.float32)
        g_inh = cp.zeros(N, dtype=cp.float32)
        refr = cp.zeros(N, dtype=cp.int32)

        tau_m_g = cp.array(tau_m, dtype=cp.float32)
        V_rest_g = cp.array(V_rest, dtype=cp.float32)
        V_reset_g = cp.array(V_reset, dtype=cp.float32)
        g_L_g = cp.array(g_L, dtype=cp.float32)
        V_thresh_g = cp.array(V_thresh, dtype=cp.float32)

        W_exc_g = cp.array(W_exc, dtype=cp.float32)
        W_inh_g = cp.array(W_inh, dtype=cp.float32)
        kc_mbon_g = cp.array(kc_mbon_w, dtype=cp.float32).ravel()

        kc_trace = cp.zeros(self.num_kc, dtype=cp.float32)
        mbon_trace = cp.zeros(self.num_mbon, dtype=cp.float32)
        eligibility = cp.zeros(self.num_kc * self.num_mbon, dtype=cp.float32)

        mbon_counts = cp.zeros(self.num_mbon, dtype=cp.int32)
        kc_spiked_ever = cp.zeros(self.num_kc, dtype=cp.int32)

        decay_exc = float(np.exp(-self.dt / tau_exc))
        decay_inh = float(np.exp(-self.dt / tau_inh))
        decay_kc = float(np.exp(-self.dt / tau_kc_trace))
        decay_mbon = float(np.exp(-self.dt / tau_mbon_trace))
        decay_elig = float(np.exp(-self.dt / tau_elig))

        E_exc_arr = cp.full(N, E_exc_val, dtype=cp.float32)
        E_inh_arr = cp.full(N, E_inh_val, dtype=cp.float32)

        features_g = cp.array(features, dtype=cp.float32)
        spike_threshold = features_g * max_rate * self.dt

        for t in range(self.num_steps):
            rand_vals = cp.random.random(self.num_pn, dtype=cp.float32)
            spike_mask = rand_vals < spike_threshold[:self.num_pn]
            g_exc[:self.num_pn] += spike_mask.astype(cp.float32) * input_weight

            I_syn = g_exc * (E_exc_arr - v) + g_inh * (E_inh_arr - v)
            dv = (-(v - V_rest_g) + I_syn / g_L_g) / tau_m_g * self.dt

            active_mask = refr == 0
            v += dv * active_mask.astype(cp.float32)
            refr = cp.maximum(refr - 1, 0)

            spike_mask_all = (v > V_thresh_g) & (refr == 0)
            spiked_ids = cp.where(spike_mask_all)[0].astype(cp.int32)
            n_spiked = int(len(spiked_ids))

            if n_spiked > 0:
                v[spike_mask_all] = V_reset_g[spike_mask_all]
                refr[spike_mask_all] = self.refr_steps

                self.propagate_kernel(
                    (max(1, (N + self.threads - 1) // self.threads),), (self.threads,),
                    (W_exc_g.ravel(), W_inh_g.ravel(), g_exc, g_inh,
                     spiked_ids, N, n_spiked))

                kc_spike_mask = (spiked_ids >= self.kc_start) & (spiked_ids < self.kc_end)
                kc_spiked_local = spiked_ids[kc_spike_mask] - self.kc_start
                if len(kc_spiked_local) > 0:
                    kc_spiked_ever[kc_spiked_local] = 1
                    kc_trace[kc_spiked_local] += 1.0

                    self.kc_mbon_kernel(
                        (max(1, (self.num_mbon + self.threads - 1) // self.threads),), (self.threads,),
                        (kc_mbon_g, g_exc, kc_spiked_local, len(kc_spiked_local),
                         self.mbon_start, self.num_mbon))

                mbon_spike_mask = (spiked_ids >= self.mbon_start) & (spiked_ids < self.mbon_end)
                mbon_spiked_local = spiked_ids[mbon_spike_mask] - self.mbon_start
                if len(mbon_spiked_local) > 0:
                    mbon_counts[mbon_spiked_local] += 1
                    mbon_trace[mbon_spiked_local] += 1.0

            self.eligibility_kernel(
                (self.blocks_kc_mbon,), (self.threads,),
                (eligibility, kc_trace, mbon_trace, decay_elig,
                 self.num_kc, self.num_mbon))

            kc_trace *= decay_kc
            mbon_trace *= decay_mbon
            g_exc *= decay_exc
            g_inh *= decay_inh

        return (cp.asnumpy(mbon_counts), cp.asnumpy(kc_spiked_ever),
                cp.asnumpy(eligibility.reshape(self.num_kc, self.num_mbon)))

    def train_single_organism(self, W_exc, W_inh, kc_mbon_w,
                               train_features, train_labels,
                               test_features, test_labels,
                               tau_m, V_rest, V_reset, g_L,
                               E_exc_val, E_inh_val, tau_exc, tau_inh,
                               input_weight, max_rate,
                               tau_kc_trace, tau_mbon_trace, tau_elig,
                               lr, w_min, w_max, V_thresh,
                               reward_signal, punishment_signal):
        kc_mbon = kc_mbon_w.copy()
        kc_mbon_g = cp.array(kc_mbon, dtype=cp.float32).ravel()
        dopamine_g = cp.zeros(self.num_mbon, dtype=cp.float32)

        for ti in range(len(train_labels)):
            mc, ks, elig = self.simulate_with_eligibility(
                W_exc, W_inh, cp.asnumpy(kc_mbon_g).reshape(self.num_kc, self.num_mbon),
                train_features[ti], V_thresh,
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                input_weight, max_rate,
                tau_kc_trace, tau_mbon_trace, tau_elig)

            dopamine_g[:] = punishment_signal
            dopamine_g[train_labels[ti]] = reward_signal

            elig_g = cp.array(elig.ravel(), dtype=cp.float32)
            self.weight_update_kernel(
                (self.blocks_kc_mbon,), (self.threads,),
                (kc_mbon_g, elig_g, dopamine_g, lr, w_min, w_max,
                 self.num_kc, self.num_mbon))

        correct = 0
        total_kc = 0
        final_kc_mbon = cp.asnumpy(kc_mbon_g).reshape(self.num_kc, self.num_mbon)

        for ti in range(len(test_labels)):
            mc, ks = self._forward(
                W_exc, W_inh, final_kc_mbon,
                test_features[ti], V_thresh,
                tau_m, V_rest, V_reset, g_L,
                E_exc_val, E_inh_val, tau_exc, tau_inh,
                input_weight, max_rate)

            if mc.sum() > 0 and np.argmax(mc) == test_labels[ti]:
                correct += 1
            total_kc += ks.sum()

        acc = correct / len(test_labels)
        sp = total_kc / (len(test_labels) * self.num_kc)
        sp_bonus = max(0, 1 - abs(sp - 0.1) / 0.1) * 0.1
        return acc + sp_bonus

    def _forward(self, W_exc, W_inh, kc_mbon_w, features, V_thresh,
                  tau_m, V_rest, V_reset, g_L,
                  E_exc_val, E_inh_val, tau_exc, tau_inh,
                  input_weight, max_rate):
        N = self.N

        v = cp.array(V_rest, dtype=cp.float32)
        g_exc = cp.zeros(N, dtype=cp.float32)
        g_inh = cp.zeros(N, dtype=cp.float32)
        refr = cp.zeros(N, dtype=cp.int32)

        W_exc_g = cp.array(W_exc, dtype=cp.float32)
        W_inh_g = cp.array(W_inh, dtype=cp.float32)
        kc_mbon_g = cp.array(kc_mbon_w, dtype=cp.float32).ravel()

        tau_m_g = cp.array(tau_m, dtype=cp.float32)
        V_rest_g = cp.array(V_rest, dtype=cp.float32)
        V_reset_g = cp.array(V_reset, dtype=cp.float32)
        g_L_g = cp.array(g_L, dtype=cp.float32)
        V_thresh_g = cp.array(V_thresh, dtype=cp.float32)

        decay_exc = float(np.exp(-self.dt / tau_exc))
        decay_inh = float(np.exp(-self.dt / tau_inh))

        E_exc_arr = cp.full(N, E_exc_val, dtype=cp.float32)
        E_inh_arr = cp.full(N, E_inh_val, dtype=cp.float32)

        mbon_counts = cp.zeros(self.num_mbon, dtype=cp.int32)
        kc_spiked = cp.zeros(self.num_kc, dtype=cp.int32)

        features_g = cp.array(features, dtype=cp.float32)
        spike_threshold = features_g * max_rate * self.dt

        for t in range(self.num_steps):
            rand_vals = cp.random.random(self.num_pn, dtype=cp.float32)
            spike_mask = rand_vals < spike_threshold[:self.num_pn]
            g_exc[:self.num_pn] += spike_mask.astype(cp.float32) * input_weight

            I_syn = g_exc * (E_exc_arr - v) + g_inh * (E_inh_arr - v)
            dv = (-(v - V_rest_g) + I_syn / g_L_g) / tau_m_g * self.dt

            active_mask = refr == 0
            v += dv * active_mask.astype(cp.float32)
            refr = cp.maximum(refr - 1, 0)

            spike_mask_all = (v > V_thresh_g) & (refr == 0)
            spiked_ids = cp.where(spike_mask_all)[0].astype(cp.int32)
            n_spiked = int(len(spiked_ids))

            if n_spiked > 0:
                v[spike_mask_all] = V_reset_g[spike_mask_all]
                refr[spike_mask_all] = self.refr_steps

                self.propagate_kernel(
                    (max(1, (N + self.threads - 1) // self.threads),), (self.threads,),
                    (W_exc_g.ravel(), W_inh_g.ravel(), g_exc, g_inh,
                     spiked_ids, N, n_spiked))

                kc_spike_mask = (spiked_ids >= self.kc_start) & (spiked_ids < self.kc_end)
                kc_spiked_local = spiked_ids[kc_spike_mask] - self.kc_start
                if len(kc_spiked_local) > 0:
                    kc_spiked[kc_spiked_local] = 1
                    self.kc_mbon_kernel(
                        (max(1, (self.num_mbon + self.threads - 1) // self.threads),), (self.threads,),
                        (kc_mbon_g, g_exc, kc_spiked_local, len(kc_spiked_local),
                         self.mbon_start, self.num_mbon))

                mbon_spike_mask = (spiked_ids >= self.mbon_start) & (spiked_ids < self.mbon_end)
                mbon_spiked_local = spiked_ids[mbon_spike_mask] - self.mbon_start
                if len(mbon_spiked_local) > 0:
                    mbon_counts[mbon_spiked_local] += 1

            g_exc *= decay_exc
            g_inh *= decay_inh

        return cp.asnumpy(mbon_counts), cp.asnumpy(kc_spiked)


def benchmark_gpu_vs_cpu(num_neurons_list=None):
    if num_neurons_list is None:
        num_neurons_list = [500, 1000, 2000, 5000, 10000]

    available, info = check_gpu()
    print(f"GPU: {info}")

    if not available:
        print("GPU not available, skipping benchmark")
        return

    from src.simulator.growing_brain import BrainConfig

    results = []
    for n_kc in num_neurons_list:
        cfg = BrainConfig(num_pn=128, num_kc=n_kc, num_mbon=10, num_apl=1)
        N = cfg.num_neurons
        tau_m, V_rest, V_reset, g_L = cfg.build_params()

        W_exc = np.random.rand(N, N).astype(np.float32) * 1e-9
        W_inh = np.random.rand(N, N).astype(np.float32) * 1e-9
        kc_mbon = np.full((cfg.num_kc, cfg.num_mbon), 5.0, dtype=np.float32)
        features = np.random.rand(128).astype(np.float32) * 0.3
        V_thresh = np.full(N, -0.055, dtype=np.float32)

        sim = GPUSimulator(N, cfg.num_pn, cfg.kc_start, cfg.kc_end,
                           cfg.mbon_start, cfg.mbon_end)

        mc, ks, elig = sim.simulate_with_eligibility(
            W_exc, W_inh, kc_mbon, features, V_thresh,
            tau_m, V_rest, V_reset, g_L,
            0.0, -0.080, 0.005, 0.010,
            100e-9, 500.0, 0.020, 0.020, 0.040)

        import time
        t0 = time.time()
        for _ in range(5):
            sim.simulate_with_eligibility(
                W_exc, W_inh, kc_mbon, features, V_thresh,
                tau_m, V_rest, V_reset, g_L,
                0.0, -0.080, 0.005, 0.010,
                100e-9, 500.0, 0.020, 0.020, 0.040)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - t0) / 5

        r = {"kc": n_kc, "neurons": N, "gpu_ms": round(gpu_time * 1000, 1)}
        results.append(r)
        print(f"  {N:6d} neurons ({n_kc} KC): GPU={gpu_time*1000:.1f}ms/sim")

    return results
