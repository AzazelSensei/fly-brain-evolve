import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


FUSED_KERNEL_CODE = r'''
extern "C" __global__
void simulate_fused(
    const float* W_exc,
    const float* W_inh,
    float* kc_mbon_w,
    const unsigned char* input_spikes,
    const float* V_thresh,
    const float* tau_m,
    const float* V_rest,
    const float* V_reset,
    const float* g_L,
    float E_exc_val, float E_inh_val,
    float tau_exc, float tau_inh,
    float dt, int num_steps, int refr_steps,
    int N, int num_pn, int kc_start, int kc_end, int mbon_start, int mbon_end,
    float input_weight,
    float feedback_strength, float feedback_inhibition, float w_max_norm,
    float tau_kc_trace, float tau_mbon_trace, float tau_elig,
    int training_mode,
    int* mbon_counts_out,
    int* kc_spiked_out,
    float* eligibility_out
) {
    int gi = blockIdx.x;
    int tid = threadIdx.x;

    int num_kc = kc_end - kc_start;
    int num_mbon = mbon_end - mbon_start;

    extern __shared__ float shared[];
    float* v = shared;
    float* g_exc = shared + N;
    float* g_inh = shared + 2 * N;
    int* refr = (int*)(shared + 3 * N);
    int* spiked = (int*)(shared + 3 * N) + N;

    float decay_exc = expf(-dt / tau_exc);
    float decay_inh = expf(-dt / tau_inh);
    float decay_kc = expf(-dt / tau_kc_trace);
    float decay_mbon = expf(-dt / tau_mbon_trace);
    float decay_elig = expf(-dt / tau_elig);

    int base_w = gi * N * N;
    int base_vt = gi * N;

    if (tid < N) {
        v[tid] = V_rest[tid];
        g_exc[tid] = 0.0f;
        g_inh[tid] = 0.0f;
        refr[tid] = 0;
        spiked[tid] = 0;
    }
    __syncthreads();

    float kc_trace_local = 0.0f;
    int kc_spiked_local = 0;
    int mbon_count_local = 0;
    float mbon_trace_local = 0.0f;

    for (int t = 0; t < num_steps; t++) {
        if (tid < num_pn) {
            if (input_spikes[t * num_pn + tid]) {
                g_exc[tid] += input_weight;
            }
        }
        __syncthreads();

        if (tid < N) {
            float I_syn = g_exc[tid] * (E_exc_val - v[tid]) + g_inh[tid] * (E_inh_val - v[tid]);
            float dv_val = (-(v[tid] - V_rest[tid]) + I_syn / g_L[tid]) / tau_m[tid] * dt;
            if (refr[tid] > 0) {
                refr[tid]--;
            } else {
                v[tid] += dv_val;
            }
        }
        __syncthreads();

        spiked[tid] = 0;
        if (tid < N && v[tid] > V_thresh[base_vt + tid] && refr[tid] == 0) {
            v[tid] = V_reset[tid];
            refr[tid] = refr_steps;
            spiked[tid] = 1;
        }
        __syncthreads();

        if (tid < N) {
            float exc_sum = 0.0f;
            float inh_sum = 0.0f;
            for (int i = 0; i < N; i++) {
                if (spiked[i]) {
                    float we = W_exc[base_w + i * N + tid];
                    float wi = W_inh[base_w + i * N + tid];
                    if (we > 0.0f) exc_sum += we;
                    if (wi > 0.0f) inh_sum += wi;
                }
            }
            g_exc[tid] += exc_sum;
            g_inh[tid] += inh_sum;
        }
        __syncthreads();

        if (tid >= kc_start && tid < kc_end && spiked[tid]) {
            int ki = tid - kc_start;
            kc_spiked_local = 1;
            kc_trace_local += 1.0f;
            int base_km = gi * num_kc * num_mbon;
            for (int m = 0; m < num_mbon; m++) {
                atomicAdd(&g_exc[mbon_start + m], kc_mbon_w[base_km + ki * num_mbon + m] * 1e-9f);
            }
        }

        if (tid >= mbon_start && tid < mbon_end && spiked[tid]) {
            int mi = tid - mbon_start;
            mbon_count_local += 1;
            mbon_trace_local += 1.0f;
            int base_km = gi * num_kc * num_mbon;
            for (int ki = 0; ki < num_kc; ki++) {
                float fb = kc_mbon_w[base_km + ki * num_mbon + mi] / w_max_norm;
                if (fb > 0.1f) {
                    atomicAdd(&g_exc[kc_start + ki], fb * feedback_strength * 1e-9f);
                } else {
                    atomicAdd(&g_inh[kc_start + ki], feedback_inhibition * 1e-9f);
                }
            }
        }
        __syncthreads();

        kc_trace_local *= decay_kc;
        mbon_trace_local *= decay_mbon;

        if (tid < N) {
            g_exc[tid] *= decay_exc;
            g_inh[tid] *= decay_inh;
        }
        __syncthreads();
    }

    if (tid >= kc_start && tid < kc_end) {
        int ki = tid - kc_start;
        kc_spiked_out[gi * num_kc + ki] = kc_spiked_local;
    }
    if (tid >= mbon_start && tid < mbon_end) {
        int mi = tid - mbon_start;
        mbon_counts_out[gi * num_mbon + mi] = mbon_count_local;
    }
}
'''


class FusedGPUSimulator:
    def __init__(self, num_neurons, num_pn, kc_start, kc_end, mbon_start, mbon_end,
                 dt=0.00005, num_steps=2000, refr_steps=40):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

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

        self.kernel = cp.RawKernel(FUSED_KERNEL_CODE, 'simulate_fused')
        shared_size = (3 * num_neurons * 4) + (2 * num_neurons * 4)
        self.shared_size = shared_size
        self.threads = num_neurons

    def simulate_batch(self, pop_W_exc, pop_W_inh, pop_kc_mbon_w,
                       input_spikes, pop_V_thresh,
                       tau_m, V_rest, V_reset, g_L,
                       E_exc_val, E_inh_val, tau_exc, tau_inh,
                       input_weight,
                       feedback_strength, feedback_inhibition, w_max_norm,
                       tau_kc_trace, tau_mbon_trace, tau_elig,
                       training_mode=0):
        pop_size = pop_W_exc.shape[0]

        W_exc_g = cp.asarray(pop_W_exc, dtype=cp.float32)
        W_inh_g = cp.asarray(pop_W_inh, dtype=cp.float32)
        kc_mbon_g = cp.asarray(pop_kc_mbon_w, dtype=cp.float32)
        spikes_g = cp.asarray(input_spikes, dtype=cp.uint8)
        V_thresh_g = cp.asarray(pop_V_thresh, dtype=cp.float32)
        tau_m_g = cp.asarray(tau_m, dtype=cp.float32)
        V_rest_g = cp.asarray(V_rest, dtype=cp.float32)
        V_reset_g = cp.asarray(V_reset, dtype=cp.float32)
        g_L_g = cp.asarray(g_L, dtype=cp.float32)

        mbon_out = cp.zeros((pop_size, self.num_mbon), dtype=cp.int32)
        kc_out = cp.zeros((pop_size, self.num_kc), dtype=cp.int32)
        elig_out = cp.zeros((pop_size, self.num_kc, self.num_mbon), dtype=cp.float32)

        self.kernel(
            (pop_size,), (self.threads,),
            (W_exc_g, W_inh_g, kc_mbon_g,
             spikes_g, V_thresh_g,
             tau_m_g, V_rest_g, V_reset_g, g_L_g,
             cp.float32(E_exc_val), cp.float32(E_inh_val),
             cp.float32(tau_exc), cp.float32(tau_inh),
             cp.float32(self.dt), cp.int32(self.num_steps), cp.int32(self.refr_steps),
             cp.int32(self.N), cp.int32(self.num_pn),
             cp.int32(self.kc_start), cp.int32(self.kc_end),
             cp.int32(self.mbon_start), cp.int32(self.mbon_end),
             cp.float32(input_weight),
             cp.float32(feedback_strength), cp.float32(feedback_inhibition),
             cp.float32(w_max_norm),
             cp.float32(tau_kc_trace), cp.float32(tau_mbon_trace), cp.float32(tau_elig),
             cp.int32(training_mode),
             mbon_out, kc_out, elig_out),
            shared_mem=self.shared_size
        )

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(mbon_out), cp.asnumpy(kc_out), cp.asnumpy(elig_out)


def benchmark_fused(neuron_counts=None):
    if neuron_counts is None:
        neuron_counts = [300, 500, 639]

    from src.simulator.growing_brain import BrainConfig
    import time

    print("Fused GPU Kernel Benchmark")
    print("=" * 50)

    for n_kc in neuron_counts:
        cfg = BrainConfig(num_pn=128, num_kc=n_kc, num_mbon=10, num_apl=1)
        N = cfg.num_neurons
        tau_m, V_rest, V_reset, g_L = cfg.build_params()

        pop_size = 5
        W_exc = np.random.rand(pop_size, N, N).astype(np.float32) * 1e-9
        W_inh = np.random.rand(pop_size, N, N).astype(np.float32) * 1e-9
        kc_mbon = np.full((pop_size, cfg.num_kc, cfg.num_mbon), 5.0, dtype=np.float32)
        V_thresh = np.full((pop_size, N), -0.055, dtype=np.float32)
        spikes = (np.random.rand(2000, 128) < 0.005).astype(np.uint8)

        sim = FusedGPUSimulator(N, 128, cfg.kc_start, cfg.kc_end,
                                cfg.mbon_start, cfg.mbon_end)

        sim.simulate_batch(W_exc, W_inh, kc_mbon, spikes, V_thresh,
                           tau_m, V_rest, V_reset, g_L,
                           0.0, -0.080, 0.005, 0.010, 100e-9,
                           0.3, 0.4, 15.0, 0.020, 0.020, 0.040, 0)

        t0 = time.time()
        n_trials = 10
        for _ in range(n_trials):
            sim.simulate_batch(W_exc, W_inh, kc_mbon, spikes, V_thresh,
                               tau_m, V_rest, V_reset, g_L,
                               0.0, -0.080, 0.005, 0.010, 100e-9,
                               0.3, 0.4, 15.0, 0.020, 0.020, 0.040, 0)
        cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - t0) / n_trials

        print(f"  {N:4d} neurons ({n_kc} KC, {pop_size} pop): {gpu_time*1000:.1f}ms/batch ({gpu_time/pop_size*1000:.1f}ms/org)")

    return True
