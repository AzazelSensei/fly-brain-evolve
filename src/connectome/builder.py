import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, StateMonitor,
    ms, mV, nS, second, Hz, volt, siemens,
    PoissonGroup, Network, prefs,
)
from src.neurons.models import get_neuron_equations, get_neuron_params
from src.neurons.plasticity import STDP_EQUATIONS, STDP_MODEL, get_stdp_params

prefs.codegen.target = "numpy"


class MushroomBodyBuilder:
    def __init__(self, config, connectome):
        self.config = config
        self.connectome = connectome
        self.num_pn = connectome["num_pn"]
        self.num_kc = connectome["num_kc"]
        self.num_mbon = connectome["num_mbon"]

    def build(self, enable_stdp=True, enable_monitors=True):
        eqs = get_neuron_equations()
        sw = self.config.get("synapse_weights", {})

        pn_params = get_neuron_params("pn", self.config)
        pn = NeuronGroup(
            self.num_pn, eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=pn_params["refractory"] * second,
            method="euler",
            namespace=self._brian2_namespace(pn_params),
            name="pn",
        )
        pn.v = pn_params["V_rest"] * volt

        kc_params = get_neuron_params("kc", self.config)
        kc = NeuronGroup(
            self.num_kc, eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=kc_params["refractory"] * second,
            method="euler",
            namespace=self._brian2_namespace(kc_params),
            name="kc",
        )
        kc.v = kc_params["V_rest"] * volt

        mbon_params = get_neuron_params("mbon", self.config)
        mbon = NeuronGroup(
            self.num_mbon, eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=mbon_params["refractory"] * second,
            method="euler",
            namespace=self._brian2_namespace(mbon_params),
            name="mbon",
        )
        mbon.v = mbon_params["V_rest"] * volt

        apl_params = get_neuron_params("apl", self.config)
        apl = NeuronGroup(
            1, eqs,
            threshold="v > V_thresh",
            reset="v = V_reset",
            refractory=apl_params["refractory"] * second,
            method="euler",
            namespace=self._brian2_namespace(apl_params),
            name="apl",
        )
        apl.v = apl_params["V_rest"] * volt

        pn_kc_syn = Synapses(pn, kc, "w_syn : 1", on_pre="g_exc_post += w_syn * nS", name="pn_kc_syn")
        pn_indices, kc_indices = np.nonzero(self.connectome["pn_kc"])
        pn_kc_syn.connect(i=pn_indices.astype(int), j=kc_indices.astype(int))
        pn_kc_range = sw.get("pn_kc_range", [1.5, 4.0])
        raw_weights = self.connectome["pn_kc"][pn_indices, kc_indices]
        scaled_weights = raw_weights / raw_weights.max() * (pn_kc_range[1] - pn_kc_range[0]) + pn_kc_range[0]
        pn_kc_syn.w_syn = scaled_weights

        if enable_stdp:
            stdp_params = get_stdp_params(self.config)
            kc_mbon_syn = Synapses(
                kc, mbon,
                STDP_EQUATIONS,
                on_pre=STDP_MODEL["pre"],
                on_post=STDP_MODEL["post"],
                namespace=self._stdp_namespace(stdp_params),
                name="kc_mbon_syn",
            )
        else:
            kc_mbon_syn = Synapses(
                kc, mbon,
                "w : 1",
                on_pre="g_exc_post += w * nS",
                name="kc_mbon_syn",
            )

        kc_all, mbon_all = np.nonzero(self.connectome["kc_mbon"])
        kc_mbon_syn.connect(i=kc_all.astype(int), j=mbon_all.astype(int))
        kc_mbon_range = sw.get("kc_mbon_range", [2.0, 5.0])
        raw_km = self.connectome["kc_mbon"][kc_all, mbon_all]
        scaled_km = raw_km / raw_km.max() * (kc_mbon_range[1] - kc_mbon_range[0]) + kc_mbon_range[0]
        kc_mbon_syn.w = scaled_km

        kc_apl_w = sw.get("kc_apl", 2.0)
        kc_apl_syn = Synapses(kc, apl, on_pre="g_exc_post += %.1f*nS" % kc_apl_w, name="kc_apl_syn")
        kc_apl_syn.connect()

        apl_kc_w = sw.get("apl_kc", 200.0)
        apl_kc_syn = Synapses(apl, kc, on_pre="g_inh_post += %.1f*nS" % apl_kc_w, name="apl_kc_syn")
        apl_kc_syn.connect()

        result = {
            "pn": pn,
            "kc": kc,
            "mbon": mbon,
            "apl": apl,
            "pn_kc_syn": pn_kc_syn,
            "kc_mbon_syn": kc_mbon_syn,
            "kc_apl_syn": kc_apl_syn,
            "apl_kc_syn": apl_kc_syn,
        }

        if enable_monitors:
            result["pn_monitor"] = SpikeMonitor(pn, name="pn_monitor")
            result["kc_monitor"] = SpikeMonitor(kc, name="kc_monitor")
            result["mbon_monitor"] = SpikeMonitor(mbon, name="mbon_monitor")

        return result

    def build_network(self, enable_stdp=True, enable_monitors=True):
        components = self.build(enable_stdp=enable_stdp, enable_monitors=enable_monitors)
        net = Network()
        for obj in components.values():
            net.add(obj)
        return net, components

    def _brian2_namespace(self, params):
        return {
            "tau_m": params["tau_m"] * second,
            "V_rest": params["V_rest"] * volt,
            "V_thresh": params["V_thresh"] * volt,
            "V_reset": params["V_reset"] * volt,
            "tau_exc": params["tau_exc"] * second,
            "tau_inh": params["tau_inh"] * second,
            "g_L": params["g_L"] * siemens,
            "E_exc": params["E_exc"] * volt,
            "E_inh": params["E_inh"] * volt,
        }

    def _stdp_namespace(self, params):
        return {
            "tau_pre": params["tau_pre"] * second,
            "tau_post": params["tau_post"] * second,
            "A_pre": params["A_pre"],
            "A_post": params["A_post"],
            "A_post_factor": params["A_post_factor"],
            "w_max": params["w_max"],
            "w_min": params["w_min"],
        }
