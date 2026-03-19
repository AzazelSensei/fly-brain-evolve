from src.neurons.models import get_neuron_equations, get_neuron_params


class TestNeuronModels:
    def test_equations_string(self):
        eqs = get_neuron_equations()
        assert "dv/dt" in eqs
        assert "g_exc" in eqs
        assert "g_inh" in eqs

    def test_params_keys(self):
        params = get_neuron_params("kc")
        required_keys = ["tau_m", "V_rest", "V_thresh", "V_reset", "tau_exc", "tau_inh", "refractory"]
        for key in required_keys:
            assert key in params

    def test_all_neuron_types(self):
        for ntype in ["pn", "kc", "mbon", "apl"]:
            params = get_neuron_params(ntype)
            assert params["tau_m"] > 0
