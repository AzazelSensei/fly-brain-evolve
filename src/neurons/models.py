LIF_EQUATIONS = """
dv/dt = (-(v - V_rest) + I_syn/g_L) / tau_m : volt (unless refractory)
dg_exc/dt = -g_exc / tau_exc : siemens
dg_inh/dt = -g_inh / tau_inh : siemens
I_syn = g_exc * (E_exc - v) + g_inh * (E_inh - v) : amp
"""

NEURON_DEFAULTS = {
    "pn": {
        "tau_m": 0.010,
        "V_rest": -0.070,
        "V_thresh": -0.055,
        "V_reset": -0.070,
        "tau_exc": 0.005,
        "tau_inh": 0.010,
        "refractory": 0.002,
        "g_L": 25e-9,
        "E_exc": 0.0,
        "E_inh": -0.080,
    },
    "kc": {
        "tau_m": 0.020,
        "V_rest": -0.070,
        "V_thresh": -0.050,
        "V_reset": -0.070,
        "tau_exc": 0.005,
        "tau_inh": 0.010,
        "refractory": 0.002,
        "g_L": 25e-9,
        "E_exc": 0.0,
        "E_inh": -0.080,
    },
    "mbon": {
        "tau_m": 0.015,
        "V_rest": -0.070,
        "V_thresh": -0.050,
        "V_reset": -0.070,
        "tau_exc": 0.005,
        "tau_inh": 0.010,
        "refractory": 0.002,
        "g_L": 25e-9,
        "E_exc": 0.0,
        "E_inh": -0.080,
    },
    "apl": {
        "tau_m": 0.010,
        "V_rest": -0.070,
        "V_thresh": -0.045,
        "V_reset": -0.070,
        "tau_exc": 0.005,
        "tau_inh": 0.010,
        "refractory": 0.002,
        "g_L": 25e-9,
        "E_exc": 0.0,
        "E_inh": -0.080,
    },
}


def get_neuron_equations():
    return LIF_EQUATIONS


def get_neuron_params(neuron_type, config=None):
    params = dict(NEURON_DEFAULTS[neuron_type])
    if config and neuron_type in config.get("neuron", {}):
        overrides = config["neuron"][neuron_type]
        for key, value in overrides.items():
            mapped_key = key if key.startswith(("V_", "E_", "g_")) else key
            if key == "v_rest":
                mapped_key = "V_rest"
            elif key == "v_thresh":
                mapped_key = "V_thresh"
            elif key == "v_reset":
                mapped_key = "V_reset"
            params[mapped_key] = value
    return params
