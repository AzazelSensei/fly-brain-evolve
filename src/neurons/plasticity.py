STDP_MODEL = {
    "pre": """
        g_exc_post += w * nS
        A_pre_trace += A_pre
        w = clip(w + A_post_trace * A_post, w_min, w_max)
    """,
    "post": """
        A_post_trace += A_post_factor
        w = clip(w + A_pre_trace * A_pre, w_min, w_max)
    """,
}

STDP_EQUATIONS = """
dA_pre_trace/dt = -A_pre_trace / tau_pre : 1 (event-driven)
dA_post_trace/dt = -A_post_trace / tau_post : 1 (event-driven)
w : 1
"""

STDP_DEFAULTS = {
    "tau_pre": 0.020,
    "tau_post": 0.020,
    "A_pre": 0.01,
    "A_post": -0.0105,
    "A_post_factor": 1.0,
    "w_max": 1.0,
    "w_min": 0.0,
}


def get_stdp_params(config=None):
    params = dict(STDP_DEFAULTS)
    if config and "plasticity" in config:
        for key, value in config["plasticity"].items():
            if key == "a_pre":
                params["A_pre"] = value
            elif key == "a_post":
                params["A_post"] = value
            else:
                params[key] = value
    return params
