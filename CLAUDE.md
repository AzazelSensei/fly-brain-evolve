# EvoDrosophila — Project Guidelines

## Superpowers Integration
This project uses [superpowers](https://github.com/obra/superpowers) skills from `.superpowers/` directory. Use skills for: brainstorming, writing-plans, executing-plans, test-driven-development, systematic-debugging.

## Code Standards
- Self-documenting code — NO inline comments
- All configs from YAML files
- Random seeds set everywhere for reproducibility
- Commit messages in Turkish

## Academic Documentation
This project produces an academic paper. Every decision, error, correction, and cause-effect relationship must be documented in `docs/journal/`. Each experiment run logs results as JSON.

## Architecture
Drosophila mushroom body connectome: PN → KC → MBON with APL inhibition.
Stack: Python 3.11+, Brian2, custom genetic algorithm, NumPy/SciPy/Matplotlib.
