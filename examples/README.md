# Examples

Runnable scripts that demonstrate different features of the OASIS Crisis Communication Optimizer.
All scripts work without an API key (mock mode is enabled automatically).

## Running an example

```bash
# From the project root:
python examples/01_quick_start.py

# Or from the examples/ directory:
cd examples
python 01_quick_start.py
```

## Script index

| Script | Lines | What it demonstrates |
|--------|------:|----------------------|
| [01_quick_start.py](01_quick_start.py) | ~20 | Minimal working example — 3-strategy comparison, print winner |
| [02_advanced_usage.py](02_advanced_usage.py) | ~65 | Custom `StrategyConfig`, `BeliefTracker`, effect sizes, chart generation |
| [03_custom_config.py](03_custom_config.py) | ~70 | Env-var configuration, pre-built scenario cards, JSON report |
| [04_full_pipeline.py](04_full_pipeline.py) | ~100 | End-to-end: scenario → simulation → sensitivity → all outputs |

## Script details

### 01_quick_start.py — Minimal example
The shortest path from zero to results.
- Loads the three built-in default strategies
- Runs a 30-agent, 20-timestep comparison in mock mode
- Prints the winning strategy name and plain-English recommendation

### 02_advanced_usage.py — More features
Shows how to go beyond the defaults.
- Defines two custom `StrategyConfig` objects (Rapid Blitz, Slow Build)
- Computes Cohen's d effect sizes and time-to-60%-threshold
- Uses `BeliefTracker` to parse and score three sample interview responses
- Saves a multi-panel comparison chart to `outputs/`

### 03_custom_config.py — Environment-variable configuration
Every simulation parameter is readable from the environment, making the tool
scriptable and CI-friendly.
- Reads `NUM_AGENTS`, `NUM_TIMESTEPS`, `RANDOM_SEED`, `SCENARIO` from env
- Loads a named scenario card when `SCENARIO=<id>` is set
- Saves a JSON report to `OUTPUT_DIR`
- Lists all valid scenario IDs at the end

```bash
# Run with the bioterrorism scenario, 80 agents, 30 timesteps:
NUM_AGENTS=80 NUM_TIMESTEPS=30 SCENARIO=bioterrorism python examples/03_custom_config.py
```

### 04_full_pipeline.py — End-to-end workflow
The complete project capability in one script.
1. Loads the COVID pandemic scenario card
2. Runs multi-strategy comparison simulation
3. Runs sensitivity analysis across 10 random seeds (mean ± 95% CI)
4. Computes effect sizes, threshold timing, and belief distribution stats
5. Saves: `belief_alignment.csv`, comparison chart, belief-distribution chart,
   `results.json`, `report.html`, `interview_samples.json`
6. Prints a plain-English recommendation

## Available scenario IDs

| ID | Scenario | Severity |
|----|----------|----------|
| `covid_pandemic` | COVID-19 Pandemic | critical |
| `disease_outbreak` | Disease Outbreak | high |
| `natural_disaster` | Natural Disaster | medium |
| `bioterrorism` | Bioterrorism Event | critical |
