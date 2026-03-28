# OASIS Crisis Communication Optimizer – A/B test government messaging strategies before the next outbreak

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-76%20passed-brightgreen.svg)]()

> Simulate public health messaging strategies on synthetic social networks to quantify belief alignment without risking real-world misinformation.

## Install

```bash
git clone https://github.com/dakshjain-1616/oasis-crisis-communication-opt
cd oasis-crisis-communication-opt
pip install -r requirements.txt
```

## What problem this solves

When public health agencies face an outbreak, they cannot ethically A/B test messaging strategies on real populations. Existing simulation tools like standard agent-based models lack LLM-driven nuance, while full-scale LLM simulations are too costly for iterative strategy tuning. This tool fills the gap by providing a low-cost, reproducible sandbox where you can compare authoritative vs. empathetic tones, timing delays, and misinformation resilience using INTERVIEW actions and REPOST metrics before deployment.

## Real world examples

```python
# Run a 10-timestep simulation with default government strategy
from oasis_crisis_communi.simulation import run_simulation
results = run_simulation(steps=10, strategy="authoritative")
# Output: {'alignment_score': 0.85, 'repost_rate': 0.42}
```

```python
# Extract belief trajectory for specific agent ID
from oasis_crisis_communi.belief_tracker import get_agent_trajectory
trajectory = get_agent_trajectory(agent_id="citizen_001", history=results)
# Output: [0.1, 0.3, 0.5, 0.8, 0.9]
```

```python
# Generate comparative report for multiple strategies
from oasis_crisis_communi.analyzer import compare_strategies
report = compare_strategies(["early", "late", "empathetic"])
# Output: PDF saved to reports/strategy_comparison.pdf
```

## Who it's for

Data scientists and policy analysts preparing for pandemic response or disaster management. You need this when you are tasked with designing a communication campaign but lack historical data to validate whether an authoritative tone or an empathetic approach will yield better compliance in a specific demographic.

## Quickstart

```bash
python examples/01_quick_start.py
```

## Key features

- Gradio UI for real-time strategy configuration (timing, tone, frequency)
- Mock mode simulation requiring no API keys for local development
- Belief tracking via INTERVIEW actions at each timestep
- Automated PDF report generation for stakeholder review

## Run tests

```bash
tests/test_simulation.py::test_belief_tracker_population_alignment PASSED [ 98%]
tests/test_simulation.py::test_belief_tracker_agent_trajectory PASSED    [100%]

=============================== warnings summary ===============================
tests/test_scenarios.py: 7 warnings
tests/test_sensitivity.py: 189 warnings
tests/test_simulation.py: 17 warnings
  /root/projects/oasis-crisis-communication-optimizer/oasis_crisis_communi/simulation.py:101: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
    default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"

tests/test_simulation.py::test_gradio_ui_importable
  /usr/local/lib/python3.12/dist-packages/gradio/routes.py:63: PendingDeprecationWarning: Please use `import python_multipart` instead.
    from multipart.multipart import parse_options_header

tests/test_simulation.py::test_gradio_create_ui
  /usr/local/lib/python3.12/dist-packages/gradio/utils.py:98: DeprecationWarning: There is no current event loop
    asyncio.get_event_loop()

tests/test_simulation.py::test_gradio_create_ui
tests/test_simulation.py::test_gradio_create_ui
  /usr/local/lib/python3.12/dist-packages/gradio/routes.py:1215: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
          
    @app.on_event("startup")

tests/test_simulation.py::test_gradio_create_ui
tests/test_simulation.py::test_gradio_create_ui
  /usr/local/lib/python3.12/dist-packages/fastapi/applications.py:4599: DeprecationWarning: 
          on_event is deprecated, use lifespan event handlers instead.
  
          Read more about it in the
          [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
          
    return self.router.on_event(event_type)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 76 passed, 219 warnings in 43.79s =======================
```

## Project structure

```
oasis-crisis-communication-opt/
├── oasis_crisis_communi/      ← main library
├── tests/          ← test suite
├── examples/       ← demo scripts
└── requirements.txt
```