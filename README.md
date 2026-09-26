# RESCAD

Core research code for language-grounded CAD assembly. Given a language request
and candidate STEP parts, RESCAD compiles connection requirements, applies
geometric filtering, ranks components and interfaces with two B-rep GNNs, and
composes the selected parts with a CAD kernel.

## Release contents

- `neurocad/`: compiler, graph extraction, two-branch model, filtering,
  training, execution, export, and evaluation with their required dependencies.
- `checkpoints/`: three original tensor-only model weights (seeds 1701–1703).
- `examples/offline_demo.py`: a generated CAD example requiring no API key.
- `tests/`: focused offline tests for the retained implementation.
- `docs/REPRODUCIBILITY.md`: source map, commands, and reproduction limits.

This core snapshot omits obsolete executors, historical experiment runners,
generation/transfer sweeps, baseline API pipelines, raw logs, and paper assets.
Internal `neurocad` and `linkcad_*` names are retained for compatibility.
Some shared model variants remain because training and initialization import them.

## Install

Tested with Python 3.12, CadQuery 2.8.0, and PyTorch on Windows:

```bash
conda create -n rescad --file environment-lock-win-64.txt
conda activate rescad
python -m pip install --no-deps --require-hashes -r requirements-torch-cu130-win-64.txt
python -m pip install --no-deps -e .
```

The Torch lock is for CPython 3.12, Windows, and CUDA 13.0. On another platform,
use `environment.yml` and install the appropriate official PyTorch build before
installing this package. CadQuery must select its compatible OCP/OCCT dependency.
Cross-platform numerical equivalence has not been established.

## Offline example

```bash
python -m pytest -q
python examples/offline_demo.py --output outputs/demo
```

The example generates one four-component request with 16 STEP candidates,
extracts B-rep features, ranks them using seed 1701, and attempts bounded
execution of the first proposal. Its structured requirements come from the
procedural generator, not a fresh LLM call. It is a software smoke test, not a
paper benchmark or evidence of success on arbitrary prompts.

The trimmed snapshot passed 47 offline tests; the example produced eight
ranked hypotheses and exported one assembly accepted by its execution checks.
These checks do not certify manufacturing correctness.

## Scope and credentials

The full Fusion benchmark, experiment-specific labels/caches, source-adaptation
weights, and complete baseline records are not included. This checkout alone
does **not** reproduce every paper table. See
[reproduction instructions](docs/REPRODUCIBILITY.md) and
[data and attribution](docs/DATA_AND_BASELINES.md).

Optional live language calls use environment variables such as
`DEEPSEEK_API_KEY` and may incur charges. Offline tests and the example make
no live API calls. Do not publish credentials or generated local logs.

## License

Original project code is released under the [MIT License](LICENSE).
External datasets, models, and dependencies retain their own licenses.
