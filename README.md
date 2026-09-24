# RESCAD

Research code for **Language-Grounded CAD Assembly via Geometry-Constrained Interface Resolution**.

Given a language request and candidate STEP components, RESCAD compiles the requested connections, filters geometrically incompatible choices, ranks components and local interfaces using two learned B-rep graph encoders, and executes the selected connections with a CAD kernel.
The method assembles supplied solids; it does not generate their geometry or certify manufacturing processes.

## What is included

- Language-compiler prompts, output validation, and environment-variable-based API clients.
- B-rep/primitive graph extraction, geometric masks, component and interface rankers.
- Three frozen Fusion-trained, tensor-only model checkpoints (seeds 1701, 1702, 1703); source-specific adaptation weights are not included.
- Training, prediction, exact execution, assembly export, and evaluation entry points.
- Adapted baseline implementations and focused, offline unit tests.
- A procedural CAD demonstration requiring neither a private dataset nor an API key.

Internal `neurocad` package names and `linkcad_*` module names are retained for compatibility with the experiments. The paper's method is RESCAD.

## Install

The recorded research environment uses Python 3.12, CadQuery 2.8.0, and PyTorch.
CadQuery owns the matching OCP/OCCT installation; do not independently install an incompatible OCP wheel.

For an exact Windows environment, with conda already installed:

```bash
conda create -n rescad --file environment-lock-win-64.txt
conda activate rescad
python -m pip install --no-deps --require-hashes -r requirements-torch-cu130-win-64.txt
python -m pip install --no-deps -e .
```

The Torch lock is specific to CPython 3.12 on Windows and CUDA 13.0.
For another platform, create an environment using `environment.yml` and install a compatible official PyTorch build for that platform before installing this package. Cross-platform numerical equivalence has not been established by this release.

## Offline smoke tests and demonstration

Run from the repository root:

```bash
python -m pytest -q
python examples/offline_demo.py --output outputs/demo
```

The demonstration generates one synthetic four-component request and 16 unposed STEP candidates, constructs their graph cache, and runs the released seed-1701 ranker without accessing the generated reference labels.
It also attempts bounded exact execution of the first proposal and exports the result if accepted.
The request's structured constraints are supplied by the procedural generator, not by a fresh LLM call. This is a software demonstration, not a reproduction of a paper test set or evidence of success on arbitrary prompts.

## Reproduction guide

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the input schemas, entry points, budget settings, and limitations of this release.
See [docs/DATA_AND_BASELINES.md](docs/DATA_AND_BASELINES.md) for external-data and baseline attribution.

The repository does **not** contain the full Fusion benchmark archive, all experiment-specific caches and labels, or every baseline response and checkpoint. Reproducing every paper table from this checkout alone is therefore not currently supported. The released code and three model checkpoints can be inspected and exercised independently with the included tests and generated example.

## Credentials

Live language calls are optional and incur provider charges. Provide credentials only through environment variables, such as `DEEPSEEK_API_KEY`. Never add credentials or local credential files to this repository. No live call is made by the offline tests or demo.

## License

Original project code is distributed under the [MIT License](LICENSE). Third-party dependencies, datasets, and external pretrained models remain subject to their own licenses; this repository does not relicense them.
