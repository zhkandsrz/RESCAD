# External data and adapted baselines

## Data

Fusion-derived experiments depend on the Autodesk Fusion 360 Gallery assembly dataset. Obtain the data and inspect its current terms from the upstream project:
https://github.com/AutodeskAILab/Fusion360GalleryDataset

Raw Fusion archives and redistributed CAD collections are not included here. The source includes data-processing utilities, but it does not supply all experiment-specific splits, face mappings, and supervised interface labels needed to reconstruct the paper's exact benchmark.

The offline demo creates new procedural solids using the included CadQuery generator; it does not download or copy third-party CAD geometry. Demo results are not part of the reported confirmation results.

Generated-part experiments used external generation methods. Their pretrained models and output collections are not redistributed by this release; their licenses and generation settings must be handled separately.

## Baselines

`linkcad_joinable_style_baseline_v1.py` implements a task adaptation inspired by JoinABLe (Learning Bottom-Up Assembly of Parametric CAD Joints). It is not the authors' original checkpoint or a reproduction of their native evaluation. See the upstream project for the original method and license: https://github.com/AutodeskAILab/JoinABLe

`linkcad_spada_style_baseline_v1.py` implements the paper's adapted generate/check/revise assembly protocol inspired by SPADA. It is not an official implementation or a native SPADA benchmark result.

`linkcad_symbolic_baseline_v1.py` provides symbolic selection; `linkcad_pure_llm_baseline_v1.py` and `linkcad_pure_llm_execution_v1.py` provide direct-LLM assembly and execution. Other architecture adaptations are retained because the shared training entry point imports them; inclusion does not make them paper main-table baselines or native state-of-the-art reproductions.

Baseline comparisons require the same task inputs and clearly declared search and execution budgets. Provider model availability and replies can change; a new live API run is not byte-identical replay of a historical run. This release includes no API credentials and does not include all frozen provider-response records.

## Third-party software

CadQuery, Open CASCADE/OCP, PyTorch, NumPy, igraph, and other dependencies are installed separately and retain their upstream licenses. No upstream dependency source tree or model download is vendored into this release. Keep required attribution when extending or redistributing the project.
