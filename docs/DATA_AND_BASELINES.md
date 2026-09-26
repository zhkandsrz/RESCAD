# Data and attribution

Fusion-derived experiments use the Autodesk Fusion 360 Gallery assembly
dataset. Obtain it and review its terms from the upstream project:
https://github.com/AutodeskAILab/Fusion360GalleryDataset

Raw Fusion archives, experiment-specific splits, face mappings, and interface
labels are not bundled. The offline example generates new CadQuery solids and
does not redistribute third-party CAD geometry. Generated-part experiment
collections and external generation models are also outside this core release.

## Shared model variants

The training implementation imports several earlier encoder/scoring variants
for initialization and comparison. In particular,
`linkcad_joinable_style_baseline_v1.py` is a task-specific adaptation inspired
by JoinABLe, not its official checkpoint or native evaluation:
https://github.com/AutodeskAILab/JoinABLe

The shared AutoMate-/Pointer-CAD-style scorers are likewise implementation
adaptations, not official upstream reproductions. Their presence does not make
this snapshot a complete baseline benchmark. Historical direct-LLM and
generate/check/revise API runners are not included in the trimmed snapshot.

CadQuery, Open CASCADE/OCP, PyTorch, NumPy, igraph, and all other dependencies
are installed separately and retain their upstream licenses. This repository's
MIT license does not relicense those projects or external datasets/models.
