# Reproduction map

## Paper components and source files

| Component | Entry point / implementation |
| --- | --- |
| Whole-request language compiler | `neurocad/linkcad_request_graph_compiler_v1.py` |
| Per-connection attribute compiler | `neurocad/linkcad_language_planner_v1.py` |
| B-rep message passing | `neurocad/linkcad_brep_encoder_v1.py` |
| Face/edge groups and primitive graph | `neurocad/linkcad_primitive_graph_v2.py` |
| Endpoint constraints and matching | `neurocad/linkcad_port_contract_v1.py`, `neurocad/linkcad_primitive_cache_v2.py` |
| Two-branch model | `neurocad/linkcad_port_conditioned_model_v6.py` |
| Geometric candidate-pair masking | `neurocad/linkcad_port_constraint_model_v9.py` |
| Unambiguous-interface dispatch | `neurocad/linkcad_contract_dispatch_v1.py` |
| Pair execution and finite pose search | `neurocad/linkcad_kinematic_execution_v2.py`, `neurocad/linkcad_kinematic_execution_v3.py` |
| Assembly pose composition and collision check | `neurocad/linkcad_global_assembly_execution_v1.py` |
| Product-structured STEP export | `neurocad/linkcad_assembly_export_v1.py` |
| Geometry and task metrics | `neurocad/linkcad_assembly_geometry_evaluation_v1.py`, `neurocad/tools/evaluate_linkcad_volume_iou_v1.py` |

The source includes supporting modules reached by these implementations. Internal version numbers identify implementations, not successive public benchmark results. The V2 kinematic module retains only shared primitive frames, pose proposals, and inference; obsolete V2 execution/oracle routines and their historical dependencies have been removed. V3 pair execution and the learned model implementations are unchanged.

## Inputs and outputs

`public.json` contains a `queries` array. Each query includes named component slots (`roles` in the serialized schema), `candidate_sets`, and `functional_edges`. Each candidate binds a STEP path and file hash. Each connection identifies its endpoints, requested motion, and endpoint constraints when available.

At inference, candidate STEP paths are interpreted relative to the supplied dataset root. Private reference assignments and interface labels are not prediction inputs. Prediction files contain ranked assignments and interface/program alternatives; exact-execution files record attempts and rejection/acceptance results. An exported STEP assembly includes component instances and their placement, not an industrial assembly-path plan.

The demo generates examples of these schemas. Its `private_targets.sealed.json` filename is an existing protocol convention: it is plain JSON separated from inference, **not encryption**. Do not treat this filename as an access-control mechanism.

## Released weights and inference

`checkpoints/manifest.json` lists the three original state dictionaries and their hashes. Load only trusted weights with `torch.load(..., weights_only=True)`.

Given a prepared public protocol and STEP root:

```bash
python -m neurocad.tools.materialize_linkcad_primitive_cache_v2 --public data/public.json --dataset-root data/step --output-dir outputs/cache --workers 1
python -m neurocad.tools.materialize_linkcad_public_predictions_v2 --public data/public.json --primitive-cache outputs/cache --checkpoint checkpoints/rescad_seed1701.pt --model-kind port_v9 --seed 1701 --beam-size 25 --interface-alternatives-per-edge 8 --output outputs/predictions.json
python -m neurocad.tools.materialize_linkcad_contract_dispatch_v1 --public data/public.json --primitive-cache outputs/cache --predictions outputs/predictions.json --output outputs/dispatched_predictions.json
```

The released model uses hidden dimension 64, three message-passing layers per encoder, attention pooling, up to 16 interface groups per candidate, and no global-score head at decoding. Repeat prediction with each matching seed/checkpoint for a three-seed evaluation. Do not pick a seed using test outcomes.

## Training and evaluation

Training entry point: `python -m neurocad.tools.train_linkcad_factorized_pilot_v1 --help`.
Interface pretraining entry point: `python -m neurocad.tools.pretrain_linkcad_primitive_proposer_v2 --help`.
The training loader expects public protocols plus reference assignments and interface supervision; these cannot be inferred from arbitrary STEP files alone. The complete historical training data, initialization stages, and experiment-specific recipe are not bundled, so the CLI defaults must not be described as the exact recipe for reproducing the released checkpoints.

The paper uses AdamW, learning rate 0.0003, weight decay 0.0001, and seeds 1701/1702/1703. CLI flags and model state are provided; full from-scratch reproduction still requires the missing data/initialization artifacts.

Current execution/export entry points (each exposes `--help`):

```bash
python -m neurocad.tools.run_rescad_pair_execution --help
python -m neurocad.tools.run_linkcad_global_assembly_execution_v1 --help
python -m neurocad.tools.export_linkcad_assembly_package_v1 --help
python -m neurocad.tools.evaluate_linkcad_assembly_geometry_v1 --help
python -m neurocad.tools.evaluate_linkcad_volume_iou_v1 --help
```

Pair execution takes public ranked predictions and a STEP root; global execution
composes those pair results; export writes accepted assemblies. The example
provides an executable illustration of this data flow. Its first-proposal
configuration is not the paper's bounded multi-proposal evaluation.

Missing outputs must remain failures for failure-aware metrics; conditional
geometric distances must retain their valid denominators. Keep collision
freedom, recorded execution acceptance, intended component/interface labels,
and independently checked physical contact distinct. The geometry evaluator's
`intended_exact` combines label agreement with the recorded executor status;
it is not an additional independent final-contact test. Trimming this release
does not rerun or strengthen the evidence behind historical benchmark numbers.

## Release scope

This is a core source-and-model release, not a claim that every paper result was
rerun during packaging. It excludes private logs, raw API responses, the full
dataset archive, unreviewed third-party binaries, manuscript editing history,
local environments, obsolete executors, and historical experiment/baseline
drivers. Third-party attribution remains in DATA_AND_BASELINES.md.

Validation of this trimmed snapshot: 47 offline tests passed. The procedural
demo generated 16 candidates, produced eight ranked hypotheses, and exported
one assembly accepted by its pair/global checks. No live API calls were made.
Seven upstream ezdxf/Pyparsing deprecation warnings were observed.
