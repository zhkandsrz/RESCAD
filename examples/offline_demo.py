"""Exercise released inference and CAD execution on new synthetic geometry.

This is an offline software demo, not a reported test-set evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocad.linkcad_procedural_transfer_v1 import (
    ProceduralTransferConfigV1, materialize_procedural_transfer_v1,
)
from neurocad.tools.materialize_linkcad_primitive_cache_v2 import materialize_cache_v2
from neurocad.linkcad_primitive_cache_v2 import LinkCADPrimitiveGraphCacheV2
from neurocad.linkcad_kinematic_execution_v2 import materialize_public_primitive_predictions_v2
from neurocad.linkcad_kinematic_execution_v3 import execute_public_prediction_subset_v3
from neurocad.linkcad_global_assembly_execution_v1 import execute_global_assembly_subset_v1
from neurocad.linkcad_assembly_export_v1 import export_assembly_package_v1


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    parser.add_argument('--skip-exact', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Use a new, empty output directory for this demo.')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / 'checkpoints/rescad_seed1701.pt'
    print('Generating one new synthetic request and 16 STEP candidates.', flush=True)
    materialize_procedural_transfer_v1(
        config=ProceduralTransferConfigV1(seed=271828, query_count=1),
        dataset_root=out / 'step', artifact_root=out / 'protocol',
        historical_step_sha256s=set(), code_revision='offline-software-demo',
        checkpoint_paths=(checkpoint,),
    )
    public_path = out / 'protocol/public.json'
    public = json.loads(public_path.read_text(encoding='utf-8'))
    print('Extracting B-rep graph features.', flush=True)
    cache_manifest = materialize_cache_v2(
        public_paths=[public_path], dataset_root=out / 'step', output_dir=out / 'cache',
        workers=1, shard_size=128,
    )
    if cache_manifest['exclusions']:
        raise RuntimeError('Demo graph extraction failed; inspect the cache manifest.')
    cache = LinkCADPrimitiveGraphCacheV2(out / 'cache')
    cache.load_all()
    print('Scoring public candidates with frozen seed 1701.', flush=True)
    predictions = materialize_public_primitive_predictions_v2(
        public=public, cache=cache, checkpoint_path=checkpoint,
        model_kind='port_v9', seed=1701, beam_size=25,
        interface_alternatives_per_edge=8,
    )
    write(out / 'predictions.json', predictions)
    summary = {'scope': 'synthetic_offline_software_demo', 'query_count': len(public['queries']),
               'candidate_count': cache_manifest['candidate_step_count'],
               'prediction_read_private_targets': False, 'live_llm_calls': 0,
               'hypothesis_count': sum(len(row['hypotheses']) for row in predictions['rows'])}
    if not args.skip_exact:
        print('Checking the first proposal with the CAD kernel.', flush=True)
        pair = execute_public_prediction_subset_v3(
            public=public, predictions=predictions, dataset_root=out / 'step',
            query_limit=1, hypotheses_per_query=1, alternatives_per_edge=3,
            maximum_attempts_per_query=25, maximum_accepted_poses_per_edge=8,
        )
        write(out / 'pair_execution.json', pair)
        assembled = execute_global_assembly_subset_v1(
            public=public, predictions=predictions, pair_execution=pair,
            dataset_root=out / 'step', query_limit=1,
        )
        write(out / 'assembly_execution.json', assembled)
        summary['assembly_status_counts'] = assembled['status_counts']
        for row in assembled['rows']:
            if row['status'] != 'accepted_global_assembly':
                continue
            query = next(q for q in public['queries'] if q['query_id'] == row['query_id'])
            prediction = next(p for p in predictions['rows'] if p['query_id'] == row['query_id'])
            export_assembly_package_v1(
                query=query, hypothesis=prediction['hypotheses'][0], global_execution=row,
                pair_execution_rows=pair['rows'], dataset_root=out / 'step',
                output_step=out / 'assembly.step', output_manifest=out / 'assembly_manifest.json',
            )
    write(out / 'demo_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
