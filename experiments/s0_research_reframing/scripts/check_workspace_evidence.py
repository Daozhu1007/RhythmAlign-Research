"""Read-only spot checks on historical assets; writes only S0 evidence receipts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/s0_research_reframing'
FIX = ROOT / 'experiments/r5_fix_validity_repair'

paths = [
    'experiments/independent_review_r0_r56/INDEPENDENT_REVIEW.md',
    'experiments/r5_fix_validity_repair/REPORT.md',
    'experiments/r5_fix_validity_repair/corrected_comparison_matrix.json',
    'experiments/r5_fix_validity_repair/logs/evaluation_corrected.json',
    'experiments/r5_fix_validity_repair/scripts/fixenv.py',
    'experiments/r1_golden_sample/scripts/06_cancel.py',
    'experiments/r1_golden_sample/work/audit_metrics.json',
    'experiments/r3_audio_query/logs/00_env_audit.md',
    'experiments/r3_audio_query/third_party/CLAPSep/model/CLAPSep.py',
    'experiments/r3_audio_query/third_party/CLAPSep/model/CLAPSep_decoder.py',
    'experiments/r2_target_separation/third_party/AudioSep/config/audiosep_base.yaml',
]
receipt = {'date': '2026-09-10', 'scope': 'Independent numerical spot checks, not new separation or listening',
           'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}}
raw, sr = sf.read(ROOT / 'experiments/r1_golden_sample/outputs/golden_raw.wav', always_2d=True)
receipt['r1_total_mixture_energy_reduction_db'] = {}
for name in ['A1', 'A2', 'A3']:
    x, fs = sf.read(ROOT / f'experiments/r1_golden_sample/outputs/{name}_residual.wav', always_2d=True)
    assert fs == sr and x.shape == raw.shape
    receipt['r1_total_mixture_energy_reduction_db'][name] = float(10 * np.log10(np.sum(raw**2) / np.sum(x**2)))
raw_d, sr_d = sf.read(ROOT / 'experiments/r56_precision_cleanup/outputs/holdoutD_raw.wav', always_2d=True)
for tag in ['e0', 'oracle']:
    env = np.load(FIX / f'work/env_holdoutD_{tag}_fixed.npy')
    wav_name = 'E0' if tag == 'e0' else 'oracle'
    stem, fs = sf.read(FIX / f'outputs/holdoutD_{wav_name}_C1_fixed_interaction.wav', always_2d=True)
    assert fs == sr_d and len(env) == len(stem) == len(raw_d)
    sl = slice(6 * fs, 8 * fs)
    eraw, eout = np.sum(raw_d[sl]**2), np.sum(stem[sl]**2)
    receipt[f'holdout_d_{tag}'] = {
        'max_gain_step': float(np.max(np.abs(np.diff(env)))),
        'rest_exposure_gt0_seconds': float(np.count_nonzero(env[sl] > 0) / fs),
        'rest_output_to_raw_energy_db': float(10*np.log10(eout/eraw)) if eout else None,
        'rest_output_is_digital_zero': bool(eout == 0),
        'max_error_vs_stored_gain_times_raw': float(np.max(np.abs(stem - env[:, None]*raw_d))),
    }
matrix = json.loads((FIX / 'corrected_comparison_matrix.json').read_text(encoding='utf-8'))
receipt['selected_saved_metrics_not_reestimated'] = {
    key: [{k: row[k] for k in ['case','stem_onsets_matched_50','stem_unmatched_output_onsets_50',
                             'median_event_peak_ratio','spearman_amp_all']} for row in matrix[key]]
    for key in ['golden','holdoutD']
}
(OUT / 'WORKSPACE_EVIDENCE_CHECKS.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k not in ['sha256','selected_saved_metrics_not_reestimated']}, indent=2))
