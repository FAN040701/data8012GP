"""Fitting-only input/geometry diagnostics; no prediction or model selection."""
import concurrent.futures as cf
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from continuation_common import ROOT, OUT


def one(case):
    folder = ROOT/'data/processed'/case
    meta = json.loads((folder/'metadata.json').read_text())
    spacing = meta['original_spacing_xyz_mm'][2]
    extent = (meta['original_shape_xyz'][2]-1)*spacing
    rows = []
    for factor in [1, 2, 3]:
        image = np.load(folder/f'image_f{factor}.npy').astype(np.float32)[:, ::2, ::2]
        d = len(image)
        for offset in [1, 3]:
            indices = np.arange(offset, d-offset)
            valid = (3*(indices-offset) >= 0) & (3*(indices+offset) <= extent)
            indices = indices[valid]
            before, centre, after = image[indices-offset], image[indices], image[indices+offset]
            # Exclude saturation so clipping does not manufacture curvature.
            mask = ((before > .02) & (before < .98) & (centre > .02) & (centre < .98) &
                    (after > .02) & (after < .98))
            a, b, c = before[mask], centre[mask], after[mask]
            curvature = a-2*b+c
            n = len(curvature)
            assert n > 0
            dz = spacing*factor
            lower = 3*(indices-offset)/dz
            upper = 3*(indices+offset)/dz
            # Inclusive intervals: a plane exactly at the high endpoint belongs
            # to the preceding segment for this existence test.
            same_segment = np.floor(lower+1e-8) == np.ceil(upper-1e-8)-1
            selected_segments = same_segment[:, None, None] & mask
            full_curvature = before-2*centre+after
            segment_values = full_curvature[selected_segments]
            rows.append({'case': case, 'factor': factor, 'offset_mm': 3*offset,
                         'native_z_spacing_mm': spacing, 'effective_sampling_mm': dz,
                         'valid_z_centres': len(indices), 'unsaturated_sampled_pixels': n,
                         'same_linear_segment_z_fraction': float(same_segment.mean()),
                         'neighbour_difference_rms_normalized': float(np.sqrt(((a-b)**2+(c-b)**2).mean()/2)),
                         'second_difference_rms_normalized': float(np.sqrt(np.mean(curvature**2))),
                         'second_difference_within_float16_tolerance_fraction': float((np.abs(curvature) <= 2**-10).mean()),
                         'same_segment_pixels': len(segment_values),
                         'same_segment_second_difference_max': float(np.abs(segment_values).max()) if len(segment_values) else None})
    return rows


def main():
    dest = OUT/'input_diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT/'data/metadata/split.json').read_text())['splits']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'case_ids': split['train'],
                'fitting_patients_only': True, 'factors': [1, 2, 3], 'offsets_mm': [3, 9],
                'xy_stride': 2, 'intensity_mask': 'All three normalized inputs strictly between 0.02 and 0.98',
                'geometry': 'Fraction of valid centres whose three positions lie in one inclusive linear interpolation segment',
                'interpretation_limit': 'Linear dependence of three channels does not imply that neighbours contain no predictive information beyond the centre. This is an input diagnostic, not a causal mechanism test or a model-performance endpoint.',
                'no_model_or_protocol_change': True}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with cf.ProcessPoolExecutor(4) as pool:
        rows = [r for result in pool.map(one, split['train']) for r in result]
    frame = pd.DataFrame(rows)
    assert len(frame) == 80*3*2 and set(frame.case) == set(split['train'])
    frame.to_csv(dest/'patient_input_diagnostics.csv', index=False)
    means = frame.groupby(['factor', 'offset_mm'], as_index=False).mean(numeric_only=True)
    means.to_csv(dest/'input_diagnostic_summary.csv', index=False)
    # Cohort summaries use released scanner metadata; no individual dates copied.
    manifest = [('train80', split['train']), ('development20', split['val']),
                ('original_test40', split['test']),
                ('previous_followup40', json.loads((ROOT/'data/metadata/followup_context/split.json').read_text())['test']),
                ('fresh_test20', json.loads((ROOT/'data/metadata/continuation/split.json').read_text())['test'])]
    scanner = pd.read_csv(ROOT/'data/metadata/labeled_data_meta_0000_0599.csv').set_index('amos_id')
    cohort_rows = []
    for cohort, ids in manifest:
        for case in ids:
            meta = json.loads((ROOT/'data/processed'/case/'metadata.json').read_text())
            info = scanner.loc[int(case.split('_')[1])]
            age = str(info["Patient's Age"])
            cohort_rows.append({'cohort': cohort, 'case': case,
                                'z_spacing_mm': meta['original_spacing_xyz_mm'][2],
                                'x_spacing_mm': meta['original_spacing_xyz_mm'][0],
                                'native_z_extent_mm': (meta['original_shape_xyz'][2]-1)*meta['original_spacing_xyz_mm'][2],
                                'manufacturer': str(info['Manufacturer']),
                                'model': str(info["Manufacturer's Model Name"]),
                                'site': str(info['Site']), 'sex': str(info["Patient's Sex"]),
                                'age_years': int(age[:-1]) if age.endswith('Y') and age[:-1].isdigit() else None})
    cohort = pd.DataFrame(cohort_rows)
    cohort.to_csv(dest/'cohort_acquisition_metadata.csv', index=False)
    cohort.groupby(['cohort', 'z_spacing_mm']).size().rename('patients').to_csv(dest/'cohort_spacing_counts.csv')
    cohort.groupby(['cohort', 'manufacturer', 'site']).size().rename('patients').to_csv(dest/'cohort_scanner_counts.csv')
    (dest/'complete.json').write_text(json.dumps({'fitting_diagnostic_rows': len(frame),
        'cohort_description_patients': len(cohort), 'test_performance_used': False,
        'completed_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    print(means[['factor', 'offset_mm', 'same_linear_segment_z_fraction',
                 'second_difference_rms_normalized']].to_string(index=False))


if __name__ == '__main__': main()
