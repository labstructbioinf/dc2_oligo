import glob

import pandas as pd
import numpy as np

from dc2_oligo.predictor import predict_oligo_state

def test_predict_oligo_state():
    test_df = pd.DataFrame()
    test_cases = glob.glob('tests/data/*')
    pdb_ids = [x.split('/')[-1][:4] for x in glob.glob('tests/data/*/*') if '_pair_repr_rank_001_' in x]
    for pdb_id, test_case in zip(pdb_ids, test_cases):
        df = predict_oligo_state(test_case)
        df['pdb'] = pdb_id
        test_df = pd.concat([test_df, df], axis=0).sort_values(by='pdb')

    assertion_df = pd.read_csv('tests/test_df.csv')
    assert np.isclose(test_df.select_dtypes(include=np.number),
                assertion_df.select_dtypes(include=np.number),
                atol=0.1).all(), 'Test case results do not match assertion results'

