import glob

import pandas as pd
import numpy as np

from dc2_oligo.predictor import predict_oligo_state

pdb_ids = ['3e7k', '6us8', '5kht', '5c9n', '3a2a', '3bj4', '4w80', '3w8v', '5k9l', '5k92','6osd']
test_cases = [f'tests/{i}'  for i in range(11)]
def test_predict_oligo_state():
    test_df = pd.DataFrame()

    for pdb_id, test_case in zip(pdb_ids, test_cases):
        print(pdb_id, test_case)
        df = predict_oligo_state(test_case)
        df['pdb'] = pdb_id
        test_df = pd.concat([test_df, df], axis=0).sort_values(by='pdb')
    test_df.to_csv('tests/test_df.csv', index=False)
    assertion_df = pd.read_csv('tests/test_df.csv')
    assert np.isclose(test_df.select_dtypes(include=np.number),
                    assertion_df.select_dtypes(include=np.number),
                    atol=0.1).all(), 'Test case results do not match assertion results'
