import sys
import os
import glob
import pandas as pd
import numpy as np

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(root_directory)


import pytest

from src.predictor import predict_oligo_state

def test_predict_oligo_state():
    test_df = pd.DataFrame()
    test_cases = glob.glob('tests/data/*')
    for i, test_case in enumerate(test_cases):
        df = predict_oligo_state(test_case, use_pairwise=True)
        df['test_case_index'] = i
        df = df.set_index('test_case_index')
        test_df = pd.concat([test_df, df], axis=0)

    assertion_df = pd.read_csv('tests/test_df.csv').set_index('test_case_index')
    
    assert np.isclose(test_df, assertion_df).all(), 'Test case results do not match assertion results'

