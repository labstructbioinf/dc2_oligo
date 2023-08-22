import sys
import os
import glob
root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(root_directory)


import pytest

from src.predict import predict_oligo_state

# @pytest.mark.parametrize("use_pairwise", [True, False])
def test_predict_oligo_state():
    result_1 = predict_oligo_state("tests/data/7/", use_pairwise=True)
    result_2= predict_oligo_state("tests/data/10/", use_pairwise=True)
    # Add your assertions here to verify that the result is as expected
    assert result_1 == 0
    assert result_2 == 1

def test_on_paper_results():
    path_to_colabfold_train_data = glob.glob('/home/nfs/jludwiczak/af2_cc/af2_multimer/calc/')
    print(path_to_colabfold_train_data)



