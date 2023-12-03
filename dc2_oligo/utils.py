from typing import Union
import os
import glob

def check_files_presence(cf_results: str) -> None:
    """
    Check if there are embeddings in the ColabFold output directory.

    Parameters:
        cf_results (str): Path to the ColabFold output directory.
    
    Returns:
        None
    """
    _single_repr_fns = sorted(glob.glob(f"{cf_results}/*_single_repr_rank_*_model_*"))
    _pair_repr_fns = sorted(glob.glob(f"{cf_results}/*_pair_repr_rank_*_model_*"))

    if len(_single_repr_fns) < 5 or len(_pair_repr_fns) < 5:
        raise FileNotFoundError("Not embeddings found in ColabFold output directory (5 paired and 5 single embeddings are required)")



def check_alphafold_model_type(cf_results: str) -> Union[bool, ValueError, str]:
    """
    Check whether the ColabFold output directory contains json file and if it exists, check multimer version

    Parameters:
        cf_results (str): Path to the ColabFold output directory.

    Returns:
        Union: True if AlphaFold2 in proper model was used, otherwise ValueError or str with warning
    """

    _json_exists = os.path.exists(os.path.join(cf_results, "config.json"))

    if _json_exists:
        with open(os.path.join(cf_results, "config.json"), "r") as f:
            _json = f.read()
        if "alphafold2_multimer_v3" in _json:
            return True
        else:
            raise ValueError("AlphaFold2 models are present but not multimer version 3 - please provide embeddings from alphafold2_multimer_v3 model")
    else:
        return print("***!WARNING!*** No information about AlphaFold model type - make sure they have been predicted with alphafold2_multimer_v3, otherwise results may be incorrect")    

