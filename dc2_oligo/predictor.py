import glob
import joblib

import pandas as pd
import numpy as np

from dc2_oligo.utils import check_files_presence, check_alphafold_model_type, get_af2_emb


def get_af2_emb(cf_results: str, model_id: int, use_pairwise: bool):
    """
    Get AF2 embeddings from ColabFold output directory.

    Parameters:
        cf_results (str): Path to the ColabFold output directory.
        model_id (int): Model ID to retrieve embeddings from.
        use_pairwise (bool): Whether to include pairwise embeddings.

    Returns:
        np.ndarray: Array containing the AF2 embeddings.
    """


    representations = sorted(glob.glob(f"{cf_results}/*_repr*_rank*_model_{model_id+1}_*"))

    single_repr_fns = sorted([x for  x in representations if "single" in x])
    pair_repr_fns = sorted([x for  x in representations if "pair" in x])

    mat = np.load(single_repr_fns[0]).mean(axis=0)

    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))

    return mat

def predict_oligo_state(cf_results:  str, save_csv: str=None):
    """
    Predict the oligomer state using a trained model and return results as a DataFrame.

    Parameters:
        cf_results (str): Path to the ColabFold output directory.
        use_pairwise (bool): Whether to include pairwise embeddings.
        save_csv (str, optional): Whether to save the prediction results as a CSV file (default: False).

    Returns:
        pd.DataFrame: DataFrame containing prediction results for different oligomer states.
    """
    df = pd.DataFrame()
    model = joblib.load('dc2_oligo/model/model_cv.p') # use model/model_cv.p for benchmarking
    check_files_presence(cf_results)
    check_alphafold_model_type(cf_results)
    results = []
    for ensemble in range(0,1):
        for i in range(0,5):
            for k in range(5):
                X = np.asarray([get_af2_emb(cf_results, model_id = i, use_pairwise=False)])
                sc = model[f'scaler_{ensemble}_{i}_{k}']
                X= sc.transform(X)
                result = model[f'clf_{ensemble}_{i}_{k}'].predict_proba(X)
                results.append(result)

    results = np.array(results)
    avg_proba = np.mean(results, axis=0)
    std_proba = np.std(results, axis=0)

    y_pred_bin = avg_proba.argmax(axis=1)

    data = {avg_proba[0][0]: std_proba[0][0], avg_proba[0][1]: std_proba[0][1], avg_proba[0][2]: std_proba[0][2]}
    data = {'prob_dimer':avg_proba[:,0],
            'prob_dimer_std':std_proba[:,0],
            'prob_trimer':avg_proba[:,1],
            'prob_trimer_std':std_proba[:,1],
            'prob_tetramer':avg_proba[:,2],
            'prob_tetramer_std':std_proba[:,2],
            'y_pred':y_pred_bin[0],}
    df = pd.DataFrame(data)
    oligo_dict = {0: "Dimer", 1: "Trimer", 2: "Tetramer"}
    print(f"Predicted oligomer state: {oligo_dict[y_pred_bin[0]]} ({y_pred_bin[0]}) with probability \
          {round(avg_proba[0][y_pred_bin[0]],5)} +/- {round(std_proba[0][y_pred_bin[0]],5)}")
    if save_csv is not None:
        if not save_csv.endswith('.csv'):
            save_csv += '.csv'
            df.to_csv(f"{save_csv}")
        else:
            df.to_csv(f"{save_csv}")

    return df
