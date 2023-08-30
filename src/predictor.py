import glob
import joblib
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


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

def predict_oligo_state_and_topology(cf_results:  str, use_pairwise=True, save_csv: str='', predict_topology: bool=False):
    """
    Predict the oligomer state using a trained model and return results as a DataFrame.
    
    Parameters:
        cf_results (str): Path to the ColabFold output directory.
        use_pairwise (bool): Whether to include pairwise embeddings.
        save_csv (bool, optional): Whether to save the prediction results as a CSV file (default: False).
        predict_topology (bool, optional): Whether to predict the topology (default: False).
        
    Returns:
        pd.DataFrame: DataFrame containing prediction results for different oligomer states.
    """
    df = pd.DataFrame()
    model = joblib.load('model/model.p')
    # print(model)
    results_oligo = []
    results_parallel = []
    for i in range(0,5):
        X = np.asarray([get_af2_emb(cf_results, model_id = i, use_pairwise=use_pairwise)])
        sc = model[f'scaler_0_{i}']
        X= sc.transform(X)
        result_parallel = model[f'clf_0_{i}'].predict_proba(X)[0]
        results_parallel.append(result_parallel)
        result_oligo = model[f'clf_0_{i}'].predict_proba(X)[1]
        results_oligo.append(result_oligo)
    results_parallel = np.array(results_parallel)
    avg_proba_parallel = np.mean(results_parallel, axis=0)
    std_proba_parallel = np.std(results_parallel, axis=0)
    y_pred_bin_parallel = avg_proba_parallel.argmax(axis=1)
    results_oligo = np.array(results_oligo)
    avg_proba_oligo = np.mean(results_oligo, axis=0)
    std_proba_oligo = np.std(results_oligo, axis=0)
    y_pred_bin_oligo = avg_proba_oligo.argmax(axis=1)

    # data = {avg_proba[0][0]: std_proba[0][0], avg_proba[0][1]: std_proba[0][1], avg_proba[0][2]: std_proba[0][2]}
    if predict_topology:
        data = {'prob_parallel':avg_proba_parallel[:,1],
                'prob_parallel_std':std_proba_parallel[:,1],
                'prob_antiparallel':avg_proba_parallel[:,0],
                'prob_antiparallel_std':std_proba_parallel[:,0],
                'y_pred_parallel':y_pred_bin_parallel[0],
                'prob_dimer':avg_proba_oligo[:,0],
                'prob_dimer_std':std_proba_oligo[:,0],
                'prob_trimer':avg_proba_oligo[:,1],
                'prob_trimer_std':std_proba_oligo[:,1],
                'prob_tetramer':avg_proba_oligo[:,2],
                'prob_tetramer_std':std_proba_oligo[:,2],
                'y_pred_oligo':y_pred_bin_oligo[0],}
        df = pd.DataFrame(data)
        oligo_dict = {0: "Dimer", 1: "Trimer", 2: "Tetramer"}
        parallel_dict = {0: "Antiparallel", 1: "Parallel"}

        print(f"Predicted oligomer state: {oligo_dict[y_pred_bin_oligo[0]]} ({y_pred_bin_oligo[0]}) with probability \
            {round(avg_proba_oligo[0][y_pred_bin_oligo[0]],5)} +/- {round(std_proba_oligo[0][y_pred_bin_oligo[0]],5)}\
            \nPredicted topology: {parallel_dict[y_pred_bin_parallel[0]]} ({y_pred_bin_parallel[0]}) with probability \
                {round(avg_proba_parallel[0][y_pred_bin_parallel[0]],5)} +/- {round(std_proba_parallel[0][y_pred_bin_parallel[0]],5)}")
        
        if save_csv != '':
            if not save_csv.endswith('.csv'):
                save_csv += '.csv'
                df.to_csv(f"{cf_results}/{save_csv}")
            else:
                df.to_csv(f"{cf_results}/{save_csv}")

        return df
    else:
        data = {'prob_dimer':avg_proba_oligo[:,0],
                'prob_dimer_std':std_proba_oligo[:,0],
                'prob_trimer':avg_proba_oligo[:,1],
                'prob_trimer_std':std_proba_oligo[:,1],
                'prob_tetramer':avg_proba_oligo[:,2],
                'prob_tetramer_std':std_proba_oligo[:,2],
                'y_pred_oligo':y_pred_bin_oligo[0],}
        df = pd.DataFrame(data)
        oligo_dict = {0: "Dimer", 1: "Trimer", 2: "Tetramer"}

        print(f"Predicted oligomer state: {oligo_dict[y_pred_bin_oligo[0]]} ({y_pred_bin_oligo[0]}) with probability \
            {round(avg_proba_oligo[0][y_pred_bin_oligo[0]],5)} +/- {round(std_proba_oligo[0][y_pred_bin_oligo[0]],5)}")

        if save_csv:
            if not save_csv.endswith('.csv'):
                save_csv += '.csv'
                df.to_csv(f"{cf_results}/{save_csv}")
            else:
                df.to_csv(f"{cf_results}/{save_csv}")

        return df
