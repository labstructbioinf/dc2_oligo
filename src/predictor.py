import glob
import joblib
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_af2_emb_upd(cf_results: str, model_id: int, use_pairwise: bool):
    """Get AF2 embeddings from ColabFold output directory"""


    representations = sorted(glob.glob(f"{cf_results}/*_repr*_rank*_model_{model_id+1}_*"))

    single_repr_fns = sorted([x for  x in representations if "single" in x])
    pair_repr_fns = sorted([x for  x in representations if "pair" in x])

    mat = np.load(single_repr_fns[0]).mean(axis=0)
    
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))
    
    return mat

def predict_oligo_state(cf_results:  str, use_pairwise: bool, save_csv: bool = False):
    df = pd.DataFrame()
    model = joblib.load('model/model.p')
    results = []
    for i in range(0,5):
        X = np.asarray([get_af2_emb_upd(cf_results, model_id = i, use_pairwise=use_pairwise)])
        sc = model[f'scaler_0_{i}']
        X= sc.transform(X)
        result = model[f'clf_0_{i}'].predict_proba(X)
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
    if save_csv:
        dir_name = os.path.basename(cf_results)
        df.to_csv(f"{cf_results}/{dir_name}_dc2_oligo.csv")
    return df
