import pandas as pd
import numpy as np
import argparse
import glob
import joblib
# import os

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier

CALC_PATHS = '/home/nfs/jludwiczak/af2_cc/af2_multimer/calc'

def get_af2_emb(id_: int, model_id: int, use_pairwise: bool):
    """
    Get AF2 embeddings from ColabFold output directory.

    Parameters:
        cf_results (str): Path to the ColabFold output directory.
        model_id (int): Model ID to retrieve embeddings from.
        use_pairwise (bool): Whether to include pairwise embeddings.

    Returns:
        np.ndarray: Array containing the AF2 embeddings.
    """
    
    single_repr_fns = sorted(glob.glob(f"{CALC_PATHS}/{id_}/*_single_repr_rank_*_model_{model_id+1}_*"))
    pair_repr_fns = sorted(glob.glob(f"{CALC_PATHS}/{id_}/*_pair_repr_rank_*_model_{model_id+1}_*"))
    

    mat = np.load(single_repr_fns[0]).mean(axis=0)
    
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))
    
    return mat


def train(c=10, balanced=0, dual=1, ensemble_size=1, use_pairwise=True, use_scaler=True):
    """
    Train an ensemble of logistic regression models.

    Parameters:
        c (int, optional): Regularization parameter. Defaults to 10.
        balanced (int, optional): Whether to balance class weights. Defaults to 0.
        dual (int, optional): Whether to use dual formulation. Defaults to 1.
        ensemble_size (int, optional): Number of ensemble models. Defaults to 1.
        use_pairwise (bool, optional): Whether to use pairwise embeddings. Defaults to True.
        use_scaler (bool, optional): Whether to use data scaling. Defaults to True.

    Returns:
        dict: Dictionary containing training results, trained models, and DataFrame with results.
    """

    df = pd.read_csv("../tests/set4_homooligomers.csv", sep="\t")
    df = df.drop_duplicates(subset="full_sequence", keep="first")
    
    le = LabelEncoder()
    df['oligo_state'] = le.fit_transform(df['chains'])
    le.fit(df['parallel'])
    df['parallel'] = le.transform(df['parallel'])

    results_parallel = np.zeros((ensemble_size, 5, len(df), 2))
    results_oligo = np.zeros((ensemble_size, 5, len(df), 3))
    scaler_cache = {}
    model = {}
    probabilities_parallel = []
    probabilities_oligo = []
    for j in range(0, ensemble_size):
        for i in range(0, 5): # 5 since we have 5 AF2 models
            X = np.asarray([get_af2_emb(id_, model_id=i, use_pairwise=use_pairwise) for id_ in df.index])
            y_state = df['oligo_state'].values
            y_parallel = df['parallel'].values
            y = np.column_stack((y_parallel, y_state))

            if use_scaler == 1:
                sc = StandardScaler()
                X= sc.fit_transform(X)
                model[f"scaler_{j}_{i}"] = sc
                scaler_cache[f"scaler_{j}_{i}"] = sc
            clf = MultiOutputClassifier(LogisticRegression(C=c, max_iter=2000, solver='liblinear',
                                        dual=False if dual == 0 else True,
                                        class_weight='balanced' if balanced == 1 else None))
            clf.fit(X, y)
            model[f"clf_{j}_{i}"] = clf


    for i in range(0, 5):
        clf = model[f"clf_0_{i}"]
        proba_parallel, proba_chains = clf.predict_proba(X)
        probabilities_parallel.append(proba_parallel)
        probabilities_oligo.append(proba_chains)
    
    probabilities_parallel = np.array(probabilities_parallel)
    probabilities_oligo = np.array(probabilities_oligo)

    avg_proba_parallel = np.mean(probabilities_parallel, axis=0)
    avg_proba_oligo = np.mean(probabilities_oligo, axis=0)

    y_pred_parallel_bin = avg_proba_parallel.argmax(axis=1)
    y_pred_oligo_bin = avg_proba_oligo.argmax(axis=1)
    joblib.dump(model, '../model/model.p')

    results_ = {}
    results_["accuracy_parallel"] = accuracy_score(y_parallel, y_pred_parallel_bin)
    results_["f1_parallel"] = f1_score(y_parallel, y_pred_parallel_bin, average='macro')
    results_["accuracy_oligo"] = accuracy_score(y_state, y_pred_oligo_bin)
    results_["f1_oligo"] = f1_score(y_state, y_pred_oligo_bin, average='macro')

    df["oligo_pred"] = y_pred_oligo_bin
    df["prob_dimer"] = avg_proba_oligo[:,0]
    df["prob_trimer"] = avg_proba_oligo[:,1]
    df["prob_tetramer"] = avg_proba_oligo[:, 2]
    df["parallel_pred"] = y_pred_parallel_bin
    df["prob_parallel"] = avg_proba_parallel[:,0]
    df["prob_antiparallel"] = avg_proba_parallel[:,1]
    df.to_csv('enhanced_results.csv')

    return results_, model, df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--C', type=float, default=1)
    parser.add_argument('--dual', type=int, default=1)
    parser.add_argument('--balanced', type=int, default=1)
    parser.add_argument('--ensemble_size', type=int, default=1)
    parser.add_argument('--use_scaler', type=int, default=1)
    parser.add_argument('--use_pairwise', type=int, default=1)
    args = parser.parse_args()
    
    results, model, df = train(args.C, args.balanced, args.dual, args.ensemble_size, args.use_pairwise, args.use_scaler)
