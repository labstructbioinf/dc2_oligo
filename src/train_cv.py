import pandas as pd
import numpy as np
import argparse
import glob
import joblib
import os
import wandb

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

CALC_PATHS = '../calc'

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

    df = pd.read_csv("../tests/set5_homooligomers.csv", sep="\t")
    # df = df.drop_duplicates(subset="full_sequence", keep="first")
    
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['chains'])\

    results = np.zeros((ensemble_size, 5, len(df), 3))
    model = {}
    probabilities = []
    for j in range(0, ensemble_size):
        for i in range(0, 5): # 5 since we have 5 AF2 models

            X = np.asarray([get_af2_emb(id_, model_id=i, use_pairwise=use_pairwise) for id_ in df.index])
            y = df['y'].values

            cv = KFold(n_splits=5, shuffle=True)

            for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):

                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                if use_scaler == 1:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)
                    model[f"scaler_{j}_{i}_{k}"] = sc
                clf = LogisticRegression(C=c, max_iter=1000, solver='liblinear',
                                        dual=False if dual == 0 else True,
                                        class_weight='balanced' if balanced == 1 else None)
                clf.fit(X_tr, y_tr)
                results[j, i, te_idx, :] = clf.predict_proba(X_te)
                model[f"clf_{j}_{i}_{k}"] = clf


    # for i in range(0, 5):
    #     clf = model[f"clf_0_{i}"]
    #     proba = clf.predict_proba(X)
    #     probabilities.append(proba)
    
    y_pred_bin = results.mean(axis=0).mean(axis=0).argmax(axis=1)
    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')
    # wandb.log({
    #     'f1': results_["f1"], 
    #     'accuracy': results_["accuracy"]})
    df["y_pred"] = y_pred_bin
    df["prob_dimer"] = results.mean(axis=0).mean(axis=0)[:, 0]
    df["prob_trimer"] = results.mean(axis=0).mean(axis=0)[:, 1]
    df["prob_tetramer"] = results.mean(axis=0).mean(axis=0)[:, 2]
    joblib.dump(model, f"../model/model_cv.p")
    df.to_csv('../model/results_cv.csv')
    print(results_)

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
    # run = wandb.init()

    results, model, df = train(args.C, args.balanced, args.dual, args.ensemble_size, args.use_pairwise, args.use_scaler)
