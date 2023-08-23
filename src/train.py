import pandas as pd
import numpy as np
import argparse
import glob
import json
import joblib
import os

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

CALC_PATHS = '/home/nfs/jludwiczak/af2_cc/af2_multimer/calc'


def get_x(id_: int, rank: int, model: str = "af2", 
          use_pairwise: bool = True):
    
    single_repr_fns = sorted(glob.glob(f"{CALC_PATHS}/{id_}/*_single_repr_rank_00*"))
    pair_repr_fns = sorted(glob.glob(f"{CALC_PATHS}{id_}/*_pair_repr_rank_00*"))
    
    mat = np.load(single_repr_fns[rank]).mean(axis=0)
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[rank]).mean(axis=0).mean(axis=0)))
    return mat

def get_af2_emb(id_: int, model_id: int, use_pairwise: bool):
    
    single_repr_fns = sorted(glob.glob(f"{CALC_PATHS}/{id_}/*_single_repr_rank_*_model_{model_id+1}_*"))
    pair_repr_fns = sorted(glob.glob(f"{CALC_PATHS}/{id_}/*_pair_repr_rank_*_model_{model_id+1}_*"))
    

    mat = np.load(single_repr_fns[0]).mean(axis=0)
    
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))
    
    return mat


def train(c=10, balanced=0, dual=1, ensemble_size=1, use_pairwise=True, use_scaler=True):
    
    # Load dataset
    df = pd.read_csv("../data/set4_homooligomers.csv", sep="\t")
    df = df.drop_duplicates(subset="full_sequence", keep="first")
    
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['chains'])\

    results = np.zeros((ensemble_size, 5, len(df), 3))
    model = {}

    for j in range(0, ensemble_size):
        for i in range(0, 5): # 5 since we have 5 AF2 models

            X = np.asarray([get_af2_emb(id_, model_id=i, use_pairwise=use_pairwise) for id_ in df.index])
            y = df['y'].values

            # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

            if use_scaler == 1:
                sc = StandardScaler()
                X= sc.fit_transform(X)
                # X_te = sc.transform(X_te)
                model[f"scaler_{j}_{i}"] = sc
            clf = LogisticRegression(C=c, max_iter=1000, solver='liblinear',
                                     dual=False if dual == 0 else True,
                                     class_weight='balanced' if balanced == 1 else None)
            clf.fit(X, y)
            results = clf.predict_proba(X)
            model[f"clf_{j}_{i}"] = clf
    probabilities = []
    for i in range(0, 5):
        clf = model[f"clf_{j}_{i}"]
        proba = clf.predict_proba(X)
        probabilities.append(proba)

    # Average the predicted probabilities
    avg_proba = np.mean(probabilities, axis=0)

    # Make the final binary prediction using a threshold
    threshold = 0.5
    y_pred_bin = (avg_proba[:, 1] >= threshold).astype(int)

    # y_pred_bin = results.argmax(axis=1)
    joblib.dump(clf, '../data/model.p')

    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')

    df["y_pred"] = y_pred_bin
    df["prob_dimer"] = results[:, 0]
    df["prob_trimer"] = results[:, 1]
    df["prob_tetramer"] = results[:, 2]
    print(results_)
    df.to_csv('../data/results/results.csv')

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


    os.makedirs("results", exist_ok=True)
    joblib.dump(model, "results/model.p")
    df.to_csv(f"results/results.csv")