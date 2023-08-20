import pandas as pd
import numpy as np
import argparse
import glob
import wandb
import json
import joblib
import os

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_x(id_: int, rank: int, model: str = "af2", 
          use_pairwise: bool = True):
    
    single_repr_fns = sorted(glob.glob(f"./../calc/{id_}/*_single_repr_rank_00*"))
    pair_repr_fns = sorted(glob.glob(f"./../calc/{id_}/*_pair_repr_rank_00*"))
    
    mat = np.load(single_repr_fns[rank]).mean(axis=0)
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[rank]).mean(axis=0).mean(axis=0)))
    return mat

def get_af2_emb(id_: int, model_id: int, use_pairwise: bool):
    
    single_repr_fns = sorted(glob.glob(f"./../calc/{id_}/*_single_repr_rank_*_model_{model_id+1}_*"))
    pair_repr_fns = sorted(glob.glob(f"./../calc/{id_}/*_pair_repr_rank_*_model_{model_id+1}_*"))
    
    mat = np.load(single_repr_fns[0]).mean(axis=0)
    
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))
    
    return mat


def train(args):
    
    # Load dataset
    df = pd.read_csv("data/set4_homooligomers.csv", sep="\t")
    df = df.drop_duplicates(subset="full_sequence", keep="first")
    
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['chains'])\

    results = np.zeros((args.ensemble_size, 5, len(df), 3))
    model = {}

    for j in range(0, args.ensemble_size):
        for i in range(0, 5): # 5 since we have 5 AF2 models

            X = np.asarray([get_af2_emb(id_, model_id=i, use_pairwise=args.use_pairwise) for id_ in df.index])
            y = df['y'].values

            cv = KFold(n_splits=5, shuffle=True)

            for k, (tr_idx, te_idx) in enumerate(cv.split(X, y)):

                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                if args.use_scaler == 1:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)
                    model[f"scaler_{j}_{i}_{k}"] = sc
                clf = LogisticRegression(C=args.C, max_iter=1000, solver='liblinear',
                                         dual = False if args.dual == 0 else True, 
                                         class_weight = 'balanced' if args.balanced == 1 else None) 
                clf.fit(X_tr, y_tr)
                results[j, i, te_idx, :] = clf.predict_proba(X_te)
                model[f"clf_{j}_{i}_{k}"] = clf


    y_pred_bin = results.mean(axis=0).mean(axis=0).argmax(axis=1)
    results_ = {}
    results_["accuracy"] = accuracy_score(y, y_pred_bin)
    results_["f1"] = f1_score(y, y_pred_bin, average='macro')

    df["y_pred"] = y_pred_bin
    df["prob_dimer"] = results.mean(axis=0).mean(axis=0)[:, 0]
    df["prob_trimer"] = results.mean(axis=0).mean(axis=0)[:, 1]
    df["prob_tetramer"] = results.mean(axis=0).mean(axis=0)[:, 2]

    return results_, model, df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--C', type=float, default=1)
    parser.add_argument('--dual', type=int, default=1)
    parser.add_argument('--balanced', type=int, default=1)
    parser.add_argument('--ensemble_size', type=int, default=1)
    parser.add_argument('--use_scaler', type=int, default=1)
    parser.add_argument('--use_pairwise', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1) # WANDB dummy var
    args = parser.parse_args()
    
    run = wandb.init()
    results, model, df = train(args)

    os.makedirs(f"runs/{run.id}", exist_ok=True)
    with open(f"runs/{run.id}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    joblib.dump(model, f"runs/{run.id}/model.p")
    df.to_csv(f"runs/{run.id}/results.csv")


    wandb.log({
        'f1': results["f1"], 
        'accuracy': results["accuracy"]})
