import numpy as np
import argparse
import glob
import joblib
import sys
import os


def get_af2_emb_upd(colabfold_output_dir, use_pairwise: bool):
    """Get AF2 embeddings from ColabFold output directory"""


    representations = sorted(glob.glob(f"{colabfold_output_dir}/*_repr_rank_*"))

    single_repr_fns = sorted([x for  x in representations if "single" in x])
    pair_repr_fns = sorted([x for  x in representations if "pair" in x])

    mat = np.load(single_repr_fns[0]).mean(axis=0)
    
    if use_pairwise:
        mat = np.hstack((mat, np.load(pair_repr_fns[0]).mean(axis=0).mean(axis=0)))
    
    return mat

def predict_oligo_state(colabfold_output_dir:  str, use_pairwise: bool):
    model = joblib.load('data/model.p')#['clf_0_4_3']  # !!!!!!!!!!!!!!!!!!!
    X = np.asarray([get_af2_emb_upd(colabfold_output_dir, use_pairwise=use_pairwise)]).reshape(1, -1)
    print(model)
    output = model.predict(X)
    print(output)

    oligo_dict = {0: "Dimer", 1: "Trimer", 2: "Tetramer"}

    print(f"Predicted oligomer state: {oligo_dict[output[0]]}")

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colabfold_output_dir", type=str, required=True)
    parser.add_argument("--use_pairwise", type=bool, default=True)
    args = parser.parse_args()
    predict_oligo_state(args.colabfold_output_dir, args.use_pairwise)