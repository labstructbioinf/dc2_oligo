import argparse

from src.predictor import predict_oligo_state

def predict(colabfold_output_dir: str, use_pairwise: bool):
    """Predict oligomer state from ColabFold output directory"""
    return predict_oligo_state(colabfold_output_dir, use_pairwise)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--colabfold_output_dir", type=str, required=True)
    parser.add_argument("--use_pairwise", type=bool, default=True)
    args = parser.parse_args()
    predict_oligo_state(args.colabfold_output_dir, args.use_pairwise)