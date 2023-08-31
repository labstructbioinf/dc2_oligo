import argparse

from src.predictor import predict_oligo_state

def predict(cf_results: str, use_pairwise: bool):
    """Predict oligomer state from ColabFold output directory"""
    return predict_oligo_state(cf_results, use_pairwise)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf_results", type=str, required=True)
    parser.add_argument("--save_csv", type=str, default=None, required=False)
    args = parser.parse_args()
    predict_oligo_state(args.cf_results, args.use_pairwise, args.save_csv)