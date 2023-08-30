import argparse

from src.predictor import predict_oligo_state_and_topology

def predict(cf_results: str, use_pairwise: bool):
    """Predict oligomer state from ColabFold output directory"""
    return predict_oligo_state_and_topology(cf_results, use_pairwise)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf_results", type=str, required=True)
    parser.add_argument("--save_csv", type=str, default=None, required=False)
    parser.add_argument("--predict_topology", action="store_true", default=False, required=False)
    args = parser.parse_args()
    predict_oligo_state_and_topology(
        cf_results=args.cf_results, 
        save_csv=args.save_csv,
        predict_topology=args.predict_topology)