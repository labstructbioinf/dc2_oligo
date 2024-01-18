import argparse

from dc2_oligo.predictor import predict_oligo_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf_results", type=str, required=True)
    parser.add_argument("--save_csv", type=str, default=None, required=False)
    args = parser.parse_args()
    predict_oligo_state(args.cf_results, args.save_csv)
