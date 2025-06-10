import argparse
from src.train import train_model
# from src.predict import make_predictions # Uncomment if you add a predict.py

def main():
    parser = argparse.ArgumentParser(description="Kaggle Home Price Prediction Pipeline.")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train",
                        help="Choose operation mode: 'train' for model training, 'predict' for generating predictions.")
    parser.add_argument("--input_dir", type=str, default="data/raw/",
                        help="Path to the input data directory).")
    # Add other arguments for hyperparameters, model paths, etc., as needed.

    args = parser.parse_args()

    if args.mode == "train":
        print(f"--- Running in training mode with input directory: {args.input_dir} ---")
        train_model(args.input_dir)
        print("--- Training completed successfully ---")
    # elif args.mode == "predict":
    #     print(f"--- Running in prediction mode with input directory: {args.input_dir} ---")
    #     # make_predictions(args.input_dir) # Call your prediction function
    #     print("--- Prediction completed successfully ---")
    else:
        print(f"Error: Invalid mode '{args.mode}'. Please choose 'train' or 'predict'.")

if __name__ == "__main__":
    main()