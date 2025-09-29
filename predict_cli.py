import sys
import pandas as pd
import joblib

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_cli.py <model_name> <new_data.csv>")
        print("Available models: xgb, catboost, lgbm")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    data_path = sys.argv[2]


    if model_name == "xgb":
        model = joblib.load("final_model_xgb.pkl")
    elif model_name == "catboost":
        model = joblib.load("final_model_catboost.pkl")
    elif model_name == "lgbm":
        model = joblib.load("final_model_lgbm.pkl")
    else:
        print("Invalid model name. Choose from: xgb, catboost, lgbm")
        sys.exit(1)

    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)


    predictions = model.predict(data)


    output_file = f"predictions_{model_name}.csv"
    pd.DataFrame(predictions, columns=["Predicted"]).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
