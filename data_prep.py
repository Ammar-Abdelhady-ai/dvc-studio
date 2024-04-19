import os
import pandas as pd
import yaml
import dvc




out_path = os.path.join(os.getcwd(), "outs")
os.makedirs(out_path, exist_ok=True)

cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, "dataset.csv"))

def prepare_func(age_threshold: int):
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
    df.drop(index=df[df["Age"] > 80].index.to_list(), axis=0, inplace=True)

    df.to_csv(os.path.join(out_path, "prepared_df.csv"), index=False)

def main():
    with open("params.yaml") as f:
        threshold_params = yaml.safe_load(f)["prepare"]
    #threshold_params = dvc.api.params_show()["prepare"]
    threshold: int =  threshold_params["age_threshold"]
    prepare_func(threshold)


if __name__ == "__main__":
    main()