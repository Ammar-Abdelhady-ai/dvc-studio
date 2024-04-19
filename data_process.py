from sklearn_features.transformers import DataFrameSelector
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import yaml, dvc, os
 





out_path = os.path.join(os.getcwd(), "outs")
os.makedirs(out_path, exist_ok=True)


processed_file_path = os.path.join(out_path, "prepared_df.csv")
df = pd.read_csv(processed_file_path) 



X = df.drop("Exited", axis=1)
y = df["Exited"]

def process_func(test_size: float, seed: int):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=seed, 
                                                        shuffle=True, stratify=y)

    num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    cat_cols = ['Geography', 'Gender']

    ready_cols = list(set(X_train.columns.to_list()) - set(cat_cols) - set(num_cols))



    # Pipeline

    num_pipeline = Pipeline(steps=[
            ("selector", DataFrameSelector(num_cols)), 
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()), 
        ], verbose=True
    )

    cat_pipeline = Pipeline(steps=[
            ("selector", DataFrameSelector(cat_cols)), 
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoded", OneHotEncoder(drop="first", sparse_output=False)),
        ], verbose=True
    )

    ready_pipeline = Pipeline(steps=[
            ("selector", DataFrameSelector(ready_cols)),  
            ("imputer", KNNImputer(n_neighbors=5)),
        ], verbose=True
    )

    # Concatenates results of multiple transformer objects.
    all_pipeline = FeatureUnion(transformer_list=[
            ("numerical", num_pipeline), 
            ("categorical", cat_pipeline),
            ("ready", ready_pipeline),
        ], n_jobs=-1, verbose=True
    )


    X_train_final = all_pipeline.fit_transform(X_train)
    X_test_final = all_pipeline.transform(X_test)

    out_careg_cols = all_pipeline.named_transformers["categorical"].named_steps["encoded"].get_feature_names_out(cat_cols)

    X_train_final = pd.DataFrame(X_train_final, columns=num_cols + list(out_careg_cols) + ready_cols)
    X_test_final = pd.DataFrame(X_test_final, columns=num_cols + list(out_careg_cols) + ready_cols)


    # Train
    X_train_final.to_csv(os.path.join(out_path, "process_train_x.csv"), index=False)
    y_train.to_csv(os.path.join(out_path, "process_train_y.csv"), index=False)

    # Teast
    X_test_final.to_csv(os.path.join(out_path, "process_test_x.csv"), index=False)
    y_test.to_csv(os.path.join(out_path, "process_test_y.csv"), index=False)



def main():
    with open("params.yaml") as f:
        process_params = yaml.safe_load(f)["process"]
    #threshold_params = dvc.api.params_show()["prepare"]
    test_size: float =  process_params["test_size"]
    seed: int =  process_params["seed"]
    process_func(test_size, seed)


if __name__ == "__main__":
    main()
