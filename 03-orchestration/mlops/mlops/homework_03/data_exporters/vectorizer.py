import pickle

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    with mlflow.start_run() as run:
        train = data[['PULocationID', 'DOLocationID']].to_dict(orient='records')

        dv = DictVectorizer()
        X_train = dv.fit_transform(train)
        y_train = data['duration']

        model = LinearRegression().fit(X_train, y_train)

        print(round(model.intercept_, 2))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
        )

        vectorizer_path = "dict_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(vectorizer_path, "dict_vectorizer")

        run_id = run.info.run_id

    client = MlflowClient()
    model_name = "LinearRegressiontWithDictVectorizer"
    model_uri = f"runs:/{run_id}/sklearn-model"
    vectorizer_uri = f"runs:/{run_id}/dict_vectorizer/dict_vectorizer.pkl"

    client.create_registered_model(model_name)
    model_version = client.create_model_version(model_name, model_uri, run_id)
    vectorizer_version = client.create_model_version(model_name, vectorizer_uri, run_id)

    return dv, model