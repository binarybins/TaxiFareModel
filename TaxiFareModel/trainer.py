# imports
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import TimeFeaturesEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

class Trainer():
    MLFLOW_URI = "https://mlflow.lewagon.ai/"
    experiment_name = "[SG] [SG] [BM] [test_experiment] [220813v04]"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.rsme = None
        self.X = X
        self.y = y

    def set_pipeline(self, model_name, model):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (model_name, model)
        ])
        return pipe

    def run(self, model_name, model):
        """set and train the pipeline"""
        self.pipeline = Trainer.set_pipeline(self, model_name, model)
        self.pipeline = self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(y_pred, y_test)
        print(self.rmse)
        return self.rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self,model_name, model):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')
        self.mlflow_run
        self.mlflow_log_metric("rmse", self.rmse)
        self.mlflow_log_param(model_name, model)
        pass

if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate

    N = 10_000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)

    dic = {"linear":LinearRegression(),
           "SGD":SGDRegressor(),
           "KNN":KNeighborsRegressor(),
           "SVR":SVR()}
    for k,v in dic.items():
        trainer.run(k, v)
        trainer.evaluate(X_test, y_test)
        trainer.save_model(k, v)
