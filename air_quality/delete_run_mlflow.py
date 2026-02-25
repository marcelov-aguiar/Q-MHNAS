from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "air_quality"
    parent_run_id = "4b0068b60b744746ae15ae16f0dcfd48"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
