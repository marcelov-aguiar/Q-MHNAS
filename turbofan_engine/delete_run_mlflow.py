from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD001"
    parent_run_id = "cc2b546774904f4e826d53e30b1ae7db"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
