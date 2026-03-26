from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD001"
    parent_run_id = "279b90f928364112bb5714ae878ceb4b"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
