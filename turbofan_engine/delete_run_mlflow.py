from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD003"
    parent_run_id = "6a4bd5de9eb74b14b19733a920e13893"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
