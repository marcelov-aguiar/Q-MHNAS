from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD004"
    parent_run_id = "cd6843c5037648feb0aab80a1e941405"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
