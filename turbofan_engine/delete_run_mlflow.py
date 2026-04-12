from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "FD002"
    parent_run_id = "e2c6deffd5644dc493ed460ad1bc6383"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
