from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "etth1"
    parent_run_id = "3cf82dcdcbbb4d32877f44f5693f2b03"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
