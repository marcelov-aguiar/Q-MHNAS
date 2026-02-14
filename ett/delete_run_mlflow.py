from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "etth1"
    parent_run_id = "73093f901379435c97e9167ce6978c22"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
