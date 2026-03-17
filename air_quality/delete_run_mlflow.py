from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "air_quality"
    parent_run_id = "4fabd5a4f4354d9dbb18ed4bb7091799"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
