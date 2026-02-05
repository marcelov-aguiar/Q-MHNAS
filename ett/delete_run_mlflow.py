from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "etth1"
    parent_run_id = "2620a6883e7b457ca8ee1821c87dc25e"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
