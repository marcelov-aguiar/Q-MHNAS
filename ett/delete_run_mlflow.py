from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "etth1"
    parent_run_id = "4aeb5a55880f44a3b65ff738497c26ce"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
