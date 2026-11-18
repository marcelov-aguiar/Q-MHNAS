from mlflow_utils import mlflow_tracking


if __name__ == "__main__":
    experiment_name = "etth1_MISO_NAS"
    parent_run_id = "df6be48c5fc9415f82add37fbc1b2348"
    mlruns_path = "./mlruns"
    mlflow_tracking.delete_run_physical_and_mlflow(experiment_name, parent_run_id, mlruns_path)
