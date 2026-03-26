"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Semelhante ao runner_evolution.py, mas chama o run_retreining.py
"""
import subprocess
import os
from util import load_yaml


if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_retraining.py")

	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		#"FD001/config_files/config_turbofan_FD001_v63.txt", # MLFlow
		#"FD003/config_files/config_turbofan_FD003_v26.txt", # MLFlow
		#"FD003/config_files/config_turbofan_FD003_v27.txt", # MLFlow
		#"FD003/config_files/config_turbofan_FD003_v28.txt" # MLFlow
		# "FD001/config_files/config_turbofan_FD001_v59.txt", #falta uma rodada
		# "FD002/config_files/config_turbofan_FD002_v28.txt",
		# "FD001/config_files/config_turbofan_FD001_v65.txt",
		# "FD001/config_files/config_turbofan_FD001_v60.txt", #falta uma rodada
		# "FD001/config_files/config_turbofan_FD001_v64.txt",
		# "FD001/config_files/config_turbofan_FD001_v59.txt",
		# "FD002/config_files/config_turbofan_FD002_v29.txt"
		#"FD004/config_files/config_turbofan_FD004_v33.txt"


		# "FD004/config_files/config_turbofan_FD004_v30.txt",
		# "FD004/config_files/config_turbofan_FD004_v32.txt",
		# "FD001/config_files/config_turbofan_FD001_v60.txt",
		# "FD001/config_files/config_turbofan_FD001_v59.txt",
		# "FD001/config_files/config_turbofan_FD001_v64.txt",
		# "FD001/config_files/config_turbofan_FD001_v30.txt"
		# "FD001/config_files/config_turbofan_FD001_v66.txt",
		"FD001/config_files/config_turbofan_FD001_v30.txt",
		"FD004/config_files/config_turbofan_FD004_v34.txt"
	]
	for cfg in config_files:
		config_path = os.path.join(base_path, cfg)
		config_data = load_yaml(config_path)
		repeat_count = config_data['train']['repeat']
		for repetition in range(1, repeat_count + 1):
			print(f"\n=== Executando retreinamento com {cfg} {repetition}/{repeat_count} ===\n")
			subprocess.run(
				["python", run_script, "--config", cfg, "--repeat", str(repetition)],
				cwd=base_path,
				check=True
       		)