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
		# "etth1/hufl/config_files/config_etth1_v15.txt",
		# "etth1/hull/config_files/config_etth1_v13.txt"
		# "etth1/hull/config_files/config_etth1_v14.txt",
		# "etth1/hull/config_files/config_etth1_v15.txt",
		# "etth1/lufl/config_files/config_etth1_v11.txt",
		# "etth1/lufl/config_files/config_etth1_v12.txt",
		# "etth1/lufl/config_files/config_etth1_v13.txt",
		# "etth1/lufl/config_files/config_etth1_v14.txt",
		# "etth1/lufl/config_files/config_etth1_v15.txt",

		# "etth1/lull/config_files/config_etth1_v4.txt",
		# "etth1/lull/config_files/config_etth1_v5.txt",
		# "etth1/lull/config_files/config_etth1_v6.txt",
		# "etth1/lull/config_files/config_etth1_v7.txt",
		# "etth1/lull/config_files/config_etth1_v8.txt",
		# "etth1/lull/config_files/config_etth1_v9.txt",
		# "etth1/lull/config_files/config_etth1_v10.txt",
		# "etth1/lull/config_files/config_etth1_v11.txt",
		# "etth1/lull/config_files/config_etth1_v12.txt",
		# "etth1/lull/config_files/config_etth1_v13.txt",
		# "etth1/lull/config_files/config_etth1_v14.txt",
		# "etth1/lull/config_files/config_etth1_v15.txt",

		# "etth1/mufl/config_files/config_etth1_v4.txt",
		# "etth1/mufl/config_files/config_etth1_v5.txt",
		# "etth1/mufl/config_files/config_etth1_v6.txt",
		# "etth1/mufl/config_files/config_etth1_v7.txt",
		# "etth1/mufl/config_files/config_etth1_v8.txt",
		# "etth1/mufl/config_files/config_etth1_v9.txt",
		# "etth1/mufl/config_files/config_etth1_v10.txt",
		# "etth1/mufl/config_files/config_etth1_v11.txt",
		# "etth1/mufl/config_files/config_etth1_v12.txt",
		# "etth1/mufl/config_files/config_etth1_v13.txt",
		# "etth1/mufl/config_files/config_etth1_v14.txt",
		
		# "etth1/mufl/config_files/config_etth1_v15.txt", # está na 156_1 (geracao 78) e DELL2_152 (geracao 3)

		# "etth1/mull/config_files/config_etth1_v4.txt",
		# "etth1/mull/config_files/config_etth1_v5.txt",
		# "etth1/mull/config_files/config_etth1_v6.txt",
		# "etth1/mull/config_files/config_etth1_v7.txt",
		# "etth1/mull/config_files/config_etth1_v8.txt",
		# "etth1/mull/config_files/config_etth1_v9.txt",
		# "etth1/mull/config_files/config_etth1_v10.txt",
		# "etth1/mull/config_files/config_etth1_v11.txt",
		# "etth1/mull/config_files/config_etth1_v12.txt",
		# "etth1/mull/config_files/config_etth1_v13.txt",
		# "etth1/mull/config_files/config_etth1_v14.txt",

		# "etth1/mull/config_files/config_etth1_v15.txt", # está na DELL1_152 (geracao 88)
		
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