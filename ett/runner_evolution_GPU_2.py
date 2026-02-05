"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Chama o run_evolution.py
"""
import subprocess
import os
from util import load_yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_evolution_dual_2.py")
	# How to execute: LD_LIBRARY_PATH= python nome_do_arquivo.py
	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		"etth1/config_files/config_etth1_v1.txt",
		"etth1/config_files/config_etth1_v2.txt",
		"etth1/config_files/config_etth1_v3.txt"
	]
	for cfg in config_files:
		config_path = os.path.join(base_path, cfg)
		config_data = load_yaml(config_path)
		repeat_count = config_data['train']['repeat']
		for repetition in range(1, repeat_count + 1):
			print(f"\n=== Executando experimento com {cfg} {repetition}/{repeat_count} ===\n")
			subprocess.run(
				["python", run_script, "--config", cfg, "--repeat", str(repetition)],
				cwd=base_path,
				check=True
        	)