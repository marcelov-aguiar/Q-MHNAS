"""
Responsável por executar o retreinamento de todos os arquivos .txt dos diretorios passados
Chama o run_evolution.py
"""
import subprocess
import os
from util import load_yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
	base_path = os.path.dirname(os.path.abspath(__file__))
	
	run_script = os.path.join(base_path, "run_evolution.py")
	# How to execute: LD_LIBRARY_PATH= python nome_do_arquivo.py
	# config_dir = os.path.join(base_path, "config_files")
	# config_files = [f for f in os.listdir(config_dir) if f.endswith(".txt")]
	config_files = [
		#"FD001/config_files/config_turbofan_FD001_v61.txt",
		#"FD001/config_files/config_turbofan_FD001_v62.txt",
		#"FD001/config_files/config_turbofan_FD001_v63.txt",
		# "FD003/config_files/config_turbofan_FD003_v26.txt", # rodando no Dell G15
		#"FD001/config_files/config_turbofan_FD001_v65.txt", # verificar se esse exp e necessario
		#"FD001/config_files/config_turbofan_FD001_v60.txt", # repetir uma vez
		#"FD001/config_files/config_turbofan_FD001_v59.txt", # repetir uma vez
		#"FD001/config_files/config_turbofan_FD001_v64.txt", # repetir uma vez; comparar aumento de geracao
		#
		# Rodando PC Santiago: "FD003/config_files/config_turbofan_FD003_v27.txt",
		# Rodando PC Santiago: "FD003/config_files/config_turbofan_FD003_v28.txt",
		# Rodando PC Santiago: "FD002/config_files/config_turbofan_FD002_v28.txt",
		# Rodando PC Santiago: "FD002/config_files/config_turbofan_FD002_v29.txt",
		# Rodando PC Santiago: "FD004/config_files/config_turbofan_FD004_v32.txt", # Zé
		# Rodando PC Santiago: "FD004/config_files/config_turbofan_FD004_v33.txt", # Zé
		# Rodando PC Santiago:  # Zé
		"FD004/config_files/config_turbofan_FD004_v34.txt", # rodar uma rodada aqui e duas no santiago
		"FD001/config_files/config_turbofan_FD001_v66.txt", # padronizar quantidade de geracoes
		"FD001/config_files/config_turbofan_FD001_v30.txt" # falta rodar duas vezes
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