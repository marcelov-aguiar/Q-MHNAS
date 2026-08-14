import os
import logging
from pathlib import Path
from util import load_yaml

from mlflow_utils import mlflow_tracking


LOG_FILE = "log_run_mlflow.log"

logging.basicConfig(
    level=logging.INFO,  # controla o nível dos logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),            # terminal
        logging.FileHandler(LOG_FILE, 'a')  # arquivo
    ]
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    for dataset in ["etth1"]:
        
        base_path = Path(__file__).resolve().parent
        target_cols = ["ot", "hufl", "hull", "lufl"]
        
        # Para o teste, estamos fixando o config da v4
        config_name = "config_etth1_v10.txt"
        
        targets_data = {}
        general_config_data = None 
        exp_name = None

        # Loop passando por cada target (ot, hufl, hull, lufl)
        for target in target_cols:
            
            # 1. Acessa o config ESPECÍFICO de cada target
            # Ex: ett/etth1/hufl/config_files/config_etth1_v4.txt
            config_path = base_path / dataset / target / "config_files" / config_name
            
            if not config_path.exists():
                logger.warning(f"⚠️ Config não encontrado para o target {target}: {config_path}")
                continue
                
            try:
                config_data = load_yaml(config_path)
                
                # Salva o exp_name e o config geral na primeira iteração (já que os hiperparâmetros da NAS são compartilhados)
                if exp_name is None:
                    exp_name = config_data['train']['exp']
                    general_config_data = config_data
                
                lr_scheduler = config_data['train']['lr_scheduler']
                num_repeats = int(config_data["train"]["repeat"])

                exp_path_base_config = config_data['train']['exp_path_base']
                
                logger.info(f"🔁 Target {target.upper()} | Config {exp_name} | Buscas: {num_repeats}")

                # 2. Carrega as buscas (repeats) para este target
                target_repeat_data = {}
                target_exp_path = base_path / dataset / target
                
                for repeat_id in range(1, num_repeats + 1):
                    # Ex: ett/etth1/hufl/exp_v4/search_1
                    repeat_dir = target_exp_path / exp_path_base_config / f"{exp_name}_repeat_{repeat_id}"
                    
                    if not repeat_dir.exists():
                        logger.warning(f"⚠️ Warning: repeat folder not found -> {repeat_dir}")
                        continue
                    
                    retrain_data = mlflow_tracking.load_retrain_jsons(repeat_dir, lr_scheduler)
                    target_repeat_data[repeat_id] = retrain_data
                    logger.info(f"✅ Target {target.upper()} - Busca {repeat_id} carregada com sucesso ({len(retrain_data)} retrains).")
                
                # Adiciona os dados deste target ao dicionário global se encontrou retreinos
                if target_repeat_data:
                    targets_data[target] = target_repeat_data

            except Exception as e:
                logger.error(f"Erro ao processar o target {target}: {e}")
                continue

        # 3. Dispara a gravação estruturada com as médias globais no MLflow
        if not targets_data:
            logger.warning("❌ Nenhum dado válido de target/repeat encontrado para logar.")
        else:
            logger.info(f"⏳ Carregando '{exp_name}' como Super-Parent Run no MLflow. Aguarde...")
            
            mlflow_tracking.log_ett_miso_super_run(
                data_set=dataset,
                exp_name=exp_name,
                targets_data=targets_data,
                base_exp_path=base_path / dataset,
                config_data=general_config_data, # Passa o config compartilhado (parâmetros da rede, intervalos da LSTM, etc)
                log_params_evolution=None 
            )
            logger.info(f"🏁 Super-Parent Run de teste registrado com sucesso no MLflow.")