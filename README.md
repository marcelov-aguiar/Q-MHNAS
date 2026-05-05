# Q-MHNAS: Busca de Arquitetura Neural Multi-Head com Inspiração Quântica

Este repositório contém o código-fonte desenvolvido para a dissertação de mestrado intitulada **"Q-MHNAS: Busca de arquitetura neural Multi-Head com inspiração quântica"**, defendida no Departamento de Engenharia Elétrica da PUC-Rio.

O **Q-MHNAS** (Quantum-inspired Multi-Head Neural Architecture Search) é uma extensão do algoritmo Q-NAS projetada especificamente para tarefas de regressão em séries temporais multivariadas. O método automatiza a definição da estrutura da rede dividindo o problema em dois níveis: uma macro-arquitetura fixa (Multi-Head CNN-LSTM) e uma microarquitetura flexível otimizada por meio de algoritmos evolucionários com inspiração quântica (QIEA). Essa abordagem introduz estratégias de arquitetura convolucional compartilhada (SCH) e independente (ICH), permitindo que a rede evolua extratores de características heterogêneos e especializados para cada variável dos dados.

## 📊 Estudos de Caso

A metodologia proposta foi amplamente validada em três domínios distintos de séries temporais:
1. **Turbofan (C-MAPSS):** Previsão da vida útil restante (Remaining Useful Life - RUL) de motores aeronáuticos.
2. **Air Quality:** Previsão de curto prazo da concentração de monóxido de carbono baseada em variáveis climáticas e de outros gases.
3. **Electricity Transformer Temperature (ETTh1):** Previsão de curto a longo prazo da temperatura do óleo de transformadores de potência.

## 📂 Repositórios Relacionados

O escopo desta pesquisa foi subdividido para facilitar a organização. Além deste repositório principal com o algoritmo Q-MHNAS, existem outros dois repositórios fundamentais para o projeto:

* **[Q-MHNAS_Exploratory_Analysis](https://github.com/marcelov-aguiar/Q-MHNAS_Exploratory_Analysis):** Contém os notebooks e scripts utilizados para a análise exploratória e o pré-processamento dos dados dos três estudos de caso mencionados.
* **[QIEA_Multi_Head](https://github.com/marcelov-aguiar/QIEA_Multi_Head):** Repositório dedicado aos experimentos isolados de comparação entre o algoritmo evolucionário de inspiração quântica (QIEA) e o Evolutionary NAS (ENAS) reportado na literatura.

## 🙏 Agradecimentos e Repositórios Base

Este projeto foi construído com base em trabalhos disponibilizadas de forma aberta na comunidade. Gostaria de referenciar e agradecer aos autores dos seguintes repositórios que serviram de base para a construção dos códigos:

* **[qnas_torch](https://github.com/DiegoPaezA/qnas_torch):** Implementação base em PyTorch do algoritmo Q-NAS original. O código serviu como ponto de partida para as adaptações estruturais da busca quântica para regressão.

* **[ENAS-PdM](https://github.com/mohyunho/ENAS-PdM):** Repositório oficial do método de referência ENAS. Utilizei este repositório para fundamentar a construção da macro-arquitetura Multi-Head. O código, originalmente escrito em TensorFlow, foi integralmente traduzido e adaptado para PyTorch neste projeto.

## ⚙️ Instalação e Ambiente

Para replicar os experimentos, é necessário configurar corretamente o ambiente Python. O processo de instalação foi dividido em duas opções, dependendo da arquitetura da sua placa de vídeo:

### Instalação Padrão (Recomendada)
Esta é a configuração principal validada ao longo da maior parte do projeto.

1. **Crie o ambiente virtual:** O ambiente padrão foi criado utilizando a versão **Python 3.8.9** através da biblioteca `virtualenv`.

2. **Instale o PyTorch:** Recomenda-se a instalação das bibliotecas do ecossistema Torch primeiro, de acordo com a compatibilidade do seu sistema e versão do CUDA (consulte o site oficial do PyTorch). Abaixo estão as versões exatas utilizadas no projeto:

```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
   --index-url https://download.pytorch.org/whl/cu121
```

3. **Instale as dependências gerais:**

```bash
   pip install -r requirements.txt
```

⚠️ **Instalação Alternativa: Placas NVIDIA RTX Série 50**

Caso você vá executar os scripts em uma GPU NVIDIA da série 50 (ex: RTX 5090), utilize as configurações abaixo:

1. **Crie o ambiente virtual**: Utilize a versão Python 3.10.20.

2. **Instale o PyTorch (Versão de pré-lançamento/Nightly):**

```bash
pip install \
torch==2.12.0.dev20260313+cu128 \
torchvision==0.26.0.dev20260313+cu128 \
torchaudio==2.11.0.dev20260313+cu128 \
--index-url https://download.pytorch.org/whl/nightly/cu128
``` 

3. **Instale as dependências específicas:**

```bash
pip install -r requirements_rtx50.txt
```
Observação: Como o ambiente da série 50 exige versões mais recentes das bibliotecas base, pequenos ajustes pontuais no código-fonte podem ser necessários para garantir a execução dos scripts.

**Hardware Utilizado**
Os extensos experimentos de busca arquitetural e retreinamento das redes (citados na dissertação) foram validados em diversas arquiteturas de alto desempenho, incluindo as seguintes GPUs:

- NVIDIA RTX 3090
- NVIDIA RTX 4060
- NVIDIA RTX 4090
- NVIDIA RTX 5080
- NVIDIA L40S
