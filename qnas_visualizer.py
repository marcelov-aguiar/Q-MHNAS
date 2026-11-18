import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.ticker as ticker
import seaborn as sns

class QNASVisualizer:

    @staticmethod
    def plot_individual(
        data,
        generations,
        individual_idx,
        labels,
        palette=None,
        figsize=(15, 6),
        bar_width=0.8,
        show=True,
    ):
        """
        Plots stacked bar charts for given generations side by side, with custom colors and adjustable figure size.

        Args:
                data (dict): your data dict where data[gen]['net_probs'] has shape (num_individuals, num_nodes, num_functions)
                generations (list of int): e.g. [0, last_gen]
                individual_idx (int): which individual to plot (0–N)
                labels (list of str): list of operator names (length = num_functions)
                palette (dict): optional mapping label → hex color
                figsize (tuple): (width, height) in inches
                bar_width (float): width of each bar (0.5 → big gaps, 0.9 → small gaps)
                show (bool): whether to display the plot
        """
        n_gens = len(generations)
        fig, axes = plt.subplots(1, n_gens, figsize=figsize, sharex=True, sharey=True)

        if n_gens == 1:
            axes = [axes]

        # Build color list matching label order
        colors = [palette[label] for label in labels] if palette else None

        for ax, gen in zip(axes, generations):
            probs = data[gen]["net_probs"][
                individual_idx
            ]  # shape: (num_nodes, num_functions)
            df = pd.DataFrame(probs, columns=labels)

            df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=bar_width,
                edgecolor="white",
                linewidth=0.5,
                color=colors,
                legend=False,
            )

            ax.set_title(
                f"Individual {individual_idx} – Gen {gen}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(np.arange(probs.shape[0]))
            ax.set_xticklabels(np.arange(probs.shape[0]), rotation=0)
            ax.set_xlabel("Network nodes", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.grid(False)

        # Add one legend to the right of the last subplot
        axes[-1].legend(
            labels, bbox_to_anchor=(1.02, 1), loc="upper left", title="Functions"
        )

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_hyperparameter_evolution_grid(
        dados: Dict[int, Dict[str, np.ndarray]],
        geracoes: List[int],
        individual_index: int,
        param_names: List[str],
        final_values: Optional[List[float]] = None,
        show=True,
    ):
        """
        Plota uma grade de gráficos comparando a evolução da PDF de múltiplos
        hiperparâmetros para um único indivíduo quântico.

        Args:
            dados (Dict):
                Dicionário com os dados de evolução, onde a chave é a geração (int).
                Cada valor é um dict com as chaves 'lower' e 'upper'.

            geracoes (List[int]):
                Lista com duas gerações para comparar: [geração_inicial, geração_final].

            individual_index (int):
                O índice do indivíduo quântico a ser plotado.

            param_names (List[str]):
                Lista com os nomes dos hiperparâmetros, que serão usados
                como títulos dos subplots.

            final_values (Optional[List[float]], optional):
                Lista opcional com os valores finais (melhores) para cada
                hiperparâmetro, que serão plotados como uma linha vertical tracejada.
                        [0.35, 0.48, 4.5e-4] # Valores para a linha tracejada do hiperparâmetro.
        """
        # --- Validação das Entradas ---
        if len(geracoes) != 2:
            raise ValueError("A lista 'geracoes' deve conter exatamente dois números.")

        gen_inicial_num, gen_final_num = geracoes[0], geracoes[1]

        try:
            dados_inicial = dados[gen_inicial_num]
            dados_final = dados[gen_final_num]
        except KeyError as e:
            raise KeyError(f"A geração {e} não foi encontrada no dicionário de dados.")

        num_params = dados_inicial["lower"].shape[1]
        if len(param_names) != num_params:
            raise ValueError(
                f"O número de nomes de parâmetros ({len(param_names)}) não "
                f"corresponde ao número de hiperparâmetros nos dados ({num_params})."
            )

            # --- Configuração do Layout do Grid ---
        if num_params <= 3:
            num_rows, num_cols = 1, num_params
        else:
            num_cols = 3
            num_rows = int(np.ceil(num_params / num_cols))

        plt.style.use("seaborn-notebook")
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), squeeze=False
        )

        # Cores fixas
        cor_inicial = "#c0392b"  # Vermelho
        cor_final = "#34495e"  # Azul/Cinza escuro

        # --- Itera e plota cada hiperparâmetro ---
        for i, ax in enumerate(axes.flat):
            if i >= num_params:
                ax.set_visible(False)
                continue

            # Extração e Cálculo dos Dados
            lower_inicial = dados_inicial["lower"][individual_index, i]
            upper_inicial = dados_inicial["upper"][individual_index, i]
            lower_final = dados_final["lower"][individual_index, i]
            upper_final = dados_final["upper"][individual_index, i]

            width_inicial = upper_inicial - lower_inicial
            width_final = upper_final - lower_final

            altura_inicial = 1.0 / width_inicial if width_inicial > 0 else 0
            altura_final = 1.0 / width_final if width_final > 0 else 0

            # Eixo Principal (Esquerda) - INICIAL (Vermelho)
            ax_twin = ax.twinx()
            x_inicial = [lower_inicial, lower_inicial, upper_inicial, upper_inicial]
            y_inicial = [0, altura_inicial, altura_inicial, 0]
            (p1,) = ax.plot(
                x_inicial, y_inicial, color=cor_inicial, linewidth=2, label="initial"
            )  # Captura o handle p1
            ax.tick_params(axis="y", labelcolor=cor_inicial)
            ax.set_ylim(0, altura_inicial * 1.15)

            # Eixo Gêmeo (Direita) - FINAL (Azul/Cinza)
            x_final = [lower_final, lower_final, upper_final, upper_final]
            y_final = [0, altura_final, altura_final, 0]
            (p2,) = ax_twin.plot(
                x_final, y_final, color=cor_final, linewidth=2, label="final"
            )  # Captura o handle p2
            ax_twin.tick_params(axis="y", labelcolor=cor_final)
            ax_twin.set_ylim(0, altura_final * 1.15)

            # Linha Tracejada
            p3 = None
            if final_values:
                p3 = ax_twin.axvline(
                    x=final_values[i],
                    color=cor_final,
                    linestyle="--",
                    linewidth=1.2,
                    label="final values",
                )  # Captura o handle p3

            # Anotações e Rótulos
            ax.set_title(param_names[i], fontsize=14)
            width_ratio = width_final / width_inicial if width_inicial > 0 else 0
            ax.text(
                0.05,
                0.9,
                f"width ratio = {width_ratio:.2f}",
                transform=ax.transAxes,
                fontsize=10,
            )

            if i % num_cols == 0:
                ax.set_ylabel(f"Individual {individual_index}", fontsize=12)

        # Monta a lista de handles (objetos das linhas) para a legenda manualmente
        legend_handles = [p1, p2]
        if p3:  # Adiciona o handle da linha tracejada se ele foi criado
            legend_handles.append(p3)

        # Cria a legenda para a figura inteira usando a lista de handles montada
        fig.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show()
        return fig

    @staticmethod
    def plot_fitness_evolution(
        generations: List[int],
        gen_best: List[float],
        best_fitness: List[float],
        avg_fitness: List[float],
        worst_fitness: List[float],
        show_best_fitness: bool = True,
        show: bool = True,
    ):
        """
        Plot fitness (loss) evolution over generations for the training data.

        Args:
            generations (List[int]): List of generation indices.
            gen_best (List[float]): Best fitness values for each generation.
            best_fitness (List[float]): Best fitness values observed so far.
            avg_fitness (List[float]): Average fitness per generation.
            worst_fitness (List[float]): Worst fitness per generation.
            show_best_fitness (bool): Whether to plot the global best fitness curve.
            show (bool): Whether to display the figure interactively (default: True).

        Returns:
                matplotlib.figure.Figure: The generated matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # ax.plot(generations, gen_best, color='lime', alpha=0.4, marker='o', linewidth=0.8, label='Best in Generation')
        if show_best_fitness:
            ax.plot(
                generations,
                best_fitness,
                color="green",
                alpha=0.4,
                marker="o",
                linewidth=0.8,
                label="Best Fitness",
            )
        ax.plot(
            generations,
            avg_fitness,
            color="blue",
            alpha=0.4,
            marker="o",
            linewidth=0.8,
            label="Average Fitness",
        )
        ax.plot(
            generations,
            worst_fitness,
            color="red",
            alpha=0.4,
            marker="o",
            linewidth=0.8,
            label="Worst Fitness",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Loss)")
        ax.set_title("QNAS Fitness Evolution")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        fig.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def aggregate_nodes_by_head(
        probs: np.ndarray, num_heads: int, nodes_per_head: int, mode: str = "mean"
    ) -> np.ndarray:
        """
        Aggregates node probabilities by head.

        Args:
                probs (np.ndarray): shape (num_nodes, num_functions)
                num_heads (int): number of heads (e.g. 14)
                nodes_per_head (int): number of nodes per head (e.g. 6)
                mode (str): aggregation mode ("mean" or "max")

        Returns:
                np.ndarray: shape (num_heads, num_functions)
        """
        aggregated = []

        for h in range(num_heads):
            start = h * nodes_per_head
            end = start + nodes_per_head
            head_probs = probs[start:end]

            if mode == "mean":
                aggregated.append(head_probs.mean(axis=0))
            elif mode == "max":
                aggregated.append(head_probs.max(axis=0))
            else:
                raise ValueError(f"Unsupported aggregation mode: {mode}")

        return np.vstack(aggregated)

    @staticmethod
    def plot_individual_head_aggregation(
        data,
        generations,
        individual_idx,
        labels,
        palette=None,
        figsize=(20, 6),
        bar_width=0.8,
        show=True,
        num_heads=14,
        nodes_per_head=6,
        aggregate_by_head=True,
    ):
        """
        Plots stacked bar charts for given generations.
        If aggregate_by_head=True, probabilities are averaged per head.

        Args:
                data (dict): data[gen]['net_probs'] → (num_individuals, num_nodes, num_functions)
                generations (list[int]): generations to plot
                individual_idx (int): individual index
                labels (list[str]): function names
                palette (dict): label → color
                figsize (tuple): figure size
                bar_width (float): bar width
                show (bool): show plot
                num_heads (int): number of heads
                nodes_per_head (int): nodes per head
                aggregate_by_head (bool): whether to aggregate nodes per head
        """
        n_gens = len(generations)
        fig, axes = plt.subplots(1, n_gens, figsize=figsize, sharey=True)

        if n_gens == 1:
            axes = [axes]

        colors = [palette[label] for label in labels] if palette else None

        for ax, gen in zip(axes, generations):
            probs = data[gen]["net_probs"][individual_idx]  # (num_nodes, num_functions)

            if aggregate_by_head:
                probs = QNASVisualizer.aggregate_nodes_by_head(
                    probs,
                    num_heads=num_heads,
                    nodes_per_head=nodes_per_head,
                    mode="mean",
                )
                x_labels = [f"H{h}" for h in range(num_heads)]
                xlabel = "Heads"
            else:
                x_labels = np.arange(probs.shape[0])
                xlabel = "Network nodes"

            df = pd.DataFrame(probs, columns=labels)

            df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                width=bar_width,
                edgecolor="white",
                linewidth=0.5,
                color=colors,
                legend=False,
            )

            ax.set_title(
                f"Individual {individual_idx} – Gen {gen} (Head Average)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=0)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.grid(False)

        axes[-1].legend(
            labels, bbox_to_anchor=(1.02, 1), loc="upper left", title="Functions"
        )

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_head_heatmap(
        data,
        generation,
        individual_idx,
        labels,
        num_heads=14,
        nodes_per_head=6,
        figsize=(12, 7),
        cmap="YlGnBu",
        head_names=None,
        show=True,
    ):
        """Gera um Heatmap (14 Heads x 16 Funções) para visualizar a especialização
        das heads na Geração 99, atendendo à análise de convergência
        heterogênea.

        Args:
            data (dict): data[gen]['net_probs'] -> (num_individuals, num_nodes,
              num_functions)
            generation (int): Geração a ser plotada (ex: 99)
            individual_idx (int): Índice do indivíduo (ex: melhor indivíduo)
            labels (list[str]): Nome das 16 funções/blocos
            num_heads (int): Número de heads (14 para o Turbofan)
            nodes_per_head (int): Nós por head (6)
            figsize (tuple): Tamanho da figura
            cmap (str): Paleta de cores (ex: 'YlGnBu', 'Viridis', 'Blues', 'magma')
            head_names (list[str]): Nomes personalizados para as heads (ex: nomes dos
              sensores)
            show (bool): Se exibe o gráfico na tela
        """
        # 1. Extrai as probabilidades da geração escolhida: shape (84, 16)
        probs = data[generation]["net_probs"][individual_idx]

        # 2. Agrupa a média dos 6 nós por head: shape (14, 16)
        head_probs = QNASVisualizer.aggregate_nodes_by_head(
            probs, num_heads=num_heads, nodes_per_head=nodes_per_head, mode="mean"
        )

        # 3. Define os rótulos das linhas (Heads)
        if head_names is None:
            y_labels = [f"Head {h}" for h in range(num_heads)]
        else:
            y_labels = head_names

        # 4. Cria o DataFrame para o Seaborn
        df_heatmap = pd.DataFrame(head_probs, index=y_labels, columns=labels)

        # 5. Plota o Heatmap
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            df_heatmap,
            annot=True,  # Mostra o valor da probabilidade dentro da célula
            fmt=".2f",  # Formata com 2 casas decimais
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Probability Mass"},
            ax=ax,
        )

        ax.set_title(
            f"Categorical PMF Distribution per Head – Gen {generation} (Individual"
            f" {individual_idx})",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Candidate Convolutional Functions / Operations", fontsize=11)
        ax.set_ylabel("Input Sensor Heads", fontsize=11)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_stacked_bars_per_node_old(
        data,
        generation,
        individual_idx,
        labels,
        num_heads=14,
        nodes_per_head=6,
        figsize=(18, 10),
        cmap_name="tab20",
        head_names=None,
        show=True,
    ):
        """
        Gera um painel com 6 subgráficos (um por nó). 
        Cada subgráfico mostra as barras empilhadas (distribuição das PMFs) 
        para todas as heads naquele nó específico.
        """
        # 1. Extrai as probabilidades da geração escolhida: shape (84, 16)
        probs = data[generation]["net_probs"][individual_idx]
        num_functions = len(labels)

        # 2. Define os rótulos do eixo X (Heads)
        if head_names is None:
            x_labels = [f"H{h+1}" for h in range(num_heads)]
        else:
            x_labels = head_names

        # 3. Cria a paleta de cores (precisamos de 16 cores distintas)
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / num_functions) for i in range(num_functions)]

        # 4. Configura o painel (2 linhas x 3 colunas)
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        axes = axes.flatten()  # Transforma a matriz 2x3 em uma lista de 6 eixos

        x_pos = np.arange(num_heads)

        # 5. Itera sobre cada nó (Node 0 a Node 5)
        for node_idx in range(nodes_per_head):
            ax = axes[node_idx]
            
            # Extrai apenas as linhas de 'probs' correspondentes a este nó para todas as heads.
            # Como os genes estão em formato plano (84 genes), o nó 'i' da head 'h' 
            # está no índice: h * nodes_per_head + node_idx
            node_probs = probs[node_idx::nodes_per_head, :]  # Shape: (14, 16)
            
            bottom = np.zeros(num_heads)
            
            # Empilha as barras para as 16 funções
            for func_idx in range(num_functions):
                func_probs = node_probs[:, func_idx]
                ax.bar(
                    x_pos, 
                    func_probs, 
                    bottom=bottom, 
                    color=colors[func_idx], 
                    edgecolor="white",
                    linewidth=0.5,
                    label=labels[func_idx] if node_idx == 0 else "" # Legenda apenas no 1º gráfico
                )
                bottom += func_probs
            
            # Formatação do subgráfico
            ax.set_title(f"Node {node_idx + 1}", fontsize=12, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, 1.01)
            
            if node_idx % 3 == 0:
                ax.set_ylabel("Probability Mass", fontsize=11)

        # 6. Formatação Global
        fig.suptitle(
            f"Categorical PMF Distribution by Node – Gen {generation} (Individual {individual_idx})\n"
            f"Comparing structural specialization across {num_heads} independent heads",
            fontsize=14, fontweight="bold", y=0.98
        )
        
        # Coloca uma única legenda global abaixo de todos os gráficos
        fig.legend(
            loc="lower center", 
            ncol=8, 
            bbox_to_anchor=(0.5, -0.05),
            fontsize=10,
            title="Candidate Convolutional Functions / Operations"
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15, top=0.88) # Abre espaço para a legenda embaixo e título em cima

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_stacked_bars_per_node(
        data,
        generation,
        individual_idx,
        labels,
        palette_dict,  # <-- Novo parâmetro: Dicionário de cores da QNASPalette
        num_heads=14,
        nodes_per_head=6,
        figsize=(18, 10),
        head_names=None,
        show=True,
    ):
        """
        Gera um painel com 6 subgráficos (um por nó). 
        Usa a paleta de cores padronizada do QNAS.
        """
        probs = data[generation]["net_probs"][individual_idx]
        num_functions = len(labels)

        if head_names is None:
            x_labels = [f"H{h+1}" for h in range(num_heads)]
        else:
            x_labels = head_names

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        axes = axes.flatten()

        x_pos = np.arange(num_heads)

        for node_idx in range(nodes_per_head):
            ax = axes[node_idx]
            
            # Extrai apenas as linhas correspondentes a este nó para todas as heads
            node_probs = probs[node_idx::nodes_per_head, :]  # Shape: (14, 16)
            
            bottom = np.zeros(num_heads)
            
            for func_idx in range(num_functions):
                func_name = labels[func_idx]
                func_probs = node_probs[:, func_idx]
                
                ax.bar(
                    x_pos, 
                    func_probs, 
                    bottom=bottom, 
                    color=palette_dict[func_name],  # <-- Aplica a cor mapeada da sua classe
                    edgecolor="white",
                    linewidth=0.5,
                    label=func_name if node_idx == 0 else "" 
                )
                bottom += func_probs
            
            ax.set_title(f"Node {node_idx + 1}", fontsize=12, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, 1.01)
            
            if node_idx % 3 == 0:
                ax.set_ylabel("PMF", fontsize=11)

        # fig.suptitle(
        #     f"Categorical PMF Distribution by Node – Gen {generation} (Individual {individual_idx})\n"
        #     f"Comparing structural specialization across {num_heads} independent heads",
        #     fontsize=14, fontweight="bold", y=0.98
        # )
        
        fig.legend(
            loc="lower center", 
            ncol=8, 
            bbox_to_anchor=(0.5, -0.05),
            fontsize=10,
            title="Convolutional blocks"
        )

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15, top=0.88)

        if show:
            plt.show()

        return fig