% ---------------------------------------------
\subsection{Micro-arquitetura e Codificação do Cromossomo}
\label{sec:micro_cromossomo}
% ---------------------------------------------

A micro-arquitetura do Q-MHNAS define o espaço de busca arquitetural explorado pelo algoritmo evolucionário inspirado em computação quântica. Nesse nível são determinados automaticamente os blocos convolucionais de cada \textit{head}, o número de células das camadas LSTM e o comprimento das sequências de entrada $l_s$ utilizadas no processo de janelamento.

Além da definição do espaço de busca arquitetural, esta seção descreve também a forma como as arquiteturas candidatas são representadas no cromossomo quântico utilizado pelo algoritmo evolucionário. Dessa forma, são apresentados conjuntamente o espaço de busca e o mecanismo de codificação das arquiteturas, uma vez que ambos são diretamente relacionados.


% ---------------------------------------------
\subsubsection{Espaço de busca convolucional}
% ---------------------------------------------

Os blocos convolucionais constituem o principal elemento da busca arquitetural e seguem a representação da rede apresentada na Seção~\ref{sec:net-represetation}, herdada do Q-NAS.

No Q-MHNAS, o espaço de busca da arquitetura é composto por blocos convolucionais unidimensionais previamente definidos, além da operação especial \texttt{NoOp}, que permite representar arquiteturas com profundidade variável dentro de uma cadeia de comprimento fixo. Cada bloco convolucional representa uma configuração completa de convolução unidimensional, na qual o número de filtros, o tamanho do \textit{kernel} e o valor do \textit{stride} são definidos conjuntamente como uma única operação do espaço de busca.

Dessa forma, cada posição da cadeia arquitetural corresponde à escolha de um bloco convolucional específico ou da operação \texttt{NoOp}. A presença de operações \texttt{NoOp} permite que arquiteturas com diferentes profundidades efetivas sejam representadas dentro de uma estrutura de tamanho fixo, facilitando a aplicação do algoritmo evolucionário.


% ---------------------------------------------
\subsubsection{Estratégias Multi-Head}
% ---------------------------------------------

Além da definição dos blocos convolucionais, a micro-arquitetura do Q-MHNAS inclui duas diferentes estratégias de organização das \textit{heads} convolucionais. Essa escolha é realizada pelo usuário antes do início do processo evolutivo e determina a forma como o espaço de busca arquitetural será explorado pelo algoritmo.

Na primeira estratégia, denominada arquitetura convolucional compartilhada (do inglês, \textit{Shared Convolutional Architecture} - SCA), todas as \textit{heads} utilizam exatamente a mesma configuração de blocos convolucionais. O algoritmo de busca determina uma única sequência de blocos, que é replicada para todas as variáveis de entrada. Essa estratégia reduz a dimensionalidade do espaço de busca e impõe uma estrutura homogênea de extração de características entre as diferentes variáveis.

É importante destacar que o termo arquitetura convolucional compartilhada refere-se apenas à estrutura arquitetural, não significando compartilhamento de parâmetros. Dessa forma, embora as \textit{heads} possuam arquiteturas idênticas, cada extrator convolucional possui pesos independentes, os quais são ajustados separadamente durante o treinamento.

Por exemplo, considere três séries temporais ($N = 3$) como entrada. Se o algoritmo determinar uma arquitetura convolucional composta por dois blocos definidos por $\mathrm{Conv}(k=5, f=32, s=1)$ e $\mathrm{Conv}(k=3, f=64, s=1)$, a mesma sequência de blocos será aplicada a todas as \textit{heads}, resultando em três extratores convolucionais idênticos, conforme ilustrado na Figura~\ref{fig:multi_head_sca}.

\begin{figure}[htp]
    \centering
    \includegraphics[width=12cm]{main/images/multi_head_sca.png}
    \caption{Arquitetura \textit{Multi-Head} com convoluções compartilhadas, na qual a mesma sequência de blocos convolucionais é replicada em todas as variáveis de entrada.}
    \label{fig:multi_head_sca}
\end{figure}


Na segunda estratégia, denominada arquiteturas convolucionais independentes (do inglês, \textit{Independent Convolutional Architectures} - ICA), cada \textit{head} possui sua própria configuração de blocos convolucionais. Nessa abordagem, o algoritmo de busca pode especializar a extração de características para cada variável individualmente, permitindo maior flexibilidade arquitetural ao custo de um espaço de busca significativamente maior.

Por exemplo, considerando novamente $N = 3$ variáveis de entrada, o algoritmo pode determinar arquiteturas distintas para cada \textit{head}, conforme ilustrado na Figura~\ref{fig:multi_head_ica}.

\begin{figure}[htp]
    \centering
    \includegraphics[width=12cm]{main/images/multi_head_ica.png}
    \caption{Arquitetura \textit{Multi-Head} com arquiteturas convolucionais independentes entre as \textit{heads}, na qual cada variável de entrada é processada por uma sequência própria de blocos convolucionais.}
    \label{fig:multi_head_ica}
\end{figure}

Nesse caso, cada variável é processada por um extrator convolucional especializado, permitindo que o modelo capture padrões específicos de cada sinal de entrada.



% ---------------------------------------------
\subsubsection{Representação do Cromossomo}
\label{sec:cromossomo}
% ---------------------------------------------

As arquiteturas candidatas são representadas no Q-MHNAS por meio de cromossomos quânticos, seguindo a mesma abordagem adotada no Q-NAS. Cada cromossomo é composto por um conjunto de genes associados a Funções de Massa de Probabilidade (PMFs), que definem probabilisticamente as operações arquiteturais selecionadas durante o processo evolutivo.

Na representação adotada, a arquitetura convolucional é organizada como uma cadeia de comprimento fixo $L$, onde cada posição corresponde a um gene associado a uma PMF. Cada gene define uma operação selecionada a partir do conjunto discreto de blocos convolucionais candidatos, incluindo a operação \texttt{NoOp}. Durante a etapa de observação do cromossomo quântico, uma operação é amostrada de acordo com a distribuição de probabilidades associada a cada gene, definindo assim a arquitetura convolucional do indivíduo.

A estratégia de organização das \textit{heads} convolucionais é refletida diretamente na codificação do cromossomo. Na estratégia compartilhada (SCA), um único conjunto de $L$ genes representa a arquitetura convolucional aplicada a todas as \textit{heads}. Já na estratégia independente (ICA), cada \textit{head} possui um conjunto próprio de $L$ genes, permitindo que arquiteturas convolucionais distintas sejam evoluídas para cada variável de entrada.

Dessa forma, considerando $N$ variáveis de entrada, o comprimento total do cromossomo arquitetural é igual a $L$ na estratégia SCA e $N \times L$ na estratégia ICA.



% ---------------------------------------------
\subsubsection{Representação dos Hiperparâmetros}
% ---------------------------------------------

Além da configuração dos blocos convolucionais, o algoritmo de busca também determina o número de células das duas camadas LSTM e o comprimento da sequência de entrada $l_s$. Esses hiperparâmetros são incorporados ao cromossomo quântico seguindo a mesma representação utilizada no Q-NAS, na qual cada parâmetro numérico é modelado por uma distribuição uniforme definida em um intervalo contínuo.

Entretanto, diferentemente de hiperparâmetros tipicamente contínuos (como \textit{learning rate}), os parâmetros considerados pertencem naturalmente ao domínio dos números inteiros, uma vez que representam quantidades discretas, como o número de células LSTM e o tamanho das sequências de entrada. Dessa forma, após a etapa de observação do cromossomo quântico, os valores reais amostrados são mapeados para o domínio inteiro por meio da operação de arredondamento inferior (\textit{floor}). Esse procedimento preserva o mecanismo de amostragem contínua do Q-NAS durante a busca, ao mesmo tempo em que garante a geração de arquiteturas válidas.



Em arquiteturas \textit{Multi-Head} tradicionais encontradas na literatura, é comum adotar uma configuração convolucional uniforme, na qual todas as \textit{heads} compartilham exatamente a mesma arquitetura. Nesses modelos, além de ser mantida a mesma sequência de camadas convolucionais entre as \textit{heads}, também se repete a mesma configuração de convolução em todas as camadas, utilizando tipicamente valores fixos de \textit{kernel}, número de filtros e \textit{stride} ao longo de toda a rede.

O Q-MHNAS generaliza essa abordagem ao permitir tanto arquiteturas compartilhadas quanto arquiteturas independentes entre as \textit{heads}. Além disso, o método permite que diferentes camadas convolucionais utilizem configurações distintas de blocos, ampliando a flexibilidade arquitetural. Dessa forma, o método possibilita controlar o nível de especialização das representações convolucionais, ampliando o espaço de busca em relação às implementações \textit{Multi-Head} convencionais \cite{bib:Hyunho2021,bib:Mao2025}.