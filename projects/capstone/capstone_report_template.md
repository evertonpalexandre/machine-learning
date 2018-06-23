# Engenheiro Machine Learning Nanodegree
## Projeto Final
Everton Alexandre
23 de Junho de 2018

## I. Definição
_(approx. 1-2 pages)_

### Visão Geral do Projeto
O presente artigo descreve o projeto de trabalho final do grupo Nanodegree Engenheiro de Machine Learning. O assunto escolhido para este trabalho está relacionado com a utilização de Machine Learning para a previsão do preço no mercado de ações. Fundos de investimento e bancos tem usado técnicas de aprendizado de máquina de forma a ter um entendimento melhor sobre o comportamento do mercado financeiro. Existem muitas API's que fornecem dados históricos sobre este tema. Além disso, tenho negociado no mercado de ações e mercado futuro há mais de 5 anos, fato que me motivou a escolher este tema para o projeto final sobre Machine Learning.

O mercado de ações no Brasil movimenta bilhões de reais todos os dias. Milhões de negócios são fechados a cada dia. O nosso objetivo nesse trabalho será prever o comportamento dos preços das ações do mercado financeiro brasileiro. Como dados de entrada serão utilizados os preços de abertura, máxima, mínima, fechamento e volume negociado. Este problema será resolvido através da classe de modelos supervisionados, mais especificamente de regressão. As previsões serão realizadas para o dia seguinte.


### Definição do Problema
O mercado de ações é algo bem complexo, uma vez que existem diversos fatores que influenciam no preço dos ativos. Por exemplo, a greve de caminhoneiros que ocorreu em maio de 2018 no Brasil contribuiu para que o índice bovespa caísse cerca de 10%. Tais situações são difíceis de prever e estão fora do controle até mesmo de profissionais da área financeira. A previsibilidade tem interessado o setor financeiro e existem vários estudos que buscam criar estratégias vencedoras no mercado de ações.

Tentar prever o mercado de ações é uma perspectiva atraente para os cientistas de dados, motivados não apenas pelo desejo de ganho material, mas pelo desafio. Vemos os altos e baixos diários do mercado e imaginamos que deve haver padrões. Este contexto me motivou a utilizar uma solução de Machine Learning aplicada ao mercado financeiro. Basicamente, realizaremos a previsão do preço de fechamento da ação, ou seja, o fechamento será a variável alvo e o preço de abertura, preço máximo, preço mínimo e volume serão os dados de entrada. Primeiramente iremos obter os dados utilizando a ferramenta Google Finance. Será analisado um horizonte de 11 anos aproximadamente. Após, iremos realizar a manipulação dos dados através da biblioteca Pandas e Numpy. Nesta etapa será realizada a preparação dos dados, identificação dos atributos e variáveis alvo. Após, utilizaremos a bibliteca sklearn.TimeSeriesSplit para separação dos dados em treinamento e teste. Esta biblioteca será usada, pois caso usássemos o crossvalidation tradicional para avaliar o modelo, poderia acabar com uma noção da capacidade preditiva do modelo um pouco alterada (normalmente superestimada) e na hora de usar ele na prática este acabaria não correspondendo às expectativas. Dessa forma, iremos utilizar alguns anos para o treinamento e outro período para teste. Os dados de 2009 até 2015 serão utilizados para treinamento e 2016 até 2018 para testes. Em seguida, será o momento de realizar a escolha do modelo e realizar treinamento e testes. Após escolher o melhor modelo será o momento que realizar os ajustes finos no modelo, configurando e calibrando os parâmetros. Para tanto, utilizaremos GridSearch. Por fim, utilizaremos sklearn.metrics para verificar a pontuação do modelo no treinamento e teste.

### Métricas
Como estamos tratando de um problema de regressão e não de classificação o modelo não irá acertar a previsão com exatidão. Ou
seja, provavelmente o número previsto não será o número real. Entretanto, o que se espera desse modelo é que apresente uma taxa de erro baixa. Dessa forma, será utilizado neste trabalho como métrica o "Root Mean Square Error" (RMSE). O RMSE é uma medida utilizada frequentemente para diferenças entre os valores (amostra ou população de valores) previstos por um modelo ou um estimador e os valores observados. A fórmula pode ser representada da seguinte forma:



O RMSE é sempre um valor não negativo. Um valor igual a 0 (nunca alcançado na prática) indicaria um ajuste perfeito aos dados. Em geral, um valor menor para o RMSE é melhor do que um valor maior.
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
