
# Projeto 01 - Computação Paralela

**Grupo:** Gabriel Vargas, Leticia Americano, Nilson Deon e Olga Camilla  
**Data:** Novembro/2024  

## Descrição
Este projeto é uma aplicação de redes neurais treinada com o dataset Iris. Utilizamos três versões para a execução do algoritmo de Perceptron com backpropagation: sequencial, paralela com OpenMP e distribuída com MPI, explorando diferentes abordagens de paralelização para otimizar o treinamento da rede neural.

O código original não foi desenvolvido pelo grupo e está disponível no repositório: alexandremstf/neural-network (Autor: Alexandre Magno).

## Estrutura do Projeto
A estrutura básica do código se organiza em módulos para tratar das diferentes configurações de execução (Sequencial, OpenMP, MPI) e nas funções de implementação da rede neural e do treinamento via backpropagation. Abaixo, uma visão geral dos principais diretórios e arquivos:

- `src/`: Código-fonte principal, com versões Sequencial, OpenMP, e MPI do treinamento da rede neural.
- `Makefile`: Configurações de compilação para cada versão do código.

## Como Executar
### Compilação
Para compilar o código, use o comando:

```bash
make all VERSION=x
```

Substitua `x` pela versão desejada:
- `Sequencial`: para a execução sequencial.
- `OpenMP`: para a versão com paralelização utilizando OpenMP.
- `MPI`: para a versão distribuída utilizando MPI.

Exemplo de compilação para a versão com OpenMP:

```bash
make all VERSION=OpenMP
```

### Execução
Para executar o código compilado, utilize:

```bash
./neuralnetwork
```

### Dependências
Certifique-se de ter configurado as dependências necessárias para a execução de cada versão. Para compilar e executar o projeto, certifique-se de ter as seguintes dependências instaladas:

1. **Compilador C/C++**: Necessário para compilar o código em qualquer versão. Recomendado `gcc` ou `g++`.
2. **Make**: Utilizado para gerenciar a compilação do projeto.
3. **OpenMP**: Necessário para a versão paralela com OpenMP. Geralmente incluído no `gcc` (versão 4.2 ou superior).
4. **MPI (Message Passing Interface)**: Necessário para a versão distribuída com MPI. Recomendado `MPICH` ou `OpenMPI`.

## Explicação da aplicação
### Dataset: Iris
O dataset Iris é um conjunto de dados amplamente utilizado no aprendizado de máquina. Ele contém 150 amostras divididas igualmente entre três espécies de flores de íris: *Iris setosa*, *Iris virginica* e *Iris versicolor*. Cada amostra possui quatro características: comprimento e largura da sépala, e comprimento e largura da pétala. O objetivo da rede neural é classificar corretamente a espécie de uma flor com base nas suas características.

### Perceptron com Backpropagation
A rede neural utilizada neste projeto é baseada no modelo de Perceptron, uma das arquiteturas mais simples e eficazes para problemas de classificação linear. Neste caso, a rede neural é composta por uma camada de entrada, uma ou mais camadas ocultas, e uma camada de saída. O treinamento é realizado através do algoritmo de backpropagation, que ajusta os pesos da rede para minimizar o erro entre a saída prevista e a saída desejada.

### Versões de Execução
- **Sequencial**: Treinamento da rede neural executado em um único núcleo.
- **OpenMP**: Paralelização baseada em threads, utilizando OpenMP para acelerar o treinamento distribuindo o processamento entre múltiplos núcleos.
- **MPI**: Implementação distribuída utilizando MPI para execução em ambientes com múltiplas máquinas, dividindo o trabalho entre processos independentes.

## Alterações no código

1. **Configuração da Camada de Saída**: Alteramos `output_layer_size` de 1 para 3 para permitir múltiplas saídas no problema de classificação.
2. **Aprimoramento de `autoTraining`**: Automatizamos a busca por configurações ótimas de camada oculta e taxa de aprendizado.
3. **Saídas Detalhadas em `trainingClassification`**: Adicionamos informações sobre o progresso do treinamento, como taxa de acerto e número de épocas.
4. **Cálculos Otimizados em `ForwardPropagation` e `BackPropagation`**: Ajustamos o cálculo de ativações e erros para lidar com múltiplas classes de saída.
5. **Inicialização Simplificada**: Consolidamos a configuração dos parâmetros principais em uma função (`setParameter`).
6. **Cálculo de Taxa de Acerto**: Otimizamos o cálculo para melhorar a precisão na avaliação de desempenho.

Essas mudanças melhoram a precisão, flexibilidade e eficiência do código, com suporte a paralelização e otimização de parâmetros.

## Conclusão 
Durante a execução do projeto, foram realizados testes para comparar o desempenho entre as versões. A paralelização permite um treinamento mais rápido nas versões OpenMP e MPI, especialmente em conjuntos de dados maiores ou redes neurais mais complexas.

O código foi executado em um computador com as seguintes especificações:

### Versão Sequencial

- **Tempo**: 29.424 segundos

### Versão OpenMP

- **Tempo 1 Thread**: 26.720 segundos
- **Tempo 2 Threads**: 15.170 segundos
- **Tempo 4 Threads**: 8.984 segundos
- **Tempo 8 Threads**: 5.292 segundos

### Versão MPI

- **Tempo 1 Processo e 4 Threads**: 20.800 segundos
- **Tempo 2 Processos e 2 Threads**: 12.479 segundos
- **Tempo 4 Processos e 0 Threads**: 9.553 segundos

Esses resultados demonstram uma clara redução no tempo de execução com o aumento do nível de paralelismo. A versão OpenMP apresenta uma redução significativa do tempo à medida que o número de threads aumenta, com a execução em 8 threads sendo a mais rápida. A versão MPI também mostra uma diminuição do tempo de execução com o aumento do número de processos, destacando-se a execução com 4 processos como a mais eficiente entre as configurações testadas.
