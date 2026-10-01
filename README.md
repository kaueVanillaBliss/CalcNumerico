# Métodos Numéricos em Julia

Este repositório, desenvolvido por Carlos Kaue, contém implementações de diversos métodos numéricos escritos na linguagem de programação Julia. O projeto está estruturado em duas áreas principais da análise numérica: a resolução de Sistemas Lineares e o cálculo de Zeros de funções.

## Estrutura do Repositório

O código está organizado nos seguintes diretórios e arquivos:

### Sistemas Lineares
Esta pasta contém os algoritmos necessários para a resolução de sistemas de equações lineares:
* **decomposicao_lu.jl**: Implementação do método de decomposição LU.
* **eliminacao_gaussiana.jl**: Implementação do método de eliminação de Gauss.
* **gauss_jacobi.jl**: Implementação do método iterativo de Gauss-Jacobi.
* **gauss_seidel.jl**: Implementação do método iterativo de Gauss-Seidel.

### Zeros
Esta pasta contém os algoritmos utilizados para encontrar as raízes (zeros) de funções reais:
* **bissecao.jl**: Implementação do método da Bisseção.
* **falsa_posicao.jl**: Implementação do método da Falsa Posição.
* **newton.jl**: Implementação do método de Newton-Raphson.
* **secante.jl**: Implementação do método da Secante.

## Pré-requisitos

Para executar os scripts contidos neste repositório, é estritamente necessário ter a linguagem Julia instalada em seu ambiente de desenvolvimento.

## Como Executar

Para executar qualquer um dos métodos, abra o terminal, navegue até o diretório raiz do projeto e utilize o comando `julia` seguido do caminho para o arquivo correspondente. Por exemplo:

```bash
julia Sistemas_Lineares/eliminacao_gaussiana.jl
