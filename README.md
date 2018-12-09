# Projeto 4: segmentação de imagens
======================

Para o quarto projeto de superComputação aplicou-se os conceitos de GPGPU vistos em aula em um problema real de segmentação de imagens.Uma segmentação de imagem é sua divisão em regiões conexas (sem interrupções) contendo objetos de interesse ou o fundo da imagem. A segmentação iterativa de imagens é baseada na adição de marcadores em objetos de interesse para diferenciá-los do fundo (que contém todos os outros objetos e plano de fundo da imagem). 

## Compilar e rodar

Para compilar tanto o programa sequencial quanto o em GPU rode o seguinte comando:

```
make
```

Esse comando gerará dois executáveis o ./seq e o ./gpu. Para executálos rode da seguinte maneira:

```
<executável> <imagem_entrada.pgm> <imagem_saida.pgm> < <arquivo_de_entrada.txt>
```

Existem exemplos de imagens de entrada na pasta input_images e de arquivos de entrada na pasta input_files

## Arquivo de entrada

Os arquivos de entrada devem ter o seguinte formato:

```
n_sementes_frente n_sementes_fundo
x_0_semente_frente y_0_semente_frente
x_1_semente_frente y_1_semente_frente
.
.
.
x_n-1_semente_frente y_n-1_semente_frente
x_0_semente_fundo y_0_semente_fundo
x_1_semente_fundo y_1_semente_fundo
.
.
.
x_n-1_semente_fundo y_n-1_semente_fundo
```

## Análise de tempo
Para adquirir os vetores para análise de tempo explicacado no ipython notebook da pasta "Relatório", rode o seguinte comando:

```
python run.py
```

os vetores serão escritos no arquivo analysis.txt
