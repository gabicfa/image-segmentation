import os

from subprocess import PIPE, run

tamanhos =[]

caminhos_minimos_seq =[]
montagem_imagemSeg_seq =[]
tempo_total_seq = []

montagem_grafo_gpu =[]
caminhos_minimos_gpu = []
montagem_imagemSeg_gpu =[]
tempo_total_gpu =[]

f = open("analysis.txt", "w+")

print("GPU imagem jump")
comando = "./gpu input_images/jump.pgm output_images/outjumpgpu.pgm < input_files/inputjump.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
tamanhos.append(lines[0])
montagem_grafo_gpu.append(lines[1])
caminhos_minimos_gpu.append(lines[2])
montagem_imagemSeg_gpu.append(lines[3])
tempo_total_gpu.append(lines[4])

print("GPU imagem test")
comando = "./gpu input_images/teste.pgm output_images/outtestegpu.pgm < input_files/inputteste.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
tamanhos.append(lines[0])
montagem_grafo_gpu.append(lines[1])
caminhos_minimos_gpu.append(lines[2])
montagem_imagemSeg_gpu.append(lines[3])
tempo_total_gpu.append(lines[4])

print("GPU image beer")
comando = "./gpu input_images/beer.pgm output_images/outbeergpu.pgm < input_files/inputbeer.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
tamanhos.append(lines[0])
montagem_grafo_gpu.append(lines[1])
caminhos_minimos_gpu.append(lines[2])
montagem_imagemSeg_gpu.append(lines[3])
tempo_total_gpu.append(lines[4])

print("GPU imagem lemon")
comando = "./gpu input_images/lemon.pgm output_images/outlemongpu.pgm < input_files/inputlemon.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
tamanhos.append(lines[0])
montagem_grafo_gpu.append(lines[1])
caminhos_minimos_gpu.append(lines[2])
montagem_imagemSeg_gpu.append(lines[3])
tempo_total_gpu.append(lines[4])

print("SEQ imagem jump")
comando = "./seq input_images/jump.pgm output_images/outjumpseq.pgm < input_files/inputjump.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
caminhos_minimos_seq.append(lines[0])
montagem_imagemSeg_seq.append(lines[1])
tempo_total_seq.append(lines[2])

print("SEQ imagem test")
comando = "./seq input_images/teste.pgm output_images/outtesteseq.pgm < input_files/inputteste.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
caminhos_minimos_seq.append(lines[0])
montagem_imagemSeg_seq.append(lines[1])
tempo_total_seq.append(lines[2])

print("SEQ imagem beer")
comando = "./seq input_images/beer.pgm output_images/outbeerseq.pgm < input_files/inputbeer.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
caminhos_minimos_seq.append(lines[0])
montagem_imagemSeg_seq.append(lines[1])
tempo_total_seq.append(lines[2])

print("SEQ imagem lemon")
comando = "./seq input_images/lemon.pgm output_images/outlemonseq.pgm < input_files/inputlemon.txt"
resultado = os.system(comando)
with open('out.txt') as o:
    lines = o.read().splitlines()
caminhos_minimos_seq.append(lines[0])
montagem_imagemSeg_seq.append(lines[1])
tempo_total_seq.append(lines[2])

f.write("tamanhos=[")
for i in range (0,len(tamanhos)):
    if(i!=len(tamanhos)-1):
        f.write(tamanhos[i] + ',')
    else:
        f.write(tamanhos[i])
f.write("]"+ '\n')

f.write("montagem_grafo_gpu=[")
for i in range (0,len(montagem_grafo_gpu)):
    if(i!=len(montagem_grafo_gpu)-1):
        f.write(montagem_grafo_gpu[i] + ',')
    else:
        f.write(montagem_grafo_gpu[i])
f.write("]"+ '\n')

f.write("caminhos_minimos_gpu=[")
for i in range (0,len(caminhos_minimos_gpu)):
    if(i!=len(caminhos_minimos_gpu)-1):
        f.write(caminhos_minimos_gpu[i] + ',')
    else:
        f.write(caminhos_minimos_gpu[i])
f.write("]"+ '\n')

f.write("montagem_imagemSeg_gpu=[")
for i in range (0,len(montagem_imagemSeg_gpu)):
    if(i!=len(montagem_imagemSeg_gpu)-1):
        f.write(montagem_imagemSeg_gpu[i] + ',')
    else:
        f.write(montagem_imagemSeg_gpu[i])
f.write("]"+ '\n')

f.write("tempo_total_gpu=[")
for i in range (0,len(tempo_total_gpu)):
    if(i!=len(tempo_total_gpu)-1):
        f.write(tempo_total_gpu[i] + ',')
    else:
        f.write(tempo_total_gpu[i])
f.write("]"+ '\n')

f.write("caminhos_minimos_seq=[")
for i in range (0,len(caminhos_minimos_seq)):
    if(i!=len(caminhos_minimos_seq)-1):
        f.write(caminhos_minimos_seq[i] + ',')
    else:
        f.write(caminhos_minimos_seq[i])
f.write("]"+ '\n')

f.write("montagem_imagemSeg_seq=[")
for i in range (0,len(montagem_imagemSeg_seq)):
    if(i!=len(montagem_imagemSeg_seq)-1):
        f.write(montagem_imagemSeg_seq[i] + ',')
    else:
        f.write(caminhos_minimos_seq[i])
f.write("]"+ '\n')

f.write("tempo_total_seq=[")
for i in range (0,len(tempo_total_seq)):
    if(i!=len(tempo_total_seq)-1):
        f.write(tempo_total_seq[i] + ',')
    else:
        f.write(tempo_total_seq[i])
f.write("]"+ '\n')