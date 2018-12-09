all: gpu seq 

gpu: mainGPUMult.cu imagem.cpp imagem.h
	nvcc -std=c++11 -lnvgraph mainGPUMult.cu imagem.cpp -o gpu 

seq: mainSeqMult.cu imagem.cpp imagem.h
	nvcc -std=c++11 mainSeqMult.cu imagem.cpp -o seq 