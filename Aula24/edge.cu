#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>

#include "imagem.h"

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

using namespace std;

__global__ void edge(unsigned char *in, unsigned char *out, int rowStart, int rowEnd, int colStart, int colEnd){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    int di, dj;    
    if (i< rowEnd && j< colEnd) {
        int min = 256;
        int max = 0;
        for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
               if(min>in[di*(colEnd-colStart)+dj]) min = in[di*(colEnd-colStart)+dj];
               if(max<in[di*(colEnd-colStart)+dj]) max = in[di*(colEnd-colStart)+dj]; 
            }
        }
        out[i*(colEnd-colStart)+j] = max-min;
    }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {


    imagem *img = read_pgm("mona.pgm");
    imagem *saida = new_image(img->rows, img->cols);

    thrust::device_vector<unsigned char> V1_d(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> V2_d(saida->pixels, saida->pixels + saida->total_size );
    
    dim3 dimGrid (ceil(img->rows/16), ceil(img->cols/16),1);
    dim3 dimBlofck(16, 16, 1);

    edge<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(V1_d.data()), thrust::raw_pointer_cast(V2_d.data()), 0, img->rows, 0, img->cols);

    thrust::host_vector<unsigned char> V3_h(V2_d);
    for(int i = 0; i != V3_h.size(); i++) {
        saida->pixels[i] = V3_h[i];
    }

    write_pgm(saida, "mona_edge.pgm");

    return 0;
}
