#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>

#include "imagem.h"

using namespace std;

/* Rotina para somar dois vetores na GPU */ 
__global__ void blur(unsigned char *input, unsigned char *output,  int height, int width){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    double media;
    if (i< height && j< width) {
        if(i==0 || i==height-1){
            media = 0;
        }
        else if(j == 0 || j == width-1){
            media = 0;
        }
        else{
            media = (input[i*width+j]+input[(i-1)*width + j-1]+input[(i-1)*width + j]+input[(i-1)*width + j+1]+input[(i-1)*width + j-1]+input[i*width+ j-1]+input[i*width+ j+1]+input[(i+1)*width + j-1]+input[(i+1)*width + j]+input[(i+1)*width + j+1])/9;
            // int media = (input[i][j-1]+input[i][j+1]+input[i][j]+input[i+1][j-1]+input[i+1][j+1]+input[i+1][j]+input[i-1][j-1]+input[i-1][j+1]+input[i-1][j])/9;
        }
        output[i*width+j] = media;
    }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {

    cout << "inicio" << '\n';

    imagem *img = read_pgm("mona.pgm");

    cout << "lido" << '\n';

    imagem *saida = new_image(img->rows, img->cols);

    cout << "saida" << '\n';

    thrust::device_vector<unsigned char> V1_d(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> V2_d(saida->pixels, saida->pixels + saida->total_size );

    
    dim3 dimGrid (ceil(img->rows/36), ceil(img->cols/36),1);
    dim3 dimBlock(36, 36, 1);

    blur<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(V1_d.data()), thrust::raw_pointer_cast(V2_d.data()), img->rows, img->cols);

    thrust::host_vector<unsigned char> V3_h(V2_d);
    for(int i = 0; i != V3_h.size(); i++) {
        saida->pixels[i] = V3_h[i];
    }
    write_pgm(saida, "blured.pgm");
    cout << "DONE!" << std::endl;

    return 0;
}
