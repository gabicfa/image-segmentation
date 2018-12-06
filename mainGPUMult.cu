#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "imagem.h"

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

using namespace std;

void check_status(nvgraphStatus_t status){
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

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

struct info{
    size_t n;
    size_t nnz;
    float * weights_h;
    int * destination_offsets_h;
    int * source_indices_h;
};

info imgInfo(imagem *in, vector<int>& seeds_fg, vector<int>& seeds_bg){
    
    int n_seeds = seeds_fg.size() + seeds_bg.size();

    vector<int> src_indices;
    vector<float> weights;
    vector<int> dest_offsets={0};

    info inf = {};

    inf.n = in->total_size + 2;
    inf.nnz = (2*((in->cols-1)*in->rows+(in->rows-1)*in->cols))+n_seeds;
    int offset_count = 0;

    for (int v = 0; v < in->total_size; v++) {

        int v_i = v / in->cols;
        int v_j = v % in->cols;

        if (find(begin(seeds_fg), end(seeds_fg), v) != end(seeds_fg)) {
            src_indices.push_back(in->total_size);
            weights.push_back(0.0);
            offset_count++;
        }

        if (find(begin(seeds_bg), end(seeds_bg), v) != end(seeds_bg)) {
            src_indices.push_back(in->total_size+1);
            weights.push_back(0.0);
            offset_count++;
        }

        if (v_i > 0) {
            int acima = v - in->cols;
            src_indices.push_back(acima);
            weights.push_back(get_edge(in, v, acima));
            offset_count++;
        }

        if (v_i < in->rows - 1) {
            int abaixo = v + in->cols;
            src_indices.push_back(abaixo);
            weights.push_back(get_edge(in, v, abaixo));
            offset_count++;
        }

        if(v_j < in->cols - 1){
            int direita = v + 1;
            src_indices.push_back(direita);
            weights.push_back(get_edge(in, v, direita));
            offset_count++;
        }

        if (v_j > 0) {
            int esquerda = v - 1;
            src_indices.push_back(esquerda);
            weights.push_back(get_edge(in, v, esquerda));
            offset_count++;
        }
        
        dest_offsets.push_back(offset_count);
    }
    
    inf.source_indices_h = (int*) malloc(src_indices.size()*sizeof(int));
    for (int i = 0; i < src_indices.size(); i++){
        inf.source_indices_h[i] = src_indices[i];
    }

    inf.weights_h = (float*)malloc(weights.size()*sizeof(float));
    for (int i = 0; i < weights.size(); i++){
        inf.weights_h[i] = weights[i];
    }

    inf.destination_offsets_h = (int*) malloc(dest_offsets.size()*sizeof(int));
    for (int i = 0; i < dest_offsets.size(); i++){
        inf.destination_offsets_h[i] = dest_offsets[i];
    }

    return inf;
}

int GPUSSSP(float *weights_h, int *destination_offsets_h, int *source_indices_h, const size_t n, const size_t nnz, int source_seed, float *sssp_h) {
    const size_t vertex_numsets = 2, edge_numsets = 1;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_h;
    vertex_dimT[0] = CUDA_R_32F;
    vertex_dimT[1] = CUDA_R_32F;

    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    int source_vert = source_seed; //source_seed
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));//retorna o sssp de todos os vertices ao source seed
    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_h, 0));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    
    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));
    
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso: segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }

    std::string path(argv[1]);
    std::string path_output(argv[2]);

    imagem *in = read_pgm(path);
    imagem *img = new_image(in->rows, in->cols);

    int n_fg, n_bg;
    int x, y;
    
    cin >> n_fg >> n_bg;

    vector<int> seeds_fg;
    for (int i =0; i< n_fg; i++){
        cin >> x >> y;
        int seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
    }

    vector<int> seeds_bg;
    for (int i =0; i< n_bg; i++){
        cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }

    thrust::device_vector<unsigned char> V1_d(in->pixels, in->pixels + in->total_size );
    thrust::device_vector<unsigned char> V2_d(img->pixels, img->pixels + img->total_size );

    dim3 dimGrid (ceil(img->rows/16), ceil(img->cols/16),1);
    dim3 dimBlock(16, 16, 1);

    edge<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(V1_d.data()), thrust::raw_pointer_cast(V2_d.data()), 0, img->rows, 0, img->cols);

    thrust::host_vector<unsigned char> V3_h(V2_d);
    for(int i = 0; i != V3_h.size(); i++) {
        img->pixels[i] = V3_h[i];
    }

    write_pgm(img, "edge.pgm");

    info inf = imgInfo(img, seeds_fg, seeds_bg);

    float * sssp_fg = (float*)malloc(inf.n*sizeof(float));
    GPUSSSP(inf.weights_h, inf.destination_offsets_h, inf.source_indices_h, inf.n, inf.nnz, img->total_size, sssp_fg);

    info inf2 = imgInfo(img, seeds_fg, seeds_bg);
    float * sssp_bg = (float*)malloc(inf.n*sizeof(float));
    GPUSSSP(inf2.weights_h, inf2.destination_offsets_h, inf2.source_indices_h, inf2.n, inf2.nnz, img->total_size+1, sssp_bg);
    // GPUSSSP(inf.weights_h, inf.destination_offsets_h, inf.source_indices_h, inf.n, inf.nnz, img->total_size, sssp_bg);

    imagem *saida = new_image(img->rows, img->cols);
    for (int k = 0; k < saida->total_size; k++) {
        if (sssp_fg[k] > sssp_bg[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    write_pgm(saida, path_output);    
    return 0;
}