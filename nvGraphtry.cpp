#include <stdio.h>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#include "imagem.h"

void check_status(nvgraphStatus_t status){
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

int GpuSSSP(float *weights_h, int *destination_offsets_h, int *source_indices_h, const size_t n, const size_t nnz, int source_seed, float *sssp_h) {
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

struct info{
    size_t n;
    size_t nnz;
    // float * weights_h;
    // int * destination_offsets_h;
    // int * source_indices_h;
    // int source_seed;
};

info imgInfo(imagem *in){
    info inf = {};
    inf.n = in->total_size;
    inf.nnz = 2*((in->cols-1)*in->rows+(in->rows-1)*in->cols);
    return inf;
}

int main(int argc, char **argv) {
    
    imagem *img = read_pgm("small.pgm");
    info inf = imgInfo(img);
    std::cout << inf.n << '\n';
    std::cout << inf.nnz << '\n';
    return 0;
}