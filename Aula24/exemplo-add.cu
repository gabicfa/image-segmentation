// Exemplo para o curso de Super Computacao
// Criado por: Luciano P. Soares
// Modificado por: Igor Montagner

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <cmath>


/* Rotina para somar dois vetores na GPU */ 
__global__ void add(double *a, double *b, double *c, int N) {
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) { 
        c[i] = a[i] + b[i];
    }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {
    int n = 1<<23;
    int blocksize = 256;
    
    thrust::host_vector<double> A(n), B(n);
    for (int i = 0; i < n; i++) {
        A[i] = (double)i;
        B[i] = (double)n-i;
    }
    
    thrust::device_vector<double> A_d(A), B_d(B), C_d(n);
    add<<<ceil((double) n/blocksize),blocksize>>>(thrust::raw_pointer_cast(A_d.data()),
                                         thrust::raw_pointer_cast(B_d.data()),
                                         thrust::raw_pointer_cast(C_d.data()), 
                                         n
                                         );
    
    thrust::host_vector<double> C(C_d);
    
    for(int i=0;i<n;i++) {
        if(!(i%(n/8))) {
            printf("a[%d] + b[%d] = c[%d] => ",i,i,i);
            printf("%6.1f + %6.1f = %6.1f\n",A[i],B[i],C[i]);
        }
    }
    
    return 0;
}
