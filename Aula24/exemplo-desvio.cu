#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

/* Rotina para somar dois vetores na GPU */ 
__global__ void add(double *a, double *b, int N, double media) {
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) { 
        b[i] = (pow((a[i] - media),2));
    }
}

/* Programa cria dois vetores e soma eles em GPU */
int main() {
    ifstream inFile;
    int number_of_lines=0;
    string line;
    inFile.open("stocks-google.txt");
    while (getline(inFile, line)){
        number_of_lines+=1;
    }
    inFile.close();

    // int n = 1<<23;
    int n = number_of_lines;

    thrust::host_vector<float> v1(number_of_lines,0);
    int i=0;
    inFile.open("stocks-google.txt");
    while (getline(inFile, line)){
        v1[i] = stod(line);
        i++;
    }
    
    thrust::device_vector<double> V1_d(v1), V2_d(n);

    double sum = thrust::reduce(V1_d.begin(), V1_d.end(), 0, thrust::plus<int>());

    cout << sum << '\n';

    double media = sum/number_of_lines;

    cout << media << '\n';

    int blocksize = 256;

    add<<<ceil((double) n/blocksize),blocksize>>>(thrust::raw_pointer_cast(V1_d.data()),
                                         thrust::raw_pointer_cast(V2_d.data()),
                                         n,
                                         media
                                         );
    
    double sum2 = thrust::reduce(V2_d.begin(), V2_d.end(), 0, thrust::plus<int>());

    cout << sum2/n << '\n';
    
    return 0;
}
