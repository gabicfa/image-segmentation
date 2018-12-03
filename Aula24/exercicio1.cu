#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>

using namespace std::chrono;
using namespace std;

int main() {
    high_resolution_clock::time_point t1, t2;
    duration<double> tempoCriacao;
    duration<double> tempoCopia;

    ifstream inFile;
    int number_of_lines=0;
    string line;
    inFile.open("stocks-google.txt");
    while (getline(inFile, line)){
        number_of_lines+=1;
    }
    inFile.close();
    
    thrust::host_vector<float> v1(number_of_lines,0);
    int i=0;
    inFile.open("stocks-google.txt");
    while (getline(inFile, line)){
        v1[i] = stof(line);
        i++;
    }

    t1 = high_resolution_clock::now();
    thrust::device_vector<float> v2(number_of_lines,0);
    t2 = high_resolution_clock::now();
    tempoCriacao = duration_cast<duration<double> >(t2 - t1);
    cout << "tempo de alocacao: " << tempoCriacao.count() << '\n';

    t1 = high_resolution_clock::now();
    v2=v1;
    t2 = high_resolution_clock::now();
    tempoCopia = duration_cast<duration<double> >(t2 - t1);
    cout << "tempo de copia: " << tempoCopia.count() << '\n';

    inFile.close();
}