#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include "imagem.h"
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "imagem.h"

#define MAX(y,x) (y>x?y:x)
#define MIN(y,x) (y<x?y:x)

typedef std::pair<double, int> custo_caminho;
typedef std::pair<double *, int *> result_sssp;

using namespace std;

struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};


result_sssp SSSP(imagem *img, int source) {
    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    result_sssp res(custos, predecessor);
    
    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }

    Q.push(custo_caminho(0.0, source));
    predecessor[source] = source;
    custos[source] = 0.0;

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int vertex_i = vertex / img->cols;
        int vertex_j = vertex % img->cols;
        
        if (vertex_i > 0) {
            int acima = vertex - img->cols;
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        if (vertex_i < img->rows - 1) {
            int abaixo = vertex + img->cols;
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        if (vertex_j < img->cols - 1) {
            int direita = vertex + 1;
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        if (vertex_j > 0) {
            int esquerda = vertex - 1;
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }
    
    delete[] analisado;
    
    return res;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);

    imagem *img = read_pgm(path);
    
    int n_fg, n_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;

    cout << n_fg << '\n';
    cout << n_bg << '\n';

    std::vector<int> seeds_fg;

    
    for (int i =0; i< n_fg; i++){
        std::cin >> x >> y;
        cout << x << y << '\n';
        int seed_fg = y * img->cols + x;
        cout << seed_fg << '\n';
        seeds_fg.push_back(seed_fg);
    }

    std::vector<int> seeds_bg;
    for (int i =0; i< n_bg; i++){
        std::cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }

    imagem *saida = new_image(img->rows, img->cols);

    result_sssp fg_final = SSSP(img, seeds_fg[0]);
    for (int i=1; i< seeds_fg.size(); i++){
        result_sssp fg = SSSP(img, seeds_fg[i]);
        for (int k = 0; k < saida->total_size; k++) {
            if(fg_final.first[k]>fg.first[k]){
                fg_final.first[k]=fg.first[k];
            }
        }
    }

    result_sssp bg_final = SSSP(img, seeds_bg[0]);
    for (int i=1; i< seeds_bg.size(); i++){
        result_sssp bg = SSSP(img, seeds_bg[i]);
        for (int k = 0; k < saida->total_size; k++) {
            if(bg_final.first[k]>bg.first[k]){
                bg_final.first[k]=bg.first[k];
            }
        }
    }

    for (int k = 0; k < saida->total_size; k++) {
        if (fg_final.first[k] > bg_final.first[k]) {
            saida->pixels[k] = 255;
        } else {
            saida->pixels[k] = 0;
        }
    }
        
    write_pgm(saida, path_output);    
    return 0;
}
