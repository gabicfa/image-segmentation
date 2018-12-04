cmake_minimum_required(VERSION 2.6)
project(segmentacao_sequencial)

set (CMAKE_CXX_STANDARD 11)

add_executable(segmentacao_sequencial mainSeqMult.cpp imagem.cpp)

add_executable(segmentacao_gpu mainGPUMult.cpp imagem.cpp)
target_link_libraries(segmentacao_gpu "-lnvgraph")

install(TARGETS segmentacao_sequencial RUNTIME DESTINATION bin)
