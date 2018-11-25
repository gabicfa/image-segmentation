cmake_minimum_required(VERSION 2.6)
project(segmentacao_sequencial)

set (CMAKE_CXX_STANDARD 11)

add_executable(segmentacao_sequencial main.cpp imagem.cpp)

install(TARGETS segmentacao_sequencial RUNTIME DESTINATION bin)
