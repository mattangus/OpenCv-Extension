# RM_ANTI_ALIAS_GPU 

cmake_minimum_required(VERSION 3.0)

project(PixInf)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set (CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")

set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
set(CUDA_PROPAGATE_HOST_FLAGS ON)

add_subdirectory(external/ecuda)

set(MAIN_SRC
	"src/main.cu"
	"src/ops.cu"
)
include_directories("include")
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

cuda_add_executable(PixInf ${MAIN_SRC})
target_link_libraries(PixInf ${OpenCV_LIBS} ${CUDA_LIBRARIES} stdc++fs)