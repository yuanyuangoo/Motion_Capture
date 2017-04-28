cmake_minimum_required(VERSION 2.6)
project(TRAIN)


find_package(Caffe REQUIRED)

if (Caffe_FOUND)
	add_executable(train main.cpp)
	include_directories(${Caffe_INCLUDE_DIRS})
	add_definitions(${Caffe_DEFINITIONS})
	target_link_libraries(train ${Caffe_LIBRARIES})
else()
	message(STATUS "caffe is not found")
endif()

