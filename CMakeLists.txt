cmake_minimum_required (VERSION 2.6)
project (CLIENT)


# List the include paths
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/mash-simulator/dependencies"
                    "${CMAKE_CURRENT_SOURCE_DIR}/mash-simulator/dependencies/include"
		    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# List the lib paths
LINK_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/mash-simulator/build/bin")

#specify target and source
add_executable(RLAgent RLAgent.cpp)


#link target to external libraries
TARGET_LINK_LIBRARIES(RLAgent mash-utils mash-network mash-appserver -lzmq)
