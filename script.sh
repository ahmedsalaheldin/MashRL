#intilize and update submodules
git submodule init
git submodule update

#copy modified files to relevant directories
cp server/{MyServer.cpp,CMakeLists.txt,main.cpp} mash-simulator/src/
cp server/MyServer.h mash-simulator/include/
#replace modified Deep RL script

#intilize and update submodules for the mash-simulator
cd mash-simulator ; git submodule init ; git submodule update 

#build mash-simulator
mkdir build
cd build

cmake ..
make

cd .. ; cd ..

#build our client
mkdir build
cd build

cmake ..
make
