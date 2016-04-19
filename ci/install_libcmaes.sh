cd /usr/src/gtest
sudo mkdir build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
cd && git clone https://github.com/beniz/libcmaes.git
cd libcmaes
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
cd && cd $CI_HOME
