sudo apt-get -qq update
cd && hg clone https://bitbucket.org/eigen/eigen
cd eigen
hg up 3.3
mkdir build && cd build
cmake ..
sudo make install
sudo ldconfig
cd $CI_HOME
