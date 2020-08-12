sudo apt-get -qq update
sudo apt-get -qq --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev
cd /usr/src/gtest
sudo mkdir build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
cd && git clone https://github.com/resibots/libcmaes.git
cd libcmaes
git checkout fix_flags_native
mkdir build
cd build
cmake -E env CXXFLAGS="-w" cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF -DBUILD_PYTHON=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DLINK_PYTHON=OFF ..
make CXXFLAGS="-w"
sudo make install
sudo ldconfig
cd $CI_HOME
