sudo apt-get -qq update
sudo apt-get -qq --yes --force-yes install libblas-dev liblapacke liblapacke-dev
cd && git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror
git checkout branches/3.3
mkdir build && cd build
cmake ..
sudo make install
sudo ldconfig
cd $CI_HOME
