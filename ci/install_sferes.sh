cd && git clone https://github.com/sferes2/sferes2.git
cd sferes2
./waf configure
./waf
export SFERES_HOME=`pwd`
cd $CI_HOME
