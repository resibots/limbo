cd $CI_HOME && git clone https://github.com/sferes2/sferes2.git
cd sferes2
./waf configure
./waf
cd $CI_HOME
