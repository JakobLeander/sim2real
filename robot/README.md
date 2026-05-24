# remember to enable I2C in raspi-config
# Tested with python 3.11.2. Python3.13 causes issues with some of adafruit packages
# If you are on newest raspberry pi image it comes with python 3.13. You have to install python3.11.2 as well and use that for making the venv

# To generate stepper pulses use pigpiod
#wget https://github.com/joan2937/pigpio/archive/master.zip
#unzip master.zip
#cd pigpio-master
#make
#sudo make install
# may also need to indstall distutils since pypigio expects it when building