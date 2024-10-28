# EZSwitch

EZSwitch

## Install Requirements
EZSwitch requires [Python 3.10](https://www.python.org/downloads/) or greater.
To install needed library you can use:
```
pip install -r requirements.txt
```

## GIZA

We use GIZA to get word alignment between parallel sentence. And we need to install this,
We have included the repo in the submodules.

### Giza-py

To install Giza-py, clone the repo and install pip dependencies:

```
cd alignment/giza-py
pip install -r requirements.txt
```

### MGIZA++

In order to install MGIZA++ on Linux/macOS, follow these steps:

1. Download the [Boost C++ library](https://www.boost.org/) and unzip it.
2. Build Boost:

```
cd <boost_dir>
./bootstrap.sh --prefix=./build --with-libraries=thread,system
./b2 install
```


4. Build MGIZA++ (CMake is required):

```
cd alignment/mgizapp
cmake -DBOOST_ROOT=<boost_dir>/build -DBoost_USE_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX=<giza-py_dir>/.bin .
make
make install
```