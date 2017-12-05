#!/bin/sh

# load modules

module load python/2.7.13
# the following mpi version supports fork()
module load mpi/gcc-6.3.0-openmpi-1.10.6-torque4-testing

# setup virtual environment
virtualenv python_env
source ./python_env/bin/activate

# install needed packages and ignore any cached packages
pip --no-cache-dir install -r ./requirements.txt

# may be needed in order to run imageio
python -c "import imageio; imageio.plugins.ffmpeg.download()"
