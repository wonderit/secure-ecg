# Data download

1. install Google cloud sdk (272.0.0)
from https://cloud.google.com/sdk/docs/

2. Copy Data

```shell script

gsutil -m cp -r gs://broad-cho-ukb/2019-11-19 ./data/raw/

```

# Conda Setting

```shell script

curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

conda create -n ecg python=3.7
conda activate ecg
pip install -r requirements.txt
```

 

# PySyft setting

```shell script
gsutil ls -L -b gs://broad-cho-ukb/2019-11-19/
```
pip install --upgrade google-cloud-storage



# keras setting
brew install graphviz