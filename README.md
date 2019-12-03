# Data download

1. install Google cloud sdk (272.0.0)
from https://cloud.google.com/sdk/docs/

2. Copy Data

```shell script

gsutil -m cp -r gs://broad-cho-ukb/2019-11-19 ./data/raw/

```
 

# PySyft setting

```shell script
gsutil ls -L -b gs://broad-cho-ukb/2019-11-19/
```
pip install --upgrade google-cloud-storage


# keras setting
brew install graphviz