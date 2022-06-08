export DOWNLOAD_PATH=logs

[ ! -d ${DOWNLOAD_PATH} ] && mkdir ${DOWNLOAD_PATH}

wget https://www.dropbox.com/s/1u6j0ybe0l4vh1l/adaca-d4rl-logs.zip?dl=1 -O dropbox_models.zip
unzip dropbox_models.zip -d ${DOWNLOAD_PATH}
rm dropbox_models.zip
