export DOWNLOAD_PATH=logs_wplan

[ ! -d ${DOWNLOAD_PATH} ] && mkdir ${DOWNLOAD_PATH}

wget https://www.dropbox.com/s/6q5tsgl2kxriimh/adaca-d4rl-logs-wplan.zip?dl=1 -O dropbox_models.zip
unzip dropbox_models.zip -d ${DOWNLOAD_PATH}
rm dropbox_models.zip
