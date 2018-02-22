
# !/bin/bash

PATH_TO=$1

MYCUSTOMTAB='   '
NC='\033[0m'
Cyan='\033[0;36m'
Orange='\033[0;33m'
Purple='\033[0;35m'

echo ======================================================================
echo This shell is to replace tf-faster-rcnn library with new python scripts...
echo to allow training and testing with 4-channel images on the library...
echo ======================================================================

echo -e ${Purple}the following files will be replaced:
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}experiments/scripts/${Cyan}test_faster_rcnn.sh"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}experiments/scripts/${Cyan}train_faster_rcnn.sh"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}tools/${Cyan}tf_faster_rcnn_predict.py"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/datasets/${Cyan}berryCalyx.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/datasets/${Cyan}berryCalyx_eval.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/datasets/${Cyan}factory.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/datasets/${Cyan}imdb.py"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/model/${Cyan}config.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/model/${Cyan}test.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/model/${Cyan}train_val.py"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/nets/${Cyan}network.py"
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/nets/${Cyan}vgg16.py"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/roi_data_layer/${Cyan}minibatch.py"
echo ----------
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/utils/${Cyan}blob.py"

echo -e ${Purple}the following files will be added:
echo -e "${MYCUSTOMTAB}${NC=}./tf-faster-rcnn/${Orange}lib/utils/${Cyan}cython_nms.so"


sudo rm /opt/tf-faster-rcnn/experiments/scripts/test_faster_rcnn.sh
sudo rm /opt/tf-faster-rcnn/experiments/scripts/train_faster_rcnn.sh
sudo ln -s ${PATH_TO}/experiments/scripts/test_faster_rcnn.sh  /opt/tf-faster-rcnn/experiments/scripts/
sudo ln -s ${PATH_TO}/experiments/scripts/train_faster_rcnn.sh  /opt/tf-faster-rcnn/experiments/scripts/

sudo rm /opt/tf-faster-rcnn/tools/tf_faster_rcnn_predict.py
sudo ln -s ${PATH_TO}/tools/tf_faster_rcnn_predict.py  /opt/tf-faster-rcnn/tools/

sudo rm /opt/tf-faster-rcnn/lib/datasets/berryCalyx.py
sudo rm /opt/tf-faster-rcnn/lib/datasets/berryCalyx_eval.py
sudo rm /opt/tf-faster-rcnn/lib/datasets/factory.py
sudo rm /opt/tf-faster-rcnn/lib/datasets/imdb.py
sudo ln -s ${PATH_TO}/lib/datasets/berryCalyx.py  /opt/tf-faster-rcnn/lib/datasets/
sudo ln -s ${PATH_TO}/lib/datasets/berryCalyx_eval.py  /opt/tf-faster-rcnn/lib/datasets/
sudo ln -s ${PATH_TO}/lib/datasets/factory.py  /opt/tf-faster-rcnn/lib/datasets/
sudo ln -s ${PATH_TO}/lib/datasets/imdb.py  /opt/tf-faster-rcnn/lib/datasets/

sudo rm /opt/tf-faster-rcnn/lib/model/config.py
sudo rm /opt/tf-faster-rcnn/lib/model/test.py
sudo rm /opt/tf-faster-rcnn/lib/model/train_val.py
sudo ln -s ${PATH_TO}/lib/model/config.py  /opt/tf-faster-rcnn/lib/model
sudo ln -s ${PATH_TO}/lib/model/test.py  /opt/tf-faster-rcnn/lib/model
sudo ln -s ${PATH_TO}/lib/model/train_val.py  /opt/tf-faster-rcnn/lib/model

sudo rm /opt/tf-faster-rcnn/lib/nets/network.py
sudo rm /opt/tf-faster-rcnn/lib/nets/vgg16.py
sudo ln -s ${PATH_TO}/lib/nets/network.py  /opt/tf-faster-rcnn/lib/nets
sudo ln -s ${PATH_TO}/lib/nets/vgg16.py  /opt/tf-faster-rcnn/lib/nets

sudo rm /opt/tf-faster-rcnn/lib/roi_data_layer/minibatch.py
sudo ln -s ${PATH_TO}/lib/roi_data_layer/minibatch.py  /opt/tf-faster-rcnn/lib/roi_data_layer

sudo rm /opt/tf-faster-rcnn/lib/utils/blob.py
sudo ln -s ${PATH_TO}/lib/utils/blob.py  /opt/tf-faster-rcnn/lib/utils

sudo ln -s ${PATH_TO}/lib/utils/cython_nms.so  /opt/tf-faster-rcnn/lib/utils