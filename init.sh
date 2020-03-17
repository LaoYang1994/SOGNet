# Download pretrained models
if [ ! -f data/pretrained_model/resnet-101-caffe.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/resnet-101-caffe.pth -o data/pretrained_model/resnet-101-caffe.pth
fi
if [ ! -f data/pretrained_model/resnet-50-caffe.pth ]; then
    curl http://www.yuwenxiong.com/pretrained_model/resnet-50-caffe.pth -o data/pretrained_model/resnet-50-caffe.pth
fi

# Install essential python packages

pip install pyyaml pycocotools

# Download panopticapi devkit
git clone https://github.com/cocodataset/panopticapi tools/panopticapi

# Build essential operators

# build cython modules
cd sognet/bbox; python setup.py build_ext --inplace
cd ../rpn; python setup.py build_ext --inplace
cd ../nms; python setup.py build_ext --inplace
# build operators
cd ../operators
python build_deform_conv.py build_ext --inplace 
python build_roialign.py build_ext --inplace

