1 Install Linux system, suggest CentOS Linux 8.5.2111 with ARM64 (aarch64) and GPU Nvidia RTX 3080 or 3090. If not use suggestion hardware and OS may occur problems.

2 Install the library:
pip install -r requirements.txt

3 Inference the images or video with command line:

#For just localization
python detectNew.py --img-size 640 --device 0  --weight weights\groc60.pt --source 'your_path\inference\videos'

#For localization and tracking using ONNX model and SORT respectively
python detectNewwithONNX.py --img-size 640 --device 0  --weight weights\groc60.pt --source 'your_path\inference\videos'


If this code needs to be tested on images then simply give path of the test set instead of the video

4 Train the images: (Not sure if it works correctly)

python train.py --data yourdatafolder\data.yaml --weights ''
