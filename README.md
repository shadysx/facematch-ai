



# Usage example
```bash
python face_recognition.py "models/shape_predictor_5_face_landmarks.dat" "models/dlib_face_recognition_resnet_model_v1.dat" "faces/500_humans" "faces/unknown"
```

## Make sure to use gcc 8 and read every logs while installing!
## Make sur nvidia drivers are installed
## https://ubuntu.com/server/docs/nvidia-drivers-installation

```bash
$ conda create -n facematch_gpu python=3.8 cmake ipython
$ conda activate facematch_gpu
$ conda install -c conda-forge gcc=8.5.0
$ conda install -c conda-forge cudatoolkit=11.8.0
$ conda install -c conda-forge cudnn=8.9.2.26
$ conda install -c conda-forge libxcrypt
$ t clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ 
$ cmake --build .
$ cd ..
$ python setup.py install --set DLIB_USE_CUDA=1
```

### pornstar database: https://www.thumbnailseries.com/pornstars/

### API usage example
```bash
curl -X POST -F "file=@unknown_face.jpg" http://localhost:8000/compare-face
curl -X POST -F "file=@unknown_face.jpg" http://192.168.0.100:8000/compare-face
```

### Conda env names:
```bash
- Ubuntu Desktop: facematch_gpu
- MacOs: facematch_gpu
- Ubuntu Server: dlib 
```
