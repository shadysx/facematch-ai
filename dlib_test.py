import dlib
print(f"CUDA enabled: {dlib.DLIB_USE_CUDA}")
print(f"CUDA devices: {dlib.cuda.get_num_devices()}")