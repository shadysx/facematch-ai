import sys
import site

def check_real_dlib():
    try:
        import dlib
        print(f"dlib trouvé: {dlib.__file__}")
        print(f"Version: {dlib.__version__}")
    except ImportError:
        print("dlib n'est pas installé")

    print("\nEnvironnement Python:")
    print(f"Python: {sys.executable}")
    print("\nSite-packages:")
    for path in site.getsitepackages():
        print(path)

check_real_dlib()