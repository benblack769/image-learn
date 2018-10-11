import scipy
import scipy.ndimage
import os

def read_image(fname):
    data = scipy.ndimage.imread(fname)
    data = data.astype(np.float32)/255.0
    return data

def get_paths(folder):
    filenames = os.listdir(folder)
    jpg_filenames = [fname for fname in filenames if ".jpg" in fname]
    paths = [os.path.join(folder,fname) for fname in jpg_filenames]
    return paths

def get_images():
    folder = "example_images/"
    paths = get_paths(folder)
    images = [read_image(path) for path in paths]
    np_images = np.stack(images)
    return np_images

def identity_train()
