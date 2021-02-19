import pickle
import glob
import numpy as np
from sklearn.utils import shuffle

WINDOW=15

def append(a,b):
    if b is None:
        return a
    if a is None:
        a=b
    else:
        a=np.concatenate((a,b))
    return a

def get_XY():
    path = './*.bin'
    files = glob.glob(path)
    X = None
    y = None

    for j in range(len(files)):
        f = files[j]
        with open(f, 'rb') as fp:
            signal = pickle.load(fp)
        xi = None
        for example in signal:
            data = None
            for i in range(len(example)-WINDOW):
                chunk = np.array(example[i:i+WINDOW])
                chunk = chunk[None,:]
                data = append(data, chunk)
            if(len(example) < WINDOW):
                chunk = np.array(example)
                rem = WINDOW - len(chunk)
                data = np.concatenate((chunk, np.random.choice(chunk, rem)))
                data = data[None,:]
            xi = append(xi, data)
        yi = np.ones(xi.shape[0]) * j
        X = append(X, xi)
        y = append(y, yi)

    return shuffle(X, y)
