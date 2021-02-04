import numpy as np

def y2yhot(x, y, NO_CLASSES):
    '''
    input x shape is (no_images, height, width)
    input y shape is (no_images,)
    output: yhot (no_images, height, width, NO_CLASSES) y_res[b, h, w, i] = 1 if the image b belongs to class i
    '''
    yhot = np.zeros(list(x.shape) + [NO_CLASSES])
    for i in range(x.shape[0]):
        yhot[i, :, :, y[i]] = np.ones((x.shape[1], x.shape[2]))
    return yhot.astype(np.float32)

def build_dataset(size_ds, obj1, obj2, WW, seed=1, type_ds='xor'):
    H, W = obj1.shape
    np.random.seed(seed)
    X = np.zeros((size_ds, H, 2 * W + WW), np.float32)
    Y = np.zeros((size_ds), np.int)

    for i in range(size_ds):
        if type_ds == 'xor':
            if np.random.rand() < 0.5:
                X[i, :, :W] = obj1
                j = 0
            else:
                X[i, :, :W] = obj2
                j = 1
            if np.random.rand() < 0.5:
                X[i, :, -W:] = obj1
                Y[i] = 1 - j
            else:
                X[i, :, -W:] = obj2
                Y[i] = j
        elif type_ds == 'simple':
            if np.random.rand() < 0.5:
                X[i, :, :W] = obj1
                X[i, :, -W:] = obj1
                Y[i] = 0
            else:
                X[i, :, :W] = obj2
                X[i, :, -W:] = obj2
                Y[i] = 1
    x_train, x_test = np.split(X, [int(size_ds*0.8)])
    y_train, y_test = np.split(Y, [int(size_ds*0.8)])
    return x_train, x_test, y_train, y_test

def embed_objs(t_obj1, t_obj2, H):
    obj1 = np.zeros((H, H))
    h1, w1 = t_obj1.shape
    h_i, w_i = int((H - h1) / 2), int((H - w1) / 2)
    obj1[h_i:(h_i + h1), w_i:(w_i + w1)] = t_obj1
    obj2 = np.zeros((H, H))
    h2, w2 = t_obj2.shape
    h_i, w_i = int((H - h2) / 2), int((H - w2) / 2)
    obj2[h_i:(h_i + h2), w_i:(w_i + w2)] = t_obj2
    return obj1, obj2