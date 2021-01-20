import numpy as np

def to_classes_dim_label(x, y, NO_CLASSES):
    '''
    input x shape is (no_images, height, width)
    input y shape is (no_images,)
    output: y_res (no_images, height, width, NO_CLASSES) y_res[b, h, w, i] = 1 if the image b is digit i, and only at the positions h, w where it is alive
    '''
    y_res = np.zeros(list(x.shape) + [NO_CLASSES])
    y_expanded = np.broadcast_to(y, x.T.shape).T # broadcast y to match x shape:
    for i in range(x.shape[0]):
        y_res[i, :, :, y[i]] = np.ones((x.shape[1], x.shape[2]))
    return y_res.astype(np.float32)

def build_dataset(folder, target_classes, H, W, SEED_DATASET, ds_str):
    with open(folder + '/dataset/emotion_dataset.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    original_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    class_labels = [original_labels.index(target_classes[i]) for i in range(len(target_classes))]

    X_final = np.zeros((0, H, W))
    y_final = np.zeros((0,))
    for new_class, class_label in enumerate(class_labels):
        idx = np.where(y == class_label)[0]
        len_class = idx.shape[0]
        X_final = np.concatenate((X_final, X[idx, :, :, 0]))
        y_final = np.concatenate((y_final, new_class * np.ones(len_class)))
    X_final /= 255
    X_final = X_final.astype(np.float32)
    y_final = y_final.astype(np.int)
    size_ds = y_final.shape[0]
    np.random.seed(SEED_DATASET)
    rnd_idx = np.random.permutation(size_ds)
    x_train, x_test = np.split(X_final[rnd_idx, :, :], [int(size_ds*0.8)])
    y_train, y_test = np.split(y_final[rnd_idx], [int(size_ds*0.8)])
    return x_train, x_test, y_train, y_test