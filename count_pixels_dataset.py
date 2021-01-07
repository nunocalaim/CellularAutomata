import numpy as np
import random

def class_indice_f(no_pixels, limits_classes):
    '''
    find the indice of the class of the no_pixels
    '''
    NO_CLASSES = len(limits_classes) + 1
    indices = np.where(no_pixels <= np.array(limits_classes))[0]
    if indices.shape[0] == 0:
        return NO_CLASSES - 1
    else:
        return indices[0]

def to_classes_dim_label(x, y, limits_classes):
    '''
    input x shape is (no_images, height, width)
    input y shape is (no_images,)
    output: y_res (no_images, height, width, NO_CLASSES) y_res[b, h, w, i] = 1 if the image b is digit i, and only at the positions h, w where it is alive
    '''
    NO_CLASSES = len(limits_classes) + 1
    y_res = np.zeros(list(x.shape) + [NO_CLASSES])
    y_expanded = np.broadcast_to(y, x.T.shape).T # broadcast y to match x shape:
    for i in range(x.shape[0]):
        class_indice = class_indice_f(y[i], limits_classes)
        y_res[i, :, :, class_indice] = x[i, :, :]
    return y_res.astype(np.float32)

def new_image(no_pixels, H, W):
    X = np.zeros((H, W), np.float32)
    pixels_placed = 1
    first_pixel = (int(H/2), int(W/2)) # we insert the first pixel at the center of the image
    positions_colored = [first_pixel]
    X[first_pixel[0], first_pixel[1]] = 1
    while pixels_placed < no_pixels:
        base_pixel = random.choice(positions_colored)
        x, y = base_pixel[0], base_pixel[1]
        if np.random.rand() < 0.5:
            if np.random.rand() < 0.5:
                dx = 1
                dy = 0
            else:
                dx = -1
                dy = 0
        else:
            if np.random.rand() < 0.5:
                dx = 0
                dy = 1
            else:
                dx = 0
                dy = -1
        test_pixel = (min(max(0, x + dx), H-1), min(max(0, y + dy), W-1))
        if test_pixel not in positions_colored:
            pixels_placed += 1
            positions_colored.append(test_pixel)
            X[test_pixel[0], test_pixel[1]] = 1
    return X        

def build_dataset(size_ds, H, W, limits_classes, max_pixels=20, split_ds=0.8, boundary_p=0.5):
    '''
    This function creates a balanced dataset, with special incidence for boundary conditions
    '''
    X = np.zeros((size_ds, H, W), np.float32)
    Y = np.zeros((size_ds), np.float32)
    
    for i in range(size_ds):
        if np.random.rand() < boundary_p:
            no_pixels = np.random.choice(limits_classes)
            if np.random.rand() < 0.5:
                no_pixels += 1
        else:
            no_classes = len(limits_classes) + 1
            class_i = np.random.choice(no_classes)
            if class_i == 0:
                no_pixels = np.random.randint(limits_classes[0]) + 1
            elif class_i == no_classes - 1:
                no_pixels = np.random.randint(limits_classes[-1], max_pixels) + 1
            else:
                no_pixels = np.random.randint(limits_classes[class_i-1], limits_classes[class_i]) + 1
        X[i, :, :] = new_image(no_pixels, H, W)
        Y[i] = no_pixels
    x_train, x_test = np.split(X, [int(size_ds*split_ds)])
    y_train, y_test = np.split(Y, [int(size_ds*split_ds)])
    return x_train, x_test, y_train, y_test
