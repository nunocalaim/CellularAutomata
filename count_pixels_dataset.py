import numpy as np

def get_next_pixel(base_pixel, positions_colored):
    x, y = base_pixel[0], base_pixel[1]
    keep_going = 1
    j = 0
    while keep_going and j < 10:
        j += 1
        x_or_y = np.random.rand() < 0.5
        pos_or_neg = np.random.rand() < 0.5
        if x_or_y:
            if pos_or_neg:
                dx = 1
                dy = 0
            else:
                dx = -1
                dy = 0
        else:
            if pos_or_neg:
                dx = 0
                dy = 1
            else:
                dx = 0
                dy = -1
        nx = min(max(0, x + dx), 9)
        ny = min(max(0, y + dy), 9)
        test_pixel = [nx, ny]
        if test_pixel in positions_colored:
            pass
        else:
            return 1, test_pixel
    return 0, [0, 0]

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


def build_dataset(size_ds, H, W, max_pixels=20, split_ds=0.8):
    X = np.zeros((size_ds, H, W), np.float32)
    Y = np.zeros((size_ds), np.float32)
    
    for i in range(size_ds):
        no_pixels = np.random.randint(max_pixels)
        pixels_placed = 1
        first_pixel = [4, 4]
        positions_colored = [first_pixel]
        X[i, first_pixel[0], first_pixel[1]] = 1
        for j in range(no_pixels - 1):
            base_pixel = positions_colored[np.random.choice(len(positions_colored))]
            sucess, next_pixel = get_next_pixel(base_pixel, positions_colored)
            if sucess:
                pixels_placed += 1
                positions_colored.append(next_pixel)
                X[i, next_pixel[0], next_pixel[1]] = 1
        Y[i] = pixels_placed
        x_train, x_test = np.split(X, [int(size_ds*split_ds)])
        y_train, y_test = np.split(Y, [int(size_ds*split_ds)])

    return x_train, x_test, y_train, y_test