import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob

def y2yhot(x, y, NO_CLASSES, task):
    '''
    input x shape is (no_images, height, width)
    input y shape is (no_images,)
    output: yhot (no_images, height, width, NO_CLASSES) y_res[b, h, w, i] = 1 if the image b belongs to class i
    '''
    yhot = np.zeros(list(x.shape) + [NO_CLASSES])
    for i in range(x.shape[0]):
        if task == 'count_pixels':
            yhot[i, :, :, y[i]] = x[i, :, :]
        else:
            yhot[i, :, :, y[i]] = np.ones((x.shape[1], x.shape[2]))
    return yhot.astype(np.float32)



def get_dataset(task, opt, RebuildDatasetQ, folder):

    ds_str = task
    SEED_DATASET = opt['SEED_DATASET']
    if task == 'emotions':
        classes_labels, DataBalancingQ, DataAugmentationQ = opt['classes_labels'], opt['DataBalancingQ'], opt['DataAugmentationQ']
        H, W = 48, 48
        NO_CLASSES = len(classes_labels)
        tc_str = str(classes_labels).replace('[', '').replace(']', '').replace(', ', '_').replace('\'', '')
        ds_str += '_' + tc_str
        if DataBalancingQ:
            ds_str += '_Bal'
        if DataAugmentationQ:
            ds_str += '_Aug'
    elif task == 'count_pixels':
        limits_classes, H, W, SIZE_DS, MAXPIXELS, BOUNDARY_P = opt['limits_classes'], opt['H'], opt['W'], opt['SIZE_DS'], opt['MAXPIXELS'], opt['BOUNDARY_P']
        NO_CLASSES = len(limits_classes) + 1 # Number of classes that the CA must distinguish
        cb_str = str(limits_classes).replace('[', '').replace(']', '').replace(', ', '_')
        ds_str += '_ClassB' + cb_str
        if H != 10 or W != 10:
            ds_str += '_H{}_W{}'.format(H, W)
        if SIZE_DS != 100000:
            ds_str += '_{}'.format(SIZE_DS)
        if MAXPIXELS != 44:
            ds_str += '_MaxPixels{}'.format(MAXPIXELS)
        if BOUNDARY_P != 0.4:
            ds_str += '_BoundP{}'.format(int(BOUNDARY_P*100))
        classes_labels = ['pixels<={}'.format(limits_classes[0])] + ['{}<pixels<={}'.format(limits_classes[i], limits_classes[i+1]) for i in range(len(limits_classes) - 1)] + ['{}<pixels<={}'.format(limits_classes[-1], MAXPIXELS)]
    elif task == 'xor':
        classes_labels, type_ds, SIZE_DS, obj1_str, obj2_str, H, WW, WW_test = opt['classes_labels'], opt['type_ds'], opt['SIZE_DS'], opt['obj1_str'], opt['obj2_str'], opt['H'], opt['WW'], opt['WW_test']
        NO_CLASSES = len(classes_labels)
        if type_ds != 'xor':
            ds_str += type_ds
        if SIZE_DS != 10000:
            ds_str += '_Size{}'.format(SIZE_DS)
        if obj1_str != 'vertical_small':
            ds_str += '_Obj1{}'.format(obj1_str)
        if obj2_str != 'horizontal_small':
            ds_str += '_Obj2{}'.format(obj2_str)
        if H != 3:
            ds_str += '_H{}'.format(H)
        ds_str += '_Sep{}'.format(WW)
        ds_str_test = ds_str + '_Sep{}'.format(WW_test)
        # Prepare the dataset
        dict_obj = {
            'vertical_small':
            np.array([[1], [1], [1]]),
            'horizontal_small':
            np.array([[1, 1, 1]]),
        }
        t_obj1 = dict_obj[obj1_str]
        t_obj2 = dict_obj[obj2_str]
        obj1, obj2 = embed_objs(t_obj1, t_obj2, H)
        W = 2 * H + WW
        W_test = 2 * H + WW_test
        if SEED_DATASET != 1:
            ds_str_test += '_Seed{}'.format(SEED_DATASET)
    elif task == 'fruits':
        SEED_DATASET = 1
        H, W = opt['H'], opt['W']
        NO_CLASSES = 3 # Number of classes that the CA must distinguish
        if H != 30 or W != 30:
            ds_str += '_H{}_W{}'.format(H, W)
        classes_labels = ['apple', 'banana', 'orange']
    elif task == 'frozen_noise':
        H, W, img_per_class = opt['H'], opt['W'], opt['img_per_class'] 
        if img_per_class > 1:
            ds_str += '_ImgPerClass{}'.format(img_per_class)
        NO_CLASSES = 2 # Number of classes that the CA must distinguish
        if H != 50 or W != 50:
            ds_str += '_H{}_W{}'.format(H, W)
        classes_labels = ['0', '1']
    if SEED_DATASET != 1:
        ds_str += '_Seed{}'.format(SEED_DATASET)

    BuildDS = False
    if RebuildDatasetQ:
        BuildDS = True
    else:
        try:
            res = np.load(folder + '/dataset/{}/{}.npz'.format(task, ds_str))
            x_train, x_test, y_train, y_test = res['x_train'], res['x_test'], res['y_train'], res['y_test']
            if task == 'xor':
                res = np.load(folder + '/dataset/{}/{}.npz'.format(task, ds_str_test))
                x_train_ignore, x_test, y_train_ignore, y_test = res['x_train'], res['x_test'], res['y_train'], res['y_test']
        except:
            BuildDS = True
    if BuildDS:

        if task == 'count_pixels':
            x_train, x_test, y_train, y_test = build_dataset_count_pixels(SIZE_DS, H, W, limits_classes, max_pixels=MAXPIXELS, boundary_p=BOUNDARY_P, seed=SEED_DATASET)
        elif task == 'xor':
            x_train, x_test, y_train, y_test = build_dataset_xor(SIZE_DS, obj1, obj2, WW, type_ds=type_ds, seed=SEED_DATASET)
        elif task == 'emotions':
            x_train, x_test, y_train, y_test = build_dataset_emotions(folder, classes_labels, H, W, DataBalancingQ, DataAugmentationQ, seed=SEED_DATASET)
        elif task == 'fruits':
            x_train, x_test, y_train, y_test = build_dataset_fruits(folder, H, W)
        elif task == 'frozen_noise':
            x_train, x_test, y_train, y_test = build_dataset_frozen_noise(folder, H, W, img_per_class, seed=SEED_DATASET)
        print(y_train)        
        np.savez(folder + '/dataset/{}/{}.npz'.format(task, ds_str), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        if task == 'xor':
            x_train_ignore, x_test, y_train_ignore, y_test = build_dataset_xor(SIZE_DS, obj1, obj2, WW_test, type_ds=type_ds, seed=SEED_DATASET)
            np.savez(folder + '/dataset/{}/{}.npz'.format(task, ds_str_test), x_train=x_train_ignore, x_test=x_test, y_train=y_train_ignore, y_test=y_test)


    cols=7
    rows=7
    fig, ax = plt.subplots()
    DISP = np.zeros((1, cols * W + 1))
    RES = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(rows):
        disp = np.zeros((H, 1))
        for j in range(cols):
            image_idx = np.random.randint(x_train.shape[0])
            disp = np.hstack((disp, x_train[image_idx, :, :]))
            RES[i][j] = y_train[image_idx]
        DISP = np.vstack((DISP, disp))
    ax.imshow(DISP)
    print(np.array(RES))


    return x_train, x_test, y_train, y_test, ds_str, NO_CLASSES, H, W, classes_labels


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


def build_dataset_count_pixels(size_ds, H, W, limits_classes, max_pixels=20, boundary_p=0.5, seed=1):
    '''
    This function creates a balanced dataset, with special incidence for boundary conditions
    '''
    X = np.zeros((size_ds, H, W), np.float32)
    Y = np.zeros((size_ds), np.int)
    np.random.seed(seed)
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
        Y[i] = class_indice_f(no_pixels, limits_classes)
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

def build_dataset_xor(size_ds, obj1, obj2, WW, type_ds='xor', seed=1):
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


def build_dataset_emotions(folder, classes_labels, H, W, DataBalancingQ, DataAugmentationQ, seed=1):
    with open(folder + '/dataset/emotions/emotion_dataset.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)
    original_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    class_labels = [original_labels.index(classes_labels[i]) for i in range(len(classes_labels))]

    X_final = np.zeros((0, H, W))
    y_final = np.zeros((0,))
    if DataBalancingQ:
        max_class = 0
        for new_class, class_label in enumerate(class_labels):
            idx = np.where(y == class_label)[0]
            max_class = max(max_class, idx.shape[0])

    for new_class, class_label in enumerate(class_labels):

        X_sub = np.zeros((0, H, W))
        y_sub = np.zeros((0,))

        idx = np.where(y == class_label)[0]
        len_class = idx.shape[0]

        X_sub = np.concatenate((X_sub, X[idx, :, :, 0]))
        y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
        if DataAugmentationQ:
            X_sub = np.concatenate((X_sub, X[idx, :, ::-1, 0]))
            y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
            X_sub = np.concatenate((X_sub, X[idx, :, :, 0] / 2))
            y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
            X_sub = np.concatenate((X_sub, X[idx, :, ::-1, 0] / 2))
            y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
            X_sub = np.concatenate((X_sub, np.clip(X[idx, :, :, 0] * 1.2, 0, 255)))
            y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
            X_sub = np.concatenate((X_sub, np.clip(X[idx, :, ::-1, 0] * 1.2, 0, 255)))
            y_sub = np.concatenate((y_sub, new_class * np.ones(len_class)))
            if DataBalancingQ:
                remaining_items = 6 * (max_class - len_class)
                idx_rnd = np.random.choice(idx, remaining_items)
                X_sub = np.concatenate((X_sub, X[idx_rnd, :, :, 0]))
                y_sub = np.concatenate((y_sub, new_class * np.ones(remaining_items)))
        elif DataBalancingQ:
            remaining_items = max_class - len_class
            idx_rnd = np.random.choice(idx, remaining_items)
            X_sub = np.concatenate((X_sub, X[idx_rnd, :, :, 0]))
            y_sub = np.concatenate((y_sub, new_class * np.ones(remaining_items)))

        X_final = np.concatenate((X_final, X_sub))
        y_final = np.concatenate((y_final, y_sub))
    
    X_final /= 255
    X_final = X_final.astype(np.float32)
    y_final = y_final.astype(np.int)
    size_ds = y_final.shape[0]
    np.random.seed(seed)
    rnd_idx = np.random.permutation(size_ds)
    x_train, x_test = np.split(X_final[rnd_idx, :, :], [int(size_ds*0.8)])
    y_train, y_test = np.split(y_final[rnd_idx], [int(size_ds*0.8)])
    return x_train, x_test, y_train, y_test



def build_dataset_fruits(folder, H, W):
    imgstr = '/dataset/fruits'

    x_train = np.zeros((0, W, H))
    y_train = np.zeros((0,), dtype=np.int)
    def create_image(filepath):
        img = Image.open(filepath).convert('L')
        w, h = img.size
        mwh = min(w, h)
        left = (w - mwh)//2
        top = (h - mwh)//2
        img = img.crop((left, top, left+mwh, top+mwh))
        img = img.resize((W,H))
        img_np = np.asarray(img).reshape((1, W, H))
        return img_np
    for filepath in glob.iglob(folder + imgstr + '/train/apple*.jpg'):
        img_np = create_image(filepath)
        x_train = np.vstack((x_train, img_np))
        y_train = np.concatenate((y_train, np.array([0])))
    for filepath in glob.iglob(folder + imgstr + '/train/banana*.jpg'):
        img_np = create_image(filepath)
        x_train = np.vstack((x_train, img_np))
        y_train = np.concatenate((y_train, np.array([1])))
    for filepath in glob.iglob(folder + imgstr + '/train/orange*.jpg'):
        img_np = create_image(filepath)
        x_train = np.vstack((x_train, img_np))
        y_train = np.concatenate((y_train, np.array([2])))
    x_test = np.zeros((0, W, H))
    y_test = np.zeros((0,), dtype=np.int)
    for filepath in glob.iglob(folder + imgstr + '/test/apple*.jpg'):
        img_np = create_image(filepath)
        x_test = np.vstack((x_test, img_np))
        y_test = np.concatenate((y_test, np.array([0])))
    for filepath in glob.iglob(folder + imgstr + '/test/banana*.jpg'):
        img_np = create_image(filepath)
        x_test = np.vstack((x_test, img_np))
        y_test = np.concatenate((y_test, np.array([1])))
    for filepath in glob.iglob(folder + imgstr + '/test/orange*.jpg'):
        img_np = create_image(filepath)
        x_test = np.vstack((x_test, img_np))
        y_test = np.concatenate((y_test, np.array([2])))
    for filepath in glob.iglob(folder + imgstr + '/test/mixed*.jpg'):
        img_np = create_image(filepath)
        x_test = np.vstack((x_test, img_np))
        y_test = np.concatenate((y_test, np.array([2])))
    x_train /= 255
    x_test /= 255
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int)
    y_test = y_test.astype(np.int)

    return x_train, x_test, y_train, y_test



def build_dataset_frozen_noise(folder, H, W, img_per_class, seed=1):
    x_train = np.zeros((2 * img_per_class, W, H))
    y_train = np.zeros((2 * img_per_class,), dtype=np.int)
    np.random.seed(seed)
    for i in range(img_per_class):
        x_train[2*i, :, :] = np.random.rand(H, W)
        x_train[2*i+1, :, :] = np.random.rand(H, W)
        y_train[2*i] = 0
        y_train[2*i+1] = 1
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int)

    return x_train, x_train, y_train, y_train
