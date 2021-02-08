import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import tqdm, PIL.Image 

def plot_loss(loss_log, loss_log_classes, folder, id_run, target_classes, color_lookup, save=False):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    max_log_loss = -10
    min_log_loss = 10
    for i in range(loss_log_classes.shape[1]):
        loss_log_class = loss_log_classes[:, i]
        if loss_log_class.shape[0] > 0:
            log_loss_class = np.log10(loss_log_class)
            tmp_min, tmp_max = np.percentile(log_loss_class, [0, 97])
            max_log_loss = max(max_log_loss, tmp_max)
            min_log_loss = min(min_log_loss, tmp_min)
            plt.plot(log_loss_classes, alpha=0.5, label=target_classes[i], color=color_lookup[i].numpy()/255)
    if max_log_loss == -10:
        plt.ylim([-4, 5])
    else:
        plt.ylim([min_log_loss, max_log_loss])
    plt.legend()
    if save:
        plt.savefig(folder + '/output/log_loss_{}'.format(id_run))
    plt.show()


class VideoWriter:
    def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        if len(img.shape) == 3 and img.shape[-1] == 4:
            img = img[..., :3] * img[..., 3, None]
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

def make_run_videos(folder, id_run, i_step, TST_EVOLVE, MutateTestingQ, x, ca, color_lookup):
    with VideoWriter(folder + '/output/Movie_model_{}_{}.mp4'.format(id_run, i_step)) as vid:
        for i in tqdm.trange(-1, TST_EVOLVE):
            if MutateTestingQ and i == int(TST_EVOLVE / 2):
                c_i = x[:, :, :, 0]
                x = ca.mutate(x, tf.expand_dims(tf.random.shuffle(c_i), -1))
            image = zoom(tile2d(classify_and_color(ca, x, color_lookup)), scale=4)
            if i > -1:
                x = ca(x)
                image = zoom(tile2d(classify_and_color(ca, x, color_lookup)), scale=4)
            im = np.uint8(image * 255)
            im = PIL.Image.fromarray(im)
            vid.add(np.uint8(im))

def classify_and_color(ca, x, color_lookup):
    '''
    input ca: the CA class instance with proper architecture and learned weights
    input x: a state of the CA. its shape is (batch_size, height, width, no_channels).
    output: 
    '''
    current_image = x[:, :, :, 0]
    output_ca = ca.classify(x) # x[:, :, :, -NO_CLASSES:] shape = (b, h, w, NO_CLASSES) where we want values 0 if not that number and 1 if that number
    return color_images(ca, current_image, output_ca, color_lookup)

def color_images(ca, x, output_x, color_lookup, alpha_color=0.3):
    '''
    input x grayscale image: its shape is ([batch_size, ]height, width)
    input output_x: the output of the CA for the image x. its shape is ([batch_size, ]height, width, NO_CLASSES)
    output: rgba_pixels, with numbers between 0 and 1. its shate is ([b, ]h, w, 4)
    '''
    if ca.dp:
        is_alive = x # 1 if pixel is alive, 0 if background. its shape ([b, ]h, w)
        is_background = 1. - x # 0 if pixel is alive, 1 if background. its shape ([b, ]h, w)
        output_x = output_x * tf.expand_dims(is_alive, -1) # forcibly cancels everything outside of it. y_pic will be 0 for dead pixels. and the previous value for the alive pixels
        
        black_and_bg = tf.fill(list(x.shape) + [2], 0.01) # an array filled with 0.01 of shape ([b, ]h, w, 2). 0.01 is the minimum value for outputs to be considered
        black_and_bg *= tf.stack([is_alive, is_background], -1) # an array filled with 0.01 and 0's depending on alive status. of shape ([b, ]h, w, 2)

        number_or_background =  tf.concat([output_x, black_and_bg], -1) # the voting for the NO_CLASSES+2 different categories. (NO_CLASSES colors + original grayscale and background. its shape is ([b, ]h, w, NO_CLASSES+2)
        classified_pixels = tf.argmax(number_or_background, -1) # the result of the vote.  its shape is ([b, ]h, w)
        rgb_pixels = tf.gather(color_lookup, classified_pixels) # And now I suppose this shape is ([b, ]h, w, 3)
        return tf.cast(rgb_pixels, tf.float32) / 255.
    else:
        black = tf.fill(list(x.shape) + [1], 0.01) # an array filled with 0.01 of shape ([b, ]h, w, 1). 0.01 is the minimum value for outputs to be considered
        class_or_black =  tf.concat([output_x, black], -1) # the voting for the NO_CLASSES+1 different categories. (NO_CLASSES colors + undecided. its shape is ([b, ]h, w, NO_CLASSES+1)
        classified_class = tf.argmax(class_or_black, -1) # the result of the vote.  its shape is ([b, ]h, w)
        rgb_pixels = tf.gather(color_lookup, classified_class) # And now I suppose this shape is ([b, ]h, w, 3)
        rgb_pixels = tf.cast(rgb_pixels, tf.float32) / 255.
        rgb_composite = rgb_pixels * alpha_color + tf.expand_dims(x, -1) * (1 - alpha_color)
        return rgb_composite

def tile2d(a):
    '''
    a has shape (N, H, W, 3), where N is the number of images we will show 
    '''
    a = np.asarray(a) # make sure it is an array
    width = int(np.ceil(np.sqrt(len(a)))) # Number of columns for the tile
    H, W = a.shape[1:3] # Height and Width of each image
    pad = (width - len(a)) % width
    a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
    height = len(a) // width # Number of rows for the tile
    a = a.reshape([height, width] + list(a.shape[1:])) # reshaped a to (height, width, H, W, 3)
    a = np.rollaxis(a, 2, 1) # transform it into shape (height, H, width, W, 3)
    a = a.reshape([H*height, W*width] + list(a.shape[4:])) # transform into (height * H, width * W, 3)
    a = np.insert(a, [i*W for i in range(width + 1)], 1, axis=1) # add borders between images
    a = np.insert(a, [i*H for i in range(height + 1)], 1, axis=0)
    return a

def zoom(img, scale=4):
    '''
    Takes an image array and increases its resolution by simply repeating the values
    '''
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def add_pixel(old_image):
    new_image = old_image.copy()
    indices = np.asarray(np.where(old_image == 1)).T
    success = 0
    k = 0
    while (not success) or (k > 100):
        k += 1
        idx = np.random.randint(indices.shape[0])
        x, y = indices[idx, :]
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
        if old_image[nx, ny] < 0.5:
            new_image[nx, ny] = 1
            success = 1
    return new_image

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

def make_videos_increase(folder, id_run, i_step, TST_EVOLVE, ca, color_lookup, images, InitializeRandomQ, limits_classes):
    no_steps = images.shape[0]
    if InitializeRandomQ:
        x = ca.initialize_random(images[0, :, :, :])
    else:
        x = ca.initialize(images[0, :, :, :])
    with VideoWriter(folder + '/output/Movie_test_increase_{}_{}.mp4'.format(id_run, i_step)) as vid:
        for j in range(no_steps):
            x = ca.mutate(x, tf.expand_dims(images[j, :, :, :], -1))
            for i in tqdm.trange(TST_EVOLVE):
                x = ca(x)
                image = zoom(tile2d(classify_and_color(ca, x, color_lookup)), scale=4)
                im = np.uint8(image * 255)
                im[:4, :, :] = np.uint8(color_lookup[class_indice_f(j+1, limits_classes), :].numpy())
                im[-4:, :, :] = np.uint8(color_lookup[class_indice_f(j+1, limits_classes), :].numpy())
                im[:, :4, :] = np.uint8(color_lookup[class_indice_f(j+1, limits_classes), :].numpy())
                im[:, -4:, :] = np.uint8(color_lookup[class_indice_f(j+1, limits_classes), :].numpy())
                im = PIL.Image.fromarray(im)
                vid.add(np.uint8(im))
    if InitializeRandomQ:
        x = ca.initialize_random(images[0, :, :, :])
    else:
        x = ca.initialize(images[0, :, :, :])
    with VideoWriter(folder + '/output/Movie_test_decrease_{}_{}.mp4'.format(id_run, i_step)) as vid:
        for j in range(no_steps):
            x = ca.mutate(x, tf.expand_dims(images[no_steps - 1 - j, :, :, :], -1))
            for i in tqdm.trange(TST_EVOLVE):
                x = ca(x)
                image = zoom(tile2d(classify_and_color(ca, x, color_lookup)), scale=4)
                im = np.uint8(image * 255)
                im[:4, :, :] = np.uint8(color_lookup[class_indice_f(no_steps-j, limits_classes), :].numpy())
                im[-4:, :, :] = np.uint8(color_lookup[class_indice_f(no_steps-j, limits_classes), :].numpy())
                im[:, :4, :] = np.uint8(color_lookup[class_indice_f(no_steps-j, limits_classes), :].numpy())
                im[:, -4:, :] = np.uint8(color_lookup[class_indice_f(no_steps-j, limits_classes), :].numpy())
                im = PIL.Image.fromarray(im)
                vid.add(np.uint8(im))
