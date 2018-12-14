"""Create the input data pipeline using `tf.data`"""

import numpy as np
import tensorflow as tf
import os
from collections import defaultdict


def _parse_function(filename, label, time, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label, time


def train_preprocess(image, label, time, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label, time


def input_fn(mode, filenames, labels, time, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
    mode: (string) can be 'train', 'eval' or 'infer'
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"
    assert len(labels) == len(time), "Labels and time should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l, t: _parse_function(f, l, t, params.image_size)
    train_fn = lambda f, l, t: train_preprocess(f, l, t, params.use_random_flip)

    if mode == 'train':
        params.train_size = num_samples
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels), tf.constant(time)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    elif mode == 'eval':
        params.eval_size = num_samples
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels), tf.constant(time)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    elif mode == 'infer':
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels), tf.constant(time)))
            .map(parse_fn)
            .batch(num_samples)
        )
    else:
        raise ValueError("data pipeline mode input wrong")

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels, time = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'time': time, 'iterator_init_op': iterator_init_op}
    return inputs


def load_data(mode, data_dir, data_path, params):
    class_names = sorted([dirname for dirname in os.listdir(data_path)
                          if os.path.isdir(os.path.join(data_path, dirname))])
    if mode == 'train':
        # build the dict
        class_dict = {}
        params.num_labels = len(class_names)
        for i in range(params.num_labels):
            class_dict[class_names[i]] = i
        np.save(os.path.join(data_dir, 'class_dict.npy'), class_dict)
    else:
        # read the dict
        class_dict = np.load(os.path.join(data_dir, 'class_dict.npy')).item()

    image_paths = defaultdict(list)

    for class_name in class_names:
        image_dir = os.path.join(data_path, class_name)
        for filepath in os.listdir(image_dir):
            if filepath.endswith(".jpg"):
                if class_dict[class_name] is not None:
                    image_paths[class_dict[class_name]].append(os.path.join(image_dir, filepath))
                else:
                    raise ValueError("found class not in train data")

    filenames = []
    labels = []

    for label in image_paths.keys():
        filenames.extend(image_paths[label])
        labels.extend([label] * len(image_paths[label]))

    # shuffle the lists for tri_loss evaluation
    zip_list = list(zip(filenames, labels))
    np.random.shuffle(zip_list)
    filenames, labels = zip(*zip_list)

    time = [int(f.split('/')[-1].partition('.')[0]) for f in filenames]

    return input_fn(mode, filenames, labels, time, params)
