"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""
import gzip
import os
import sys
import urllib
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

# own imports
from sklearn.metrics.classification import f1_score
from sklearn.model_selection import KFold

#from submission.mask_to_submission import patch_to_label

PROJECT_ROOT = '../../'
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
#VALIDATION_SIZE = 20  # Size of the validation set.
TEST_SIZE = 50  # Size of the test set. (fixed!)
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
DEFAULT_THRESHOLD = 0.25 # Fixed by mask_to_submission
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
VISUAL_OUTPUT = False
TRAINING_RUN = True
CROSSVAL_RUN = False
TEST_RUN = False

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
CROP_STEP = 8 # Needs to be even divisor of IMG_PATCH_SIZE

tf.app.flags.DEFINE_string('train_dir', PROJECT_ROOT+'models',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > DEFAULT_THRESHOLD:
        return 1
    else:
        return 0

# Extract patches from a given image
def img_sliding_crop(im, w, h, step):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight-IMG_PATCH_SIZE+1,step):
        for j in range(0,imgwidth-IMG_PATCH_SIZE+1,step):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(indices, filename):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Loading images from disk...')
    imgs = []
    for i in indices:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            # print(img.shape)
            # img_comb = np.concatenate((img,img), 2) #TODO: multidimension picture
            # print(img_comb.shape)
            # img_arr = np.array(img_comb)
            # print(img_arr.shape)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_sliding_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CROP_STEP) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(indices, filename):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    print('Loading labels from disk...')

    for i in indices:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_sliding_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CROP_STEP) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


# Extract groundtruth labels like submission
def extract_sub_gt(indices, filename):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    print('Loading labels from disk...')

    for i in indices:
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], 16, 16) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

# Convert array of labels to an image
def label_vote(imgwidth, imgheight, w, h, step, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    array_cnt = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight-h+1,step):
        for j in range(0,imgwidth-w+1,step):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] += l
            array_cnt[j:j+w, i:i+h] += 1
            idx = idx + 1
    array_labels = array_labels / array_cnt
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img




def full_run(train_indices, val_indices, thresh):

    if val_indices.size:
        val_dir = PROJECT_ROOT+'data_dir/' #TODO back to validation
        val_data_filename = val_dir + 'images_3/'
        val_labels_filename = val_dir + 'groundtruth_3/'

        # Extract val DATA into numpy arrays.
        val_data = extract_data(val_indices, val_data_filename)
        val_labels = extract_labels(val_indices, val_labels_filename)

    if TEST_RUN:
        # Extract TEST DATA into numpy arrays.
        test_dir = PROJECT_ROOT+'test_dir/'
        test_data = extract_data(range(1,51), test_dir)


    data_dir = PROJECT_ROOT+'data_dir/'
    train_data_filename = data_dir + 'images_3/'
    train_labels_filename = data_dir + 'groundtruth_3/'

    if TRAINING_RUN: #TODO do some serious code splitting
        # Extract TRAIN DATA into numpy arrays.
        train_data = extract_data(train_indices, train_data_filename)
        train_labels = extract_labels(train_indices, train_labels_filename)
    else:
        train_data = extract_data([1], train_data_filename)
        train_labels = extract_labels([1], train_labels_filename)

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print (len(new_indices))
    print (train_data.shape)
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]


    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))


    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    #train_all_data_node = tf.constant(train_data)
    #train_all_data_node = tf.Variable(train_data)


    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):
        data = numpy.asarray(img_sliding_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, CROP_STEP))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = tf_session.run(output)
        img_prediction = label_vote(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, CROP_STEP, output_prediction)

        return img_prediction

    # Get prediction for given input image
    def get_prediction_for_patch(filename, image_idx):
        '''Returns predictions for list of patches for given image'''

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        #img_new = np.concatenate((img, img), 2)  # TODO: multidimension picture

        data = numpy.asarray(img_sliding_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, CROP_STEP))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = tf_session.run(output)

        patch_pred = []

        for patch in output_prediction:
            patch_pred.append(patch_to_label(patch[0]))

        return patch_pred

    # Get prediction for actual output
    def get_prediction_for_actual(img_prediction, threshold):
        '''Returns actual prediction labels of submission patches for given image'''

        # threshold = 0.25 is default
        patch_size = 16 # Fixed due to submission restrictions!
        patch_pred = []

        for j in range(0, img_prediction.shape[1], patch_size):
            for i in range(0, img_prediction.shape[0], patch_size):
                patch = img_prediction[i:i + patch_size, j:j + patch_size]
                df = np.mean(patch)
                if df > threshold:
                    label = 1
                else:
                    label = 0
                patch_pred.append(label)

        return patch_pred

    # Get prediction for submission
    def get_prediction_for_submission(img_prediction):

        img_pred = img_float_to_uint8(1-img_prediction)
        simg = Image.fromarray(img_pred)

        return simg

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(img, img_prediction):

        cimg = concatenate_images(img, 1-img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(img, img_prediction):

        oimg = make_img_overlay(img, 1-img_prediction)

        return oimg

    def save_imagery(filename, image_idx, prediction_dir):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)

        pimg = get_prediction_with_groundtruth(img, img_prediction)
        Image.fromarray(pimg).save(prediction_dir + "prediction_" + '%.3d' % i + ".png")
        oimg = get_prediction_with_overlay(img, img_prediction)
        oimg.save(prediction_dir + "overlay_" + '%.3d' % i + ".png")
        simg = get_prediction_for_submission(img_prediction)
        simg.save(prediction_dir + "submission_" + '%.3d' % i + ".png")

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.image_summary('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            filter_summary2 = tf.image_summary('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.image_summary('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.image_summary('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.image_summary('summary_pool2' + summary_id, s_pool2)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    tf.scalar_summary('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.scalar_summary(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.scalar_summary('learning_rate', learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    #train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as tf_session:


        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(tf_session, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                    graph_def=tf_session.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions = tf_session.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = tf_session.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(tf_session, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        train_score_sub = 0
        train_score_pat = 0
        if TRAINING_RUN:
            print ("Running prediction on TRAINING set")
            prediction_training_dir = PROJECT_ROOT+'predictions_train/'
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)

            y_pred = []
            y = []
            y_pred_pat = []
            y_pat = []

            for i in train_indices:
                if (VISUAL_OUTPUT and False): #TODO unfalsify
                    save_imagery(train_data_filename, i, prediction_training_dir)

                # Get prediction
                imageid = "satImage_%.3d" % i
                image_filename = train_data_filename + imageid + ".png"
                gt_filename = train_labels_filename + imageid + ".png"
                img = mpimg.imread(image_filename)
                gt = mpimg.imread(gt_filename)
                img_prediction = get_prediction(img)

                # Aggregate error & gt
                y_pred += get_prediction_for_actual(img_prediction, thresh)
                y += get_prediction_for_actual(gt, thresh)

                # Aggregate error patchwise
                y_pred_pat += get_prediction_for_patch(train_data_filename, i)

            y_pat = [label[0] for label in extract_labels(train_indices, train_labels_filename)]

            train_score_sub = f1_score(y, y_pred, average='weighted')  # TODO: why weighted?
            train_score_pat = f1_score(y_pat, y_pred_pat, average='weighted')  # TODO: why weighted?
            print("Run TRAINING score sub achieved: ", train_score_sub)
            print("Run TRAINING score pat achieved: ", train_score_pat)

        val_score_sub = 0
        val_score_pat = 0
        if val_indices.size:
            print ("Running prediction on VALIDATION set")
            prediction_val_dir = PROJECT_ROOT+'predictions_val/'
            if not os.path.isdir(prediction_val_dir):
                os.mkdir(prediction_val_dir)

            y_pred = []
            y = []
            y_pred_pat = []
            y_pat = []

            for i in val_indices:
                if (VISUAL_OUTPUT):
                    save_imagery(val_data_filename, i, prediction_val_dir)

                # Get prediction
                imageid = "satImage_%.3d" % i
                image_filename = val_data_filename + imageid + ".png"
                gt_filename = val_labels_filename + imageid + ".png"
                img = mpimg.imread(image_filename)
                gt = mpimg.imread(gt_filename)
                img_prediction = get_prediction(img)

                # Aggregate error & gt
                y_pred += get_prediction_for_actual(img_prediction, thresh)
                y += get_prediction_for_actual(gt, thresh)

                # Aggregate error patchwise
                y_pred_pat += get_prediction_for_patch(val_data_filename, i)

            y_pat = [label[0] for label in extract_labels(val_indices, val_labels_filename)]

            val_score_sub = f1_score(y, y_pred, average='weighted') #TODO: why weighted?
            val_score_pat = f1_score(y_pat, y_pred_pat, average='weighted') #TODO: why weighted?
            print("Run VALIDATION score sub achieved: ", val_score_sub)
            print("Run VALIDATION score pat achieved: ", val_score_pat)


        if (TEST_RUN):
            print ("Running prediction on TEST set")
            prediction_test_dir = PROJECT_ROOT+'predictions_test/'
            if not os.path.isdir(prediction_test_dir):
                os.mkdir(prediction_test_dir)

            for i in range(1, TEST_SIZE+1):
                save_imagery(test_dir, i, prediction_test_dir)
                print('Progress: ', i,'50')

        print('--END OF RUN--')

        return val_score_pat, train_score_pat

def main(argv=None):  # pylint: disable=unused-argument

    if (CROSSVAL_RUN):
        X = range(TRAINING_SIZE)
        kf = KFold(n_splits=4)
        training_mean_err = []
        validation_mean_err = []

        threshs = np.linspace(0.1,0.9,8)
        results = []

        for t in threshs:
            for train, val in kf.split(X):
                val_score, train_score = full_run(train+1, val+1, t)
                training_mean_err.append(train_score)
                validation_mean_err.append(val_score)
                print("%s %s" % (train+1, val+1))
                tf.reset_default_graph()

            final_train_score = np.mean(training_mean_err)
            final_val_score = np.mean(validation_mean_err)
            print('Training score values: ', final_train_score)
            print('Validation score values: ', final_val_score)
            print("Final TRAINING score achieved: ", final_train_score)
            print("Final VALIDATION score achieved: ", final_val_score)
            results.append((t, final_train_score, final_val_score))

        test_score = [x[2] for x in results]
        max_idx = np.argmax(test_score)
        print('Best threshold: ', threshs[max_idx], ' with index: ', max_idx, ' and test score: ', test_score[max_idx])
        print('Final results: ', results)

    else:
        full_run(range(1, TRAINING_SIZE+1), np.array([]), 0.25)


if __name__ == '__main__':
    tf.app.run()
