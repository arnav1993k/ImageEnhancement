import tensorflow as tf
def inception_block(feature_in,num):
	# tf.nn.separable_conv2d(image, depthwise_filter=3, pointwise_filter=100, strides=1, padding='SAME')
	simple1 =  tf.layers.conv2d(feature_in, 32, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_1' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
	simple2 =  tf.layers.conv2d(feature_in, 32, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_2' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
	simple3 =  tf.layers.conv2d(feature_in, 32, 1, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_1x1_3' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
	filter1 = tf.layers.conv2d(simple1, 32, 3, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_3x3' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
	filter2 = tf.layers.conv2d(simple2, 32, 5, strides = 1, padding = 'SAME', name = ('inceptionblock_%d_CONV_5x5' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE, activation='relu')
	stack = tf.concat(axis=3, values=[simple3, filter1, filter2,feature_in])
	return stack

def resblock(feature_in, num):
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(feature_in, 32, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_1' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    # temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_1' %num))
    temp = tf.nn.relu(temp)
        
    # subblock (conv. + BN + relu)
    temp =  tf.layers.conv2d(temp, 32, 3, strides = 1, padding = 'SAME', name = ('resblock_%d_CONV_2' %num), kernel_initializer = tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE)
    # temp = tf.layers.batch_normalization(temp, name = ('resblock_%d_BN_2' %num))
    temp = tf.nn.relu(temp)
    return temp + feature_in