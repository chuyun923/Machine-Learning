import matplotlib.image as pig
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethod

image_raw_data = tf.gfile.FastGFile('./logo/maoyan.jpg','rb').read()

maoyan = tf.image.decode_jpeg(image_raw_data)

# tf.image.resize_images(pic,[500,500])


def resize_img(input):
    return tf.image.resize_images(input,[500,500],1)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    res = resize_img(maoyan)

    plt.imshow(res.eval())

plt.show()






