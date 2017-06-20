import tensorflow as tf
import os
import numpy as np

'''
制作tfrecord  二进制数据
将数据存放在data文件夹下面
image_tfrecord 内容：lable,img_raw(tf.image 协议内存块)
'''

#image 解析类
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  return '.png' in filename


def _process_image(filename, coder):
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    image_data = coder.png_to_jpeg(image_data)
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def main(argv=None):

    #---------------set dir--------------------------
    #
    #
    data_path = 'data/'
    tfrecord_path = 'tfrecord/image_tfrecord.tfrecords'
    #
    #
    #------------------------------------------------

    coder = ImageCoder()
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    class1 = os.listdir(data_path)   #统计全部class
    num_class = len(class1)
    print('your class:')
    print(class1)
    for index,name in enumerate(class1):   #python 枚举
        #----one_hot标签制作------
        class_path = data_path+name +'/'
        images_path = os.listdir(class_path)
        print('process class '+name+'...')
        for signle_image in images_path:
            temp_path = class_path+signle_image
            image_buffer, height, width  =_process_image(temp_path, coder)
            img_raw = tf.compat.as_bytes(image_buffer)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
            writer.write(example.SerializeToString())
    print('done.')
    writer.close()


if __name__ == '__main__':
    tf.app.run()

