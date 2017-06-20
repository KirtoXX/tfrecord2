import tensorflow as tf
import Rread_and_Arguement as preprocessing
from inference_keras import Inception
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.pooling import AveragePooling2D

def inference(placeholder):
    pred = Inception(placeholder)
    pred = AveragePooling2D(pool_size=[8,8],name='pool')(pred)
    pred = Flatten(name='flatten')(pred)
    logits = Dense(8,name='fc')(pred)
    return logits

def main(argv=None):

    #-----------------set parm---------------------------

    tfrecord_path = 'tfrecord/image_tfrecord.tfrecords'
    nb_batch_size = 4
    nb_class = 8
    nb_min_after_dequeue = 2000
    nb_threads = 3
    nb_capacity = nb_min_after_dequeue + 3 * nb_batch_size

    #----------------------------------------------------

    image, lable = preprocessing.read_data(tfrecord_path=tfrecord_path)
    image = preprocessing.crop_and_resize(image)
    image_batch, label_batch = tf.train.shuffle_batch([image,lable],
                                                     capacity=nb_capacity,
                                                      batch_size=nb_batch_size,
                                                      num_threads=nb_threads,
                                                      min_after_dequeue=nb_min_after_dequeue,
                                                      )
    onehot_batch = preprocessing.dense_lable(label_batch,batch_size=nb_batch_size,nb_classes=nb_class)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    #-------define train--------------
    logits = inference(image_batch)
    loss = tf.losses.softmax_cross_entropy(onehot_batch,logits)
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)

    #-------train netowrk-------------
    with tf.Session() as sess:
        writer_op = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(1):
            print(i)
            sess.run(logits)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()





