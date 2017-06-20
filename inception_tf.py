import tensorflow as tf
import numpy as np

def main(argv=None):
    inception_graph_def_file = 'inception-2015-12-05/classify_image_graph_def.pb'

    x = tf.placeholder(dtype=tf.float32,shape=[None,299,299,3])
    data = np.random.rand(1, 299, 299, 3)

    with tf.Session() as sess:
        with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            #-----define new graph----
            output = tf.import_graph_def(graph_def,input_map={'conv':x},return_elements=['pool_3'])
            logits = sess.run(output,feed_dict={x:data})

            print(logits)
            writer_op = tf.summary.FileWriter("logs/", sess.graph)

if __name__ == '__main__':
    tf.app.run()