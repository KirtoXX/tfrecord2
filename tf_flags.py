import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float('nb_class','3','your class')

FLAGS = flags.FLAGS

def main(argv=None):
    print(FLAGS.nb_class)

if __name__ == '__main__':
    tf.app.run()