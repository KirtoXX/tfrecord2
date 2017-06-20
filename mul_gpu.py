import tensorflow as tf

def average_gradients(self, tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def gpus_train_op(x,y_,inference,loss=tf.losses.softmax_cross_entropy,opt=tf.train.GradientDescentOptimizer(learning_rate=0.1),N_gpus=1):
    #默认使用 softmax 回归
    #默认使用SGD优化
    x = tf.split(axis=0, num_or_size_splits=N_gpus, value=x)
    y_ = tf.split(axis=0, num_or_size_splits=N_gpus, value=y_)
    tower_grads = []
    for i in range(N_gpus):
        with tf.device('GPU:%d'%i):
            with tf.variable_scope('grad:%d'%i):
                logits = inference(x[i])       #前向传播
                gpu_loss = loss(y_[i],logits,scope='loss')   #scope: gpu:i/loss
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(gpu_loss)    #计算反传梯度
                tower_grads.append(grads)
    average_grads = average_gradients(grads)
    apply_gradients_op = opt.apply_gradients(average_grads)
    return apply_gradients_op

