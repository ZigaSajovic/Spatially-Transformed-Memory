import tensorflow as tf

def linear(X, shape, bias_initial=0., name="linear"):
    with tf.variable_scope(name):
        w = tf.get_variable(shape=shape, name="weight",
                            initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                                                           seed=None, dtype=tf.float32))
                            #initializer=tf.random_normal_initializer(stddev=0.1))  # initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))
        b = tf.get_variable(shape=[shape[1]], name="bias", initializer=tf.constant_initializer(bias_initial))
        return tf.matmul(X,w)+b


def STM(X,memory, h, controler, hC_, shape, wordSize, mem_shape, controler_size, controller_activation, reuse=None, full=False, name="STM"):
    assert wordSize[0]*wordSize[1]==shape[1], "size of words has to match size of output-> %d * %d != %d"%(wordSize[0],wordSize[1],shape[1])
    def LSTM(H_, C_, outSize):
        with tf.variable_scope("LSTM", reuse=reuse):
            in_ = tf.concat((X, H_), axis=1)
            w = tf.get_variable(shape=(shape[0] + outSize, outSize * 4), name="weight", initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))
            b = tf.get_variable(shape=[outSize * 4], name="bias", initializer=tf.constant_initializer(0.0))
            y_ = tf.matmul(in_, w) + b
            sig_ = tf.slice(y_, (0, 0), (-1, outSize * 3))
            tanh_ = tf.slice(y_, (0, outSize * 3), (-1, outSize))
            f, i, o = tf.split(tf.sigmoid(sig_), 3, 1)
            c_ = tf.tanh(tanh_)
            ct = f * C_ + i * c_
            ht = o * tf.tanh(ct)
            return ht, ct
    with tf.variable_scope(name):
        def _controler():
            with tf.variable_scope("controller", reuse=reuse):
                ht, ct = LSTM(hC_, controler, controler_size)
                c_size=6 if full else 4
                bias_initial=[1.,0.,0.,0.,1.,0.] if full else [1.,0.,1.,0.]
                theta=controller_activation(linear(ht,(controler_size,c_size),name="firstLinear", bias_initial=bias_initial))
                if not full:
                    theta_=theta
                    a_=tf.slice(theta_,(0,0),(-1,1))
                    b_ = tf.slice(theta_, (0, 1), (-1, 1))
                    c_ = tf.slice(theta_, (0, 2), (-1, 2))
                    z1,z2=tf.split(tf.zeros((tf.shape(X)[0],2)),2,1)
                    theta=tf.concat((a_,z1,b_,z2,c_), axis=1)
                return ht, ct, theta
        # with tf.variable_scope("preMap", reuse=reuse):
        #     X_=tf.tanh(linear(X,(input_size,input_size)))
        # X=X_
        hC, cC, theta_=_controler()
        theta=tf.reshape(theta_,(-1,6))
        word, idx=transformer(tf.reshape(memory,mem_shape), theta, wordSize,"readWord")
        def _mem():
            with tf.variable_scope("mem", reuse=reuse):
                ht, ct = LSTM(h, word, shape[1])
                return ht, ct
        hM,cM=_mem()
        #memory_written=tf.scatter_update(memory,idx,tf.reshape(cM,[-1]))
        memory_written = tf.dynamic_stitch((tf.range(tf.size(memory)), idx), (memory, tf.reshape(cM, [-1])))
        return hM,memory_written, hC,cC,idx, theta

def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = ((x1_f-x) * (y1_f-y))
            wb = ((x1_f-x) * (y-y0_f))
            wc = ((x-x0_f) * (y1_f-y))
            wd = ((x-x0_f) * (y-y0_f))
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output, idx_a

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed, idx = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)
            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height*out_width]))
            return output, idx

    with tf.variable_scope(name):
        output, idx = _transform(theta, U, out_size)
        return output, idx
