from keras.layers import UpSampling3D, Conv3D, MaxPool3D, ReLU

def Conv3d(input, ch, f, pad= 'same', act = 'leakyrelu'):
    with tf.name_scope("Unet"):
        x = input
        x = Conv3D(ch, 3, padding = pad, use_bias=True)(x)
        if act == 'relu':
            x = tf.nn.relu(x)
        elif act == 'sigmoid':
            x = tf.sigmoid(x)
        elif act == 'leakyrelu':
            x = tf.nn.leaky_relu(x, alpha = 0.01)
        return x

def MaxPool3d(input, ps):
    x = input
    x = MaxPool3D(pool_size=ps, strides=ps)(x)
    return x

def Upsample3d(input, scale):
    x = input
    x = UpSampling3D(size = (scale,scale,scale), data_format ="channels_last")(x)
    return x

def Deconv3d(input, ch, upscale, batch, xdim, ydim, zdim, pad= 'SAME'):
    x = input
    x = Conv3DTranspose(x, filter=tf.constant(np.ones([upscale,upscale,upscale,ch,ch], np.float32)),
                               output_shape=[batch, xdim, ydim, zdim, ch],
                                strides=[1, upscale, upscale, upscale, 1],
                                padding= pad)
#     x = tf.layers.conv3d_transpose(x, ch, f, strides = [2,2,2], padding = pad, kernel_initializer=tf.contrib.layers.xavier_initializer(), use_bias=True)
    return x

def Unet(placeholders, batch):
    #decode
   
    x = placeholders['input']
    print("Input",x)
    x = Conv3d(x, 64, 3)
    x = Conv3d(x, 64, 3)
    conv1 = x
    x = MaxPool3d(x, 2)
    print("Conv1", conv1)
    
    x = Conv3d(x, 128, 3)
    x = Conv3d(x, 128, 3)
    conv2 = x
    x = MaxPool3d(x, 2)
    print("Conv2", conv2)
    
    x = Conv3d(x, 256, 3)
    x = Conv3d(x, 256, 3)
    conv3 = x
    x = MaxPool3d(x, 2)
    print("Conv3", conv3)
    
    x = Conv3d(x, 512, 3)
    x = Conv3d(x, 512, 3)
    conv4 = x
    x = MaxPool3d(x, 2)
    print("Conv4", conv4)
    
    #encode
    x = Conv3d(x, 512, 3)
    x = Conv3d(x, 512, 3)
    conv5 = x
    print("Conv5", conv5)
#     x = Deconv3d(x, 512, 2, batch, input_depth//8, input_height//8, input_width//8)
    x = Upsample3d(x,2)
    print("Upsample5", x)
    x = tf.concat([x, conv4], axis = 4)
    merge5 = x
    print("Merge5", merge5)
    
    x = Conv3d(x, 256, 3)
    x = Conv3d(x, 256, 3)
    conv6 = x
    print("Conv6", conv6)
#     x = Deconv3d(x, 256, 2, batch, input_depth//4, input_height//4, input_width//4)
    x = Upsample3d(x,2)
    x = tf.concat([x, conv3], axis = 4)
    merge6 = x
    print("Merge6", merge6)
    
    x = Conv3d(x, 128, 3)
    x = Conv3d(x, 128, 3)
    conv7 = x
    print("Conv7", conv7)
#     x = Deconv3d(x, 128, 2, batch, input_depth//2, input_height//2, input_width//2)
    x = Upsample3d(x,2)
    x = tf.concat([x, conv2], axis = 4)
    merge7 = x
    print("Merge7", merge7)
    
    x = Conv3d(x, 64, 3)
    x = Conv3d(x, 64, 3)
    conv8 = x
    print("Conv8", conv8)
#     x = Deconv3d(x, 64, 2, batch, input_depth, input_height, input_width)
    x = Upsample3d(x,2)
    x = tf.concat([x, conv1], axis = 4)
    merge8 = x
    print("Merge8", merge8)
    
    x = Conv3d(x, 32, 3)
    x = Conv3d(x, 32, 3)
    x = Conv3d(x, output_class_num, 1)
    conv9 = x
    print("Conv9", conv9)
    
    return x
    
    placeholders = {
    'input': tf.placeholder(tf.float32,shape=(None,input_depth,input_height,input_width,1)),
    'label': tf.placeholder(tf.float32,shape=(None,input_depth,input_height,input_width,output_class_num))
    }


    #with tf.variable_scope('Unet_model') as scope:
    Unet_out_tensor = Unet(placeholders, batch_size)