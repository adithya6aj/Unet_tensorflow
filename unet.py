from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

def conv_block(input,num_filters):
    x = Conv2D(num_filters, 3, activation='relu',padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 

    x = Conv2D(num_filters, 3, activation='relu',padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    return x

#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    s = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return s, p  

#Decoder block for unet
#skip features gets input from encoder for concatenation
def decoder_block_for_unet(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet_model(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block_for_unet(b1, s4, 512)
    d2 = decoder_block_for_unet(d1, s3, 256)
    d3 = decoder_block_for_unet(d2, s2, 128)
    d4 = decoder_block_for_unet(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())
    return model

input_shape =(256,256,3)
model = build_unet_model(input_shape)
model.compile(optimizer=Adam(lr=1e-3),loss='binary_crossentropy',metrics = ['accuracy'])
model.summary