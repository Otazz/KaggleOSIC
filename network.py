

import os

import time

import cv2

import itertools

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt




from tensorflow.keras.optimizers import Adam




import tensorflow.keras.applications.inception_v3 as inception_v3

import tensorflow.keras.applications.inception_resnet_v2 as inception_resnet_v2

import tensorflow.keras.applications.densenet as densenet

import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2

import tensorflow.keras.applications.mobilenet as mobilenet

import tensorflow.keras.applications.resnet50 as resnet50

import tensorflow.keras.applications.vgg16 as vgg16

import tensorflow.keras.applications.vgg19 as vgg19

import tensorflow.keras.applications.xception as xception




from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

from tensorflow.keras.layers import Lambda




from tensorflow.keras.constraints import NonNeg




from tensorflow.keras.layers import Activation




from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing import image







# In[3]:







def loadimgs(path, shape_img = (595, 842)):

    '''

    path => Path of train directory or test directory

    '''



    X = []

    y = []



    for cartorio in os.listdir(path):

        print("Carregando Cartorio: " + cartorio)

        cartorio_path = os.path.join(path, cartorio)



        for filename in os.listdir(cartorio_path):

            image_path = os.path.join(cartorio_path, filename)

            img = image.load_img(image_path, target_size=shape_img)

            img = image.img_to_array(img)

            img = img[:int(img.shape[0] * 1/3.) , :, :]



            try:

                X.append(img)

                y.append(cartorio)

            except Exception as e:

                print(e)





    y = np.vstack(y)

    X = np.stack(X)



    return X, y







# In[4]:







def gaussian(x):

    return K.exp(-K.pow(x,2))




def get_siamese_model(name = None, input_shape = (224,224,3),

                      embedding_vec_size = 512, not_freeze_last = 2):

    """

        Model architecture

    """



    if name == "InceptionV3":

        base_model = inception_v3.InceptionV3(

            weights='imagenet', include_top=False)

        model_preprocess_input = inception_v3.preprocess_input



    if name == "InceptionResNetV2":

        base_model = inception_resnet_v2.InceptionResNetV2(

            weights='imagenet', include_top=False)

        model_preprocess_input = inception_resnet_v2.preprocess_input



    if name == "DenseNet121":

        base_model = densenet.DenseNet121(

            weights='imagenet', include_top=False)

        model_preprocess_input = densenet.preprocess_input



    if name == "DenseNet169":

        base_model = densenet.DenseNet169(

            weights='imagenet', include_top=False)

        model_preprocess_input = densenet.preprocess_input



    if name == "DenseNet201":

        base_model = densenet.DenseNet201(

            weights='imagenet', include_top=False)

        model_preprocess_input = densenet.preprocess_input



    if name == "MobileNetV2":

        base_model = mobilenet_v2.MobileNetV2(

            weights='imagenet', include_top=False)

        model_preprocess_input = mobilenet_v2.preprocess_input



    if name == "MobileNet":

        base_model = mobilenet.MobileNet(

            weights='imagenet', include_top=False)

        model_preprocess_input = mobilenet.preprocess_input



    if name == "ResNet50":

        base_model = resnet50.ResNet50(

            weights='imagenet', include_top=False)

        model_preprocess_input = resnet50.preprocess_input



    if name == "VGG16":

        base_model = vgg16.VGG16(

            weights='imagenet', include_top=False)

        model_preprocess_input = vgg16.preprocess_input



    if name == "VGG19":

        base_model = vgg19.VGG19(

            weights='imagenet', include_top=False)

        model_preprocess_input = vgg19.preprocess_input



    if name == "Xception":

        base_model = xception.Xception(

            weights='imagenet', include_top=False)

        model_preprocess_input = xception.preprocess_input



    # Verifica se existe base_model

    if 'base_model' not in locals():

        return ["InceptionV3", "InceptionResNetV2",

                "DenseNet121", "DenseNet169", "DenseNet201",

                "MobileNetV2", "MobileNet",

                "ResNet50",

                "VGG16", "VGG19",

                "Xception"

               ]



    # desativando treinamento

    for layer in base_model.layers[:-not_freeze_last]:

        layer.trainable = False




    x = base_model.layers[-1].output

    x = GlobalAveragePooling2D()(x)

    x = Dense(

        embedding_vec_size,

        activation = 'linear', # sigmoid? relu?

        name = 'embedding',

        use_bias = False

    )(x)



    model = Model(

        inputs = base_model.input,

        outputs = x

    )



    left_input = Input(input_shape)

    right_input = Input(input_shape)



    # Generate the encodings (feature vectors) for the two images

    encoded_l = model(left_input)

    encoded_r = model(right_input)



    # Add a customized layer to compute the absolute difference between the encodings

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])



    # Add a dense layer with a sigmoid unit to generate the similarity score

    prediction = Dense(

        1,

        activation = Activation(gaussian),

        use_bias = False,

        kernel_constraint = NonNeg()

    )(L1_distance)



    # Connect the inputs with the outputs

    siamese_net = Model(

        inputs=[left_input,right_input],

        outputs=prediction

    )



    return {

        "model" : siamese_net,

        "preprocess_input" : model_preprocess_input

    }







# In[5]:







train_folder = "dataset/train/"

val_folder = 'dataset/test/'

save_path = 'model_data/'







# In[6]:







X, y = loadimgs(train_folder)







# In[7]:







Xval, yval = loadimgs(val_folder)







# In[8]:







def show_img(img):

    plt.figure(figsize = (20,7))

    plt.imshow(img/255, aspect='auto', interpolation='nearest')







# In[9]:







show_img(X[10])







# In[10]:







model_name = "MobileNetV2"







# In[11]:







model_dict = get_siamese_model(model_name, X[0].shape)

model = model_dict["model"]

preprocess_input = model_dict["preprocess_input"]







# In[12]:







model.summary()







# In[13]:







X = preprocess_input(X)







# In[14]:







X.shape







# In[15]:







def get_index(list1, list2):



    comb = list(

        itertools.product(

            enumerate(list1),

            enumerate(list2)

        )

    )



    y = np.array([int(c[0][1][0] == c[1][1][0]) for c in comb])

    idx_left = np.array([c[0][0] for c in comb])

    idx_right = np.array([c[1][0] for c in comb])



    return y, idx_left, idx_right







# In[16]:







def get_batch(X, y, batch_size, proportion = 0.5):



    n_examples, width, height, depth = X.shape



    y_, idx_left, idx_right = get_index(y, y)



    idx_one = np.random.choice(

        np.where(y_ == 1)[0],

        int(batch_size * proportion)

    ).tolist()

    idx_zero = np.random.choice(

        np.where(y_ == 0)[0],

        int(batch_size * (1-proportion))

    ).tolist()



    sel_idx = idx_one + idx_zero

    np.random.shuffle(sel_idx)



    y_batch = y_[sel_idx]

    X_batch_l = X[idx_left[sel_idx]]

    X_batch_r = X[idx_right[sel_idx]]



    return [X_batch_l, X_batch_r], y_batch







# In[17]:







optimizer = Adam(lr = 0.001)

model.compile(loss="binary_crossentropy", optimizer=optimizer)







# In[18]:







batch_size = 64

n_epochs = 20

proportion = 0.3







# In[19]:







# train model on each dataset

for epoch in tqdm(range(n_epochs)):

    X_train, y_train = get_batch(X, y, batch_size, proportion)

    model.fit(X_train, y_train, batch_size = batch_size, epochs = 5)







# In[ ]:
















# In[20]:







model.save(model_name + "_siamese.h5")







# In[26]:







model_embedding = model.layers[2]







# In[21]:







model.predict([Xval[0:4], Xval[15:19]])







# In[22]:







show_img(Xval[3])







# In[23]:







show_img(Xval[18])







# In[24]:







model.predict([Xval[0:4], Xval[0:4]])







# In[32]:







model.summary()







# In[36]:







model2= Model(inputs=model.input, outputs=model.layers[-1].output)







# In[39]:







model2.predict([Xval[0:4], Xval[0:4]]) +0.5
