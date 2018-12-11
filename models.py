import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Conv2D

def cnn_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=128, nb_classes=10):
  """
  Defines a CNN model using Keras sequential model
  :param logits: If set to False, returns a Keras model, otherwise will also
                  return logits tensor
  :param input_ph: The TensorFlow tensor for the input
                  (needed if returning logits)
                  ("ph" stands for placeholder but it need not actually be a
                  placeholder)
  :return:
  """
  model = Sequential()

  input_shape = (img_rows, img_cols, channels)
  '''
  layers = [Conv2D(filters=nb_filters, kernel_size=(8, 8), strides=(2, 2), padding="same",input_shape=input_shape),
            Activation('relu'),
            Conv2D(filters=(nb_filters * 2), kernel_size=(6, 6), strides=(2, 2), padding="valid"),
            Activation('relu'),
            Conv2D(filters=(nb_filters * 2), kernel_size=(5, 5), strides=(1, 1), padding="valid"),
            Activation('relu'),
            #Conv2D(filters=(nb_filters * 2), kernel_size=(3, 3), strides=(1, 1), padding="valid"),
            #Activation('relu'),
            Flatten(),
            Activation('sigmoid'),
            Dense(512),
            Activation('sigmoid'),
            Dense(512),
            Activation('sigmoid'),
            Dense(nb_classes)]
  '''
  layers = [Conv2D(filters=nb_filters, kernel_size=(3, 3), padding="same",input_shape=input_shape),
            Activation('elu'),
            Conv2D(filters=(nb_filters), kernel_size=(3, 3), padding="valid"),
            Activation('elu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=(nb_filters * 2), kernel_size=(3, 3), padding="same"),
            Activation('elu'),
            Conv2D(filters=(nb_filters * 2), kernel_size=(3, 3)),
            Activation('elu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=(nb_filters * 4), kernel_size=(3, 3), padding='same'),
            Activation('elu'),
            Conv2D(filters=(nb_filters * 4), kernel_size=(3, 3)),
            Activation('elu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            #Activation('sigmoid'),
            Dense(1024),
            Activation('elu'),
            Dense(128),
            Activation('elu'),
            Dense(nb_classes)]

  for layer in layers:
    model.add(layer)

  if logits:
    logits_tensor = model(input_ph)
  
  model.add(Activation('softmax'))

  if logits:
    return model, logits_tensor
  else:
    return model

def mlp_model(logits=False, input_ph=None, img_rows=28, img_cols=28,nb_classes=10):
  
  model = Sequential([
    Flatten(),
    Dense(512, input_shape=(img_rows*img_cols,)),
    Activation('sigmoid'),
    Dense(512),
    Activation('sigmoid'),
    Dense(512),
    Activation('sigmoid'),
    Dense(512),
    Activation('sigmoid'),
    Dense(512),
    Activation('sigmoid'),
    Dense(512),
    Activation('sigmoid'),
    Dense(nb_classes),
    Activation('softmax')
  ])
  if logits:
    logits_tensor = model(input_ph)

  if logits:
    return model, logits_tensor
  else:
    return model