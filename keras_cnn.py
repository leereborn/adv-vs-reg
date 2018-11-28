import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D

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
  :param img_rows: number of row in the image
  :param img_cols: number of columns in the image
  :param channels: number of color channels (e.g., 1 for MNIST)
  :param nb_filters: number of convolutional filters per layer
  :param nb_classes: the number of output classes
  :return:
  """
  model = Sequential()

  input_shape = (img_rows, img_cols, channels)

  layers = [Conv2D(filters=nb_filters, kernel_size=(8, 8), strides=(2, 2), padding="same",input_shape=input_shape),
            Activation('relu'),
            Conv2D(filters=(nb_filters * 2), kernel_size=(6, 6), strides=(2, 2), padding="valid"),
            Activation('relu'),
            Conv2D(filters=(nb_filters * 2), kernel_size=(5, 5), strides=(1, 1), padding="valid"),
            Activation('relu'),
            Flatten(),
            Activation('sigmoid'),
            Dense(512),
            Activation('sigmoid'),
            Dense(512),
            Activation('sigmoid'),
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
    Dense(nb_classes),
    Activation('softmax')
  ])
  if logits:
    logits_tensor = model(input_ph)

  if logits:
    return model, logits_tensor
  else:
    return model