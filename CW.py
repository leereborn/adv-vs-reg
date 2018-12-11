import tensorflow as tf
import keras
import numpy as np
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.dataset import MNIST
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import model_eval
from models import cnn_model, mlp_model

NB_EPOCHS_LEGITIMATE = 100
NB_EPOCHS_ADV = 50
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = 'train_dir'
FILENAME = 'mnist.ckpt'
LABEL_SMOOTHING = 0

keras.layers.core.K.set_learning_phase(0)

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
#sess = tf.Session()
#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
keras.backend.set_session(sess)

# Get MNIST test data
mnist = MNIST(train_start=0, train_end=2500,
                test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

# Define TF model graph

model = cnn_model(img_rows=img_rows, img_cols=img_cols, channels=nchannels, nb_filters=128, nb_classes=nb_classes)

#model = mlp_model(img_rows=img_rows, img_cols=img_cols,nb_classes=nb_classes)
preds = model(x)
print("Defined TensorFlow model graph.")

# Train an MNIST model
train_params_leg = {
    'nb_epochs': NB_EPOCHS_LEGITIMATE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'train_dir': TRAIN_DIR,
    'filename': FILENAME
}

rng = np.random.RandomState([2018, 11, 26])
wrap = KerasModelWrapper(model)

# function that is run after each training iteration
def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': BATCH_SIZE}
    acc_test = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    acc_train = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
    report.clean_train_clean_eval = acc_test
    print('Test set accuracy on legitimate examples: %0.4f' % acc_test)
    print('Training set accuracy on legitimate examples: %0.4f' % acc_train)

# training on legitimate data
loss = CrossEntropy(wrap, smoothing=LABEL_SMOOTHING)
train(sess, loss, x_train, y_train, evaluate=evaluate, args=train_params_leg, rng=rng)

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
cw = CarliniWagnerL2(wrap, sess=sess)
cw_params = {'binary_search_steps': 1,
               'max_iterations': 100,
               'learning_rate': .2,
               'batch_size': 16,
               'initial_const': 10}
def attack(x):
    return cw.generate(x, **cw_params)

#adv_x = attack(x)
#preds_adv = model(adv_x)

print("Repeating the process, using adversarial training")

loss_2 = CrossEntropy(wrap, smoothing=LABEL_SMOOTHING, attack=attack)

def evaluate_2():
    # Accuracy of adversarially trained model on legitimate test inputs
    eval_params = {'batch_size': BATCH_SIZE}
    accuracy = model_eval(sess, x, y, preds, x_test, y_test,
                            args=eval_params)
    print('Test accuracy on legitimate examples: %0.4f' % accuracy)
    report.adv_train_clean_eval = accuracy

train_params_adv = {
    'nb_epochs': NB_EPOCHS_ADV,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'train_dir': TRAIN_DIR,
    'filename': FILENAME
}
# Perform and evaluate adversarial training
train(sess, loss_2, x_train, y_train, evaluate=evaluate_2, args=train_params_adv, rng=rng)