import tensorflow as tf
from scipy.misc import imread, imresize
import numpy as np
use_quantized_graph = True

img = imread('Fishes/hilsha_9.jpg')
img = imresize(img, (299, 299, 3))
img = img.astype(np.float32)
img = np.expand_dims(img, 0)

img = img / 255.
img = img - 0.5
img = img * 2.

graph_filename = ('Fishes/Training/frzn_inception_v4_v-0.2.pb')
labels_file = ('Fishes/labels.txt')
labels_dict = {}

with open(labels_file, 'r') as f:
    for kv in [d.strip().split(':') for d in f]:
        labels_dict[int(kv[0])] = kv[1]

with tf.gfile.GFile(graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
    input_node = graph.get_tensor_by_name('import/input:0')
    output_node = graph.get_tensor_by_name('import/InceptionV4/Logits/Predictions:0')
    with tf.Session() as sess:
        predictions = sess.run(output_node, feed_dict={input_node:img})[0]
        top_5_predictions = predictions.argsort()[-4:][::-1]
        top_5_probabilities = predictions[top_5_predictions]
        prediction_names = [labels_dict[i] for i in top_5_predictions]
        for i in xrange(len(prediction_names)):
            print 'Prediction: %s, Probability: %s \n' % (prediction_names[i], top_5_probabilities[i])