# python tSNE_layers.py ./out_datasets/40-4_p20_d250_c_train.npz ../models/bilstm_fulldataset_pots_changes/out_model_bilstm.h5 -hog

from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import keras, sys
from keras.models import Model
import json, re, os
from cv_scripts.libs.mi_hog import normalize

layers = ['bidirectional_4','dense_8']

feature_files = False
pattern = re.compile("out_features_[a-z_0-9]+.txt")
for filepath in os.listdir("."):
    print(filepath)
    if pattern.match(filepath):
        feature_files = True
        break

# Load the model
model = keras.models.load_model(sys.argv[2])
model.summary()

# Load data
file1 = sys.argv[1]
files = np.load(file1, allow_pickle=True)
X, labels = files['a'], files['b']

N_CLASSES = np.max(labels) + 1
N_SAMPLES = X.shape[0]
N_TIMESTEPS = X.shape[1]

HOG_H = X.shape[2]
HOG_W = X.shape[3]
ORIENTATIONS = X.shape[4]

if '_train' in file1:
    metadata_file = file1.rsplit('.',maxsplit=1)[0][:-6] + "_metadata.json"
elif '_test' in file1:
    metadata_file = file1.rsplit('.',maxsplit=1)[0][:-5] + "_metadata.json"
else:
    metadata_file = file1.rsplit('.',maxsplit=1)[0] + "_metadata.json"
metadata_in = open(metadata_file,)
metadata = json.load(metadata_in)

class_labels = metadata['classes']
metadata_train_samples = metadata['samples_test']

#check if file exist
if not feature_files:
    # Normalice
    X_norm = []

    if sys.argv[3] == '-max':
        for row in range(N_SAMPLES):
            add_samples = []
            for sample in range(N_TIMESTEPS):
                ortientation_hist = X[row,sample,:,:,:]
                normalized_hist = ortientation_hist / np.max(ortientation_hist)
                add_samples.append(normalized_hist)
            X_norm.append(np.array(add_samples))

        X_norm = np.array(X_norm)
        X = X_norm
    elif sys.argv[3] == '-hog':
        X_norm = []
        for row in range(N_SAMPLES):
            add_samples = []
            for sample in range(N_TIMESTEPS):
                ortientation_hist = X[row,sample,:,:,:]
                normalized_hist = normalize(ortientation_hist)
                add_samples.append(normalized_hist)
            X_norm.append(np.array(add_samples))

        X_norm = np.array(X_norm)

    base_model = model
    base_model.summary()
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('bidirectional_4').output)

    for layer in layers:
        model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

        N_FEATURES = X_norm.shape[2]

        #matrix = np.empty((0,128+1))
        matrix = []
        for row_i in range(N_SAMPLES):
            hog_seq = X_norm[row_i,:]
            hog_seq = hog_seq.reshape(1,N_TIMESTEPS,N_FEATURES)

            class_id = labels[row_i]

            layer_features = model.predict(hog_seq, batch_size=1)

            features = np.array(layer_features).flatten()
            features = np.concatenate([[class_id],features])
            #features = features.reshape(1,features.shape[0])
            # print(features.shape)
            # print(matrix.shape)
            #matrix = np.append(matrix, features, axis=0)
            matrix.append(features)

        matrix = np.array(matrix)
        print(matrix.shape)

        np.savetxt("./out_features_" + layer + ".txt",matrix)
        print("Saved!")

# LOAD YOUR FEATURES

models = []

for layer in layers:
    path = "./out_features_" + layer + ".txt"
    model = np.loadtxt(path)
    models.append(model)

for i in range(len(models)):

  layer = layers[i]
  model = models[i]
  X = model[:,1:]
  y = model[:,0]

  # PCA reducing up to 50 components
  from sklearn import decomposition

  pca = decomposition.PCA(n_components=50)
  pca.fit(X)
  X = pca.transform(X)

  # RUN TSNE

  ############################################################
  # Fit and transform with a TSNE
  from sklearn.manifold import TSNE
  tsne = TSNE(n_components=2, random_state=0)
  # Just 2 components in order show to a 2D plot

  ############################################################
  # Project the data in 2D
  X_2d = tsne.fit_transform(X)

  ############################################################
  # Visualize the data
  target_names = class_labels
  target_ids = range(len(target_names))

  from matplotlib import pyplot as plt
  plt.figure(figsize=(6, 5))
  colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
  for i, c, label in zip(target_ids, colors, target_names):
      plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
  plt.legend()
  plt.title('tSNE layer: ' + layer)


plt.show()