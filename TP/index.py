import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# 1. Chargement des données depuis un dossier local
def load_mnist_images(file_path):
 with open(file_path, 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8)
 return data

# Chemins vers les fichiers locaux
train_images_file = "../mnist_data/train-images.idx3-ubyte"
train_labels_file = "../mnist_data/train-labels.idx1-ubyte"
test_images_file = "../mnist_data/t10k-images.idx3-ubyte"
test_labels_file = "../mnist_data/t10k-labels.idx1-ubyte"

# Charger les images et étiquettes
train_images = load_mnist_images(train_images_file)[16:].reshape(-1, 28, 28) # Les 16 premiersbytes sont des en-têtes
train_labels = load_mnist_images(train_labels_file)[8:] # Les 8 premiers bytes sont des en-têtes
test_images = load_mnist_images(test_images_file)[16:].reshape(-1, 28, 28)
test_labels = load_mnist_images(test_labels_file)[8:]

# 2. Prétraitement des images
# Normalisation pour mettre les pixels dans la plage [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Conversion des labels en one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Ajouter une dimension de canal (pour correspondre au format attendu par les CNN)
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# 3. Affichage d'une image du jeu de données
# Afficher la première image de l'ensemble d'entraînement
fig, axes = plt.subplots(3, 3, figsize=(9, 9))  # 3x3 sous-graphes
axes = axes.ravel()
for i in np.arange(0, 9):  # Afficher 9 images
    axes[i].imshow(train_images[i].reshape(28, 28), cmap='gray')  # Reshaper pour afficher 28x28
    axes[i].set_title(f"Label: {np.argmax(train_labels[i])}")  # Afficher l'étiquette
    axes[i].axis('off')  # Ne pas afficher les axes

plt.subplots_adjust(wspace=0.5)
plt.show()

# 3. Création du modèle TensorFlow (CNN)
model = models.Sequential([
 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
 layers.MaxPooling2D((2, 2)),
 layers.Conv2D(64, (3, 3), activation='relu'),
 layers.MaxPooling2D((2, 2)),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# 4. Entraînement du modèle
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# 5. Évaluation et affichage des résultats
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_accuracy:.2f}")