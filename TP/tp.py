import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Étape 2 : Charger et préparer les données
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalisation des images (mise à l'échelle des pixels entre 0 et 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Redimensionner les images pour qu'elles aient une dimension de canal
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Convertir les labels en one-hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Étape 3 : Créer le modèle CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes pour les chiffres 0 à 9
])

# Étape 4 : Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Utiliser categorical_crossentropy pour les labels one-hot
              metrics=['accuracy'])


# Étape 5 : Entraîner le modèle
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Étape 6 : Évaluer le modèle
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy: {test_acc}')
