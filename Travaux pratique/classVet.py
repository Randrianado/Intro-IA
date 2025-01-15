import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Prétraitement des données
x_train = x_train / 255.0  # Normalisation des images
x_test = x_test / 255.0

# Encodage des labels en one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Noms des catégories
class_names = ['T-shirt/Top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
               'Sandales', 'Chemise', 'Bottes', 'Sac', 'Baskets']

# 3. Affichage des premières images avec les noms des classes
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.ravel()

for i in np.arange(0, 9):
    axes[i].imshow(x_train[i], cmap='gray')
    axes[i].set_title(f"Label: {class_names[np.argmax(y_train[i])]}")  # Afficher le nom de l'objet
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()

# 4. Créer le modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 6. Évaluer le modèle
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.2f}")

# 7. Faire une prédiction
predictions = model.predict(x_test)
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Prédiction: {class_names[np.argmax(predictions[0])]}, Réel: {class_names[np.argmax(y_test[0])]}")  # Affichage avec le nom de l'objet
plt.show()
