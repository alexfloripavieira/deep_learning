import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

train_dir = "/home/alex/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/train"
test_dir = "/home/alex/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test"

# Aumento de dados para o conjunto de treinamento
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Carregamento de imagens de treinamento e teste
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=batch_size, class_mode="binary"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=batch_size, class_mode="binary"
)

# Carregamento do modelo VGG16 pré-treinado
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# Construção do modelo personalizado
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Congelar camadas convolucionais do modelo base (VGG16)
base_model.trainable = False

# Compilação do modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_generator),
)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Acurácia no conjunto de teste: {test_acc * 100:.2f}%")

# Exibição da matriz de confusão e relatório de classificação
test_labels = test_generator.classes
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_labels = (predictions > 0.5).astype(int)

conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Matriz de Confusão:")
print(conf_matrix)

class_report = classification_report(
    test_labels, predicted_labels, target_names=["Normal", "Pneumonia"]
)
print("Relatório de Classificação:")
print(class_report)

# Plotagem da curva de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Acurácia no treinamento")
plt.plot(history.history["val_accuracy"], label="Acurácia na validação")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Perda no treinamento")
plt.plot(history.history["val_loss"], label="Perda na validação")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.legend()

plt.tight_layout()
plt.show()
