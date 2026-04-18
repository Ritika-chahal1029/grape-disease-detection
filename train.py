import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# dataset path
data_dir = "D:/grape_disease_detection/dataset"

classes = ["black_rot", "esca", "leaf_blight", "healthy"]

data = []
labels = []

img_size = 128

# LOAD IMAGES (SAFE VERSION)
print(" Loading dataset...")

for class_index, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)

    if not os.path.exists(class_path):
        print(" Missing folder:", class_path)
        continue

    files = os.listdir(class_path)
    print(f" {class_name}: {len(files)} images")

    for img_name in files:
        img_path = os.path.join(class_path, img_name)

        try:
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

            data.append(img)
            labels.append(class_index)

        except:
            continue

print(" Total images loaded:", len(data))

# CONVERT TO ARRAY
data = np.array(data)

# EfficientNet preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
data = preprocess_input(data)

labels = to_categorical(labels)

# shuffle
data, labels = shuffle(data, labels, random_state=42)

# split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# MODEL
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB0(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

# fine-tuning
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

# custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# compile
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

datagen.fit(X_train)

# class weights
from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_integers),
    y=y_integers
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# TRAIN
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=12,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weights
)

# GRAPH
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

# accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy")
plt.legend()

# loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("Loss")
plt.legend()

plt.show()

# evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Final Accuracy:", accuracy)

# save model
model.save("model/grape_model.h5")

# REPORT
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=classes))

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()