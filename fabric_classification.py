import os
    class_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = np.stack([img] * 3, axis=-1)  # Grayscale to RGB
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# === Step 2: Build the model with augmentation ===
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

base_model = EfficientNetV2S(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = True  # Unfreeze for fine-tuning

model = Sequential([
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# === Step 3: Compile and train ===
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop, reduce_lr]
)

# === Step 4: Save the trained model ===
model.save("fabric_model.h5")
print("✅ Model saved as fabric_model.h5")

# === Step 5: Evaluate the model ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Final Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

# === Step 6: Plot Accuracy and Loss ===
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


