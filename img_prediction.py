def predict_image(image_path):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Failed to load image.")
        return

    # Resize and convert to 3 channels
    img = cv2.resize(img, (224, 224))
    img = np.stack([img]*3, axis=-1)  # Grayscale to RGB
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img)
    predicted_class = classes[np.argmax(predictions)s]

    # Show the image and prediction
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

