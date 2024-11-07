import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json


# Define the process_image function
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    image = image.numpy()
    return image


# Define the predict function
def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_k_probs, top_k_indices = tf.math.top_k(predictions[0], k=top_k)
    top_k_probs = top_k_probs.numpy()
    top_k_indices = top_k_indices.numpy()
    top_k_classes = [str(index) for index in top_k_indices]
    return top_k_probs, top_k_classes


# Define the function to plot the image and predictions
def plot_image_and_predictions(image_path, model, top_k, class_names):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    probs, classes = predict(image_path, model, top_k)

    class_names_predicted = [class_names[class_id] for class_id in classes]

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
    ax1.imshow(test_image)
    ax1.set_title('Original Image')
    ax2.barh(class_names_predicted, probs)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predicted Classes')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the saved model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})


    # Load class names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = {}

    # Plot image and predictions
    plot_image_and_predictions(args.image_path, model, args.top_k, class_names)


if __name__ == "__main__":
    main()
