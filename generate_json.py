import importlib
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json  # Import model_from_json
import json
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
 
# Step 1: Dynamic model loader function
def load_model_dynamically(model_name):
    """
    Dynamically load the model from Keras Applications based on the model name.
    :param model_name: The name of the Keras model to load (e.g., 'VGG16', 'ResNet50')
    :return: The Keras model class ready to be instantiated
    """
    try:
        # Dynamically import the model using importlib
        module = importlib.import_module(f"tensorflow.keras.applications")
        model_class = getattr(module, model_name)
        return model_class
    except AttributeError:
        raise ValueError(f"Model {model_name} is not available in Keras Applications.")
 
# Step 2: Function to generate random images based on the given dimensions
def generate_random_image(image_dim):
    """
    Generate a random image tensor with the given image dimensions.
    :param image_dim: Tuple (Width, Height, Channels)
    :return: A random numpy array of shape (1, Width, Height, Channels)
    """
    return np.random.rand(1, image_dim[1], image_dim[0], image_dim[2])
 
# Step 3: Function to load and compile a model dynamically with a classification head
def load_and_compile_model(model_name, image_dim, optimizer_name, num_classes=10):
    """
    Load and compile a pre-trained model with the given parameters and add a classification head.
    :param model_name: Name of the model to load
    :param image_dim: Tuple (Width, Height, Channels)
    :param optimizer_name: Name of the optimizer (e.g., 'adam', 'sgd')
    :param num_classes: Number of output classes for the classification task
    :return: Compiled Keras model
    """
    # Dynamically load the model class
    model_class = load_model_dynamically(model_name)
 
    # Load the model without the top layer (for customization)
    base_model = model_class(weights='imagenet', input_shape=image_dim, include_top=False)
 
    # Add a classification head to match the target shape
    x = layers.Flatten()(base_model.output)  # Flatten the output
    x = layers.Dense(128, activation='relu')(x)  # Add a fully connected layer (optional)
    output = layers.Dense(num_classes, activation='softmax')(x)  # Output layer for classification
 
    # Create a new model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
 
    # Compile the model with the provided optimizer
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD()
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
 
    # Compile the model for classification tasks
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# Step 4: Save model architecture to a human-readable JSON file
def save_model_architecture_to_json(model, file_name):
    """
    Save the Keras model's architecture in JSON format in a human-readable way.
    :param model: Keras model to save
    :param file_name: Name of the JSON file to save the architecture
    """
    # Convert the model architecture to JSON format
    model_json = model.to_json()
 
    # Save JSON with indentation for human readability
    with open(file_name, "w") as json_file:
        json.dump(json.loads(model_json), json_file, indent=4)
    print(f"Model architecture saved to {file_name}")
 
# Step 5: Load model from a saved JSON file
def load_model_from_json(json_file_name, optimizer_name):
    """
    Load a model from a JSON file and compile it.
    :param json_file_name: Path to the saved model architecture JSON file
    :param optimizer_name: Optimizer to compile the model with (e.g., 'adam', 'sgd')
    :return: Loaded and compiled Keras model
    """
    # Load the JSON file containing the model architecture
    with open(json_file_name, "r") as json_file:
        model_json = json_file.read()
 
    # Rebuild the model from the loaded architecture
    model = model_from_json(model_json)
 
    # Re-compile the model (the optimizer and loss are not saved in the JSON)
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam()
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD()
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"Model loaded and compiled from {json_file_name}")
    return model
 
# Step 6: Perform dummy training on the model
def dummy_train(model, image_dim, num_classes=10, epochs=1):
    """
    Perform a dummy training to verify the model works properly.
    :param model: Keras model to train
    :param image_dim: Tuple (Width, Height, Channels)
    :param num_classes: Number of classes for the output layer
    :param epochs: Number of training epochs (default: 1)
    """
    # Generate random data for training
    x_train = np.random.rand(10, image_dim[1], image_dim[0], image_dim[2])
    y_train = np.random.randint(num_classes, size=(10,))
 
    # Perform dummy training
    print("Starting dummy training...")
    model.fit(x_train, y_train, epochs=epochs)
    print("Dummy training complete.")
 
# Step 7: Main function to collect model architectures, save to JSON, and reload
def generate_and_load_model_json():
    """
    Collect architectures for all Keras Applications models and perform dummy training on them.
    The architectures are saved in JSON files.
    """
    # List of available models and their default input dimensions
    available_models = {
        "VGG16": (224, 224, 3),
        "VGG19": (224, 224, 3),
        "ResNet50": (224, 224, 3),
        "ResNet101": (224, 224, 3),
        "ResNet152": (224, 224, 3),
        "InceptionV3": (299, 299, 3),
        "Xception": (299, 299, 3),
        "MobileNet": (224, 224, 3),
        "MobileNetV2": (224, 224, 3),
        "DenseNet121": (224, 224, 3),
        "DenseNet169": (224, 224, 3),
        "DenseNet201": (224, 224, 3),
        "NASNetMobile": (224, 224, 3),
        "NASNetLarge": (331, 331, 3),
        "EfficientNetB0": (224, 224, 3),
        "EfficientNetB1": (240, 240, 3),
        "EfficientNetB7": (600, 600, 3)
    }
 
    # Iterate over all models in Keras Applications
    for model_name, image_dim in available_models.items():
        optimizer_name = "adam"  # You can change this to 'sgd' or other supported optimizers
        print(f"Generating architecture for {model_name} with input dimensions {image_dim}...")
 
        # Load and compile the model
        model = load_and_compile_model(model_name, image_dim, optimizer_name)
 
        # Save model architecture to a JSON file
        json_file_name = f"{model_name}_architecture.json"
        save_model_architecture_to_json(model, json_file_name)
        # Load the model back from JSON
        # loaded_model = load_model_from_json(json_file_name, optimizer_name)
 
        # Perform dummy training on the loaded model
        # dummy_train(loaded_model, image_dim)
 
# Run the function to save and load model architecture
if __name__ == "__main__":
    generate_and_load_model_json()