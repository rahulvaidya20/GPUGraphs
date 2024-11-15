import sys, argparse
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json  # Import model_from_json
import numpy as np
from tensorflow.keras import layers
import pandas as pd
import time, uuid
from tensorflow.keras.callbacks import Callback
from pathlib import Path
import os

DEBUG = False

to_seconds = lambda x: round(float(x), 2)

class StepTimeLogger(Callback):
    def __init__(self):
        super().__init__()
        self.batch_times = []  # List to store batch times
        self.epoch_times = []  # List to store epoch times
        self.total_training_time = 0.0  # Initialize total training time
        self.per_epoch_time = 0.0  # Initialize per epoch time
        self.per_batch_time = 0.0  # Initialize per batch time

    def on_train_begin(self, logs=None):
        # Record the start time at the beginning of training
        self.training_start_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        # Record the start time of the batch
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        # Calculate the time taken for the batch
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)  # Store the time in the list

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the time taken for the epoch
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)  # Store the time in the list

    def on_train_end(self, logs=None):
        # Calculate the total training time
        self.total_training_time = time.time() - self.training_start_time
        self.per_epoch_time = np.mean(self.epoch_times)
        self.per_batch_time = np.mean(self.batch_times)


def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process a file and a number.")

    # Add arguments
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the the model summary file")
    parser.add_argument("-o", "--opt", type=str, required=True, help="Provide optimizer adam, sgd")
    parser.add_argument("-n", "--samples", type=int, required=True, help="provide no of traning samples")
    parser.add_argument("-e", "--epoch", type=int, required=True, help="provide no of epochs")
    parser.add_argument("-b", "--batch", type=int, required=True, help="provide batch size (integer) ")
    parser.add_argument("-l", "--lr", type=float, required=True, help="provide learning rate (0-1 float)")
    parser.add_argument("-c", "--csv", type=str, required=True, help="Csv file path to log model data")

    args = None
    # Parse arguments
    try:
        args = parser.parse_args()

        # Check if file exists (throws an error if it doesn't)
        with open(args.file, 'r') as file:
            if DEBUG:
                print(f"Successfully opened file: {args.file}")

        # Check if the number is positive
        if args.opt not in  ["adam", "sgd"]:
            raise ValueError("optimizer should be 'adam' or 'opt' only!.")

    except FileNotFoundError:
        print(f"Error: The file '{args.file}' does not exist.", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        print(f"Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    return args


    
def load_model_from_json(json_file_name, optimizer_name, learning_rate=0.001):
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
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def preapre_training_data(no_of_samples, image_dim):
    """
    Generate a random image tensor with the given image dimensions.
    :param image_dim: Tuple (Width, Height, Channels)
    :return: A random numpy array of shape (1, Width, Height, Channels)
    """
    image_data = np.random.randint(0, 256, (no_of_samples, image_dim[0], image_dim[1], image_dim[2]), dtype=np.uint8)
    image_data = image_data / 255.0
    return image_data


def preapre_training_labels(no_of_samples, no_of_classes):
    """
    Generate a random image tensor with the given image dimensions.
    :param image_dim: Tuple (Width, Height, Channels)
    :return: A random numpy array of shape (1, Width, Height, Channels)
    """
    label_data = np.random.randint(0, no_of_classes, size=(no_of_samples,), dtype=np.uint8)
    return label_data


# Step 6: Perform dummy training on the model
def train_model(model, image_dim, no_of_samples=1, no_of_epochs=1, batch_size=1, callbacks=[]):
    """
    Perform a dummy training to verify the model works properly.
    :param model: Keras model to train
    :param image_dim: Tuple (Width, Height, Channels)
    :param num_classes: Number of classes for the output layer
    :param epochs: Number of training epochs (default: 1)
    """
    # Generate random data for training
    # x_train = np.random.rand(10, image_dim[1], image_dim[0], image_dim[2])
    # y_train = np.random.randint(num_classes, size=(10,))
    training_data_X = preapre_training_data(no_of_samples, image_dim)
    training_data_Y = preapre_training_labels(no_of_samples, output_classes)
    model.fit(training_data_X, training_data_Y, epochs=no_of_epochs, batch_size=batch_size, callbacks=callbacks)


def write_to_csv(config_dict, batch_time, epoch_time, fit_time, csv_file, tf_version, cuda_version):
    # model_trained  = "_".join(map(str, name))
    model_trained = "_".join(
        #f"{value}" if key in ["mod", "uid"] else f"{key}{value}"
        #for key, value in config_dict.items()
        f"{value}" if key in ["mod", "uid"] else f"{key}{str(value - int(value))[2:] if isinstance(value, float) and key == 'lr' else value}"
        for key, value in config_dict.items()
    )

    ipdim =  config_dict["ipd"].split("x")
    
    data = {
        'name': model_trained,
        'samples': config_dict["s"],
        'input_dim_w': int(ipdim[0]),
        'input_dim_h': int(ipdim[1]),
        'input_dim_c': int(ipdim[2]),
        'output_dim': config_dict["opd"],
        'epochs': config_dict["e"],
        'batch': config_dict["b"],
        'learn_rate': config_dict["lr"],
        "tf_version": tf_version,
        "cuda_version": cuda_version,
        'batch_time': batch_time,
        'epoch_time': epoch_time,
        'fit_time': fit_time
    }
    df = pd.DataFrame(data, index=[0])
    file_exists = os.path.isfile(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=not file_exists)

# Run the function to save and load model architecture
if __name__ == "__main__":
    # put the parsing code here
    parse_args = parse_arguments()
    model = load_model_from_json(parse_args.file, parse_args.opt, learning_rate=parse_args.lr)

    trainable_params = np.sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(w.shape) for w in model.non_trainable_weights])

    input_dimensionality = model.input_shape 
    output_classes = model.output_shape[-1]
    input_dimn = (input_dimensionality[1], input_dimensionality[2], input_dimensionality[3])
   
    if DEBUG:
        print(model.summary())

    tf_version = tf.__version__
    cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'NA')
    # Initialize the callback
    step_time_logger = StepTimeLogger() 
    train_model(model, input_dimn, no_of_samples=parse_args.samples, no_of_epochs=parse_args.epoch, batch_size=parse_args.batch,  callbacks=[step_time_logger])

    unique_number = int(str(uuid.uuid4().int)[:4])
    file_name = Path(parse_args.file).stem
    if DEBUG: 
        print("FILE NAME:", file_name)  # Output: file.txt
    input_dimn_str = 'x'.join(map(str, [input_dimensionality[1], input_dimensionality[2], input_dimensionality[3]]))

    config_dict = {
        "mod": file_name,
        "opt": parse_args.opt,
        "s": parse_args.samples,
        "ipd": input_dimn_str,
        "opd": output_classes,
        "e": parse_args.epoch,
        "b": parse_args.batch,
        "lr": parse_args.lr,
        "uid": unique_number
    }
    
    write_to_csv(config_dict, to_seconds(step_time_logger.per_batch_time), to_seconds(step_time_logger.per_epoch_time), to_seconds(step_time_logger.total_training_time), parse_args.csv, tf_version, cuda_version)
    if DEBUG: 
    # Access batch times after training
        print("Batch times list:", step_time_logger.batch_times)
        print(f"Total training time: {step_time_logger.total_training_time:.2f} seconds")
        print(f"Per epoch time: {step_time_logger.per_epoch_time:.2f} seconds")
        print(f"Per batch time: {step_time_logger.per_batch_time:.2f} seconds")
