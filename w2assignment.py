import os
import base64
import tensorflow as tf

# Get current working directory
current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

# Normalize pixel values
training_images = training_images / 255.0

# GRADED FUNCTION: create_and_compile_model

def create_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict predict handwriting digits.
    """

    ### START CODE HERE ###
        
    # Define the model
    model = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ]) 

    ### END CODE HERE ###
    
    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

    return model

# Save untrained model in a variable
untrained_model = create_and_compile_model()

# Use it to predict the first 5 images in the train set
predictions = untrained_model.predict(training_images[:5], verbose=False)

print(f"predictions have shape: {predictions.shape}")

# GRADED CLASS: EarlyStoppingCallback

### START CODE HERE ###

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs={}):
        
        # Check if the accuracy is greater or equal to 0.98
        if logs.get('accuracy') >= 0.98:
                            
            # Stop training once the above condition is met
            self.model.stop_training = True
            print("\nReached 98% accuracy so cancelling training!") 
### END CODE HERE ###

# GRADED FUNCTION: train_mnist

def train_mnist(training_images, training_labels):
    """Trains a classifier of handwritten digits.
    Args:
        training_images (numpy.ndarray): The images of handwritten digits
        training_labels (numpy.ndarray): The labels of each image
    Returns:
        tf.keras.callbacks.History : The training history.
    """
    ### START CODE HERE ###
    # Create a compiled (but untrained) version of the model
    # Hint: Remember you already coded a function that does this!
    model = create_and_compile_model()
    
    # Create callback
    callback = EarlyStoppingCallback()
    
    # Fit the model for 10 epochs adding the callbacks and save the training history
    history = model.fit(
        training_images, 
        training_labels, 
        epochs=10,
        callbacks=[callback]
    )
    ### END CODE HERE ###
    return history

training_history = train_mnist(training_images, training_labels)

encoded_answer = "CiAgIC0gQSB0Zi5rZXJhcy5JbnB1dCB3aXRoIHRoZSBzYW1lIHNoYXBlIGFzIHRoZSBpbWFnZXMKICAgLSBBIEZsYXR0ZW4gbGF5ZXIKICAgLSBBIERlbnNlIGxheWVyIHdpdGggNTEyIHVuaXRzIGFuZCBSZUxVIGFjdGl2YXRpb24gZnVuY3Rpb24KICAgLSBBIERlbnNlIGxheWVyIHdpdGggMTAgdW5pdHMgYW5kIHNvZnRtYXggYWN0aXZhdGlvbiBmdW5jdGlvbgo=="
encoded_answer = encoded_answer.encode('ascii')
answer = base64.b64decode(encoded_answer)
answer = answer.decode('ascii')

print(answer)

