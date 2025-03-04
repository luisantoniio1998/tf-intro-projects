import numpy as np
import tensorflow as tf

# GRADED FUNCTION: create_training_data

def create_training_data():

    ### START CODE HERE ###
    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms. 
    # For this exercise, please arrange the values in ascending order (i.e. 1, 2, 3, and so on).
    # Hint: Remember to explictly set the dtype as float when defining the numpy arrays
    n_bedrooms = np.array([1,2,3,4,5,6], dtype=float)
    price_in_hundreds_of_thousands = np.array([1.0,1.5,2.0,2.5,3.0,3.5], dtype=float)

    ### END CODE HERE ###

    return n_bedrooms, price_in_hundreds_of_thousands

features, targets = create_training_data()

print(f"Features have shape: {features.shape}")
print(f"Targets have shape: {targets.shape}")

# GRADED FUNCTION: define_and_compile_model

def define_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict house prices.
    """
    
    ### START CODE HERE ###

    # Define your model
    model = tf.keras.Sequential([ 
		# Define the Input with the appropriate shape
		tf.keras.Input(shape=(1,)),
		# Define the Dense layer
		tf.keras.layers.Dense(units=1)
	]) 
    
    # Compile your model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    ### END CODE HERE ###

    return model

untrained_model = define_and_compile_model()

untrained_model.summary()

# GRADED FUNCTION: train_model

def train_model():
    """Returns the trained model.

    Returns:
        tf.keras.Model: The trained model that will predict house prices.
    """

    ### START CODE HERE ###

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember you already coded a function that does this!
    n_bedrooms, price_in_hundreds_of_thousands = create_training_data()
    
    # Define a compiled (but untrained) model
    # Hint: Remember you already coded a function that does this!
    model = define_and_compile_model()
    
    # Train your model for 500 epochs by feeding the training data
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)
    
    ### END CODE HERE ###

    return model

# Get your trained model
trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")


