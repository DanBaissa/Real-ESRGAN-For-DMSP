
print("Torch Results:")
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())  # Prints the current GPU ID
print(torch.cuda.get_device_name(0))  # Prints the name of the current GPU

print("TensorFlow Results:")
import tensorflow as tf

# Check if TensorFlow can access GPU
gpu_available = tf.config.list_physical_devices('GPU')
print("GPU Available:", bool(gpu_available))

if gpu_available:
    # Get the name of the current GPU
    gpu_name = tf.config.experimental.get_device_details(gpu_available[0])['device_name']
    print("Current GPU Device Name:", gpu_name)
else:
    print("No GPU available.")

# Check if TensorFlow is detecting any GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

# If GPUs are detected, print their details
if gpus:
    for gpu in gpus:
        print(gpu)