import zipfile
import os
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a tensorboard callback with format: dir_name/experiment_name/%Y%m%d - %H%M%S

  Args:
    dir_name: direction name string of the main folder that all tensorboard callbacks are going to be
    experiment name: especific name of the experiment (model) where all the logs are going to be

  Returns:
    tensorboard_callback: Give a call back with the format specify
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d - %H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def unzip_data(zip_name):
   """
   Unzip the folder from the zip name

   Args:
    zip_name: zip name string that is alredy download
   """

   zip_ref = zipfile.ZipFile(zip_name)
   zip_ref.extractall()
   zip_ref.close()


def walk_trough_dir(folder_directory):
   """
   Walk through folder directory and list number of files

   Args:
    folder_directory: Folder directory string you want to walk_trough
   """
   for dirpath, dirnames, filenames in os.walk(folder_directory):
      print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")

def plot_loss_curves(history):
  """
  Returns separete loss curves for training and validation metrics.

  Args:
    history: TensorFlow History object.

  Returns:
    Plots of training/validation loss and accuracy metrics.
  """
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"])) # Get how many epochs

  # Plot loss
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()