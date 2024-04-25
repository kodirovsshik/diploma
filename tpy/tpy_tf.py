
import colorama, os, glob, PIL
colorama.init()

import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Input

def count_items_in_dir(path):
	return len([name for name in os.listdir(path)])

def get_dataset_size(root):
	_num_classes_train = count_items_in_dir(root + "/train")
	_num_classes_test = count_items_in_dir(root + "/test")
	assert _num_classes_train == _num_classes_test, "train != test"
	return _num_classes_train

dataset_root = "C:/dataset_pneumonia/png8"
num_classes = get_dataset_size(dataset_root)

class_labels = tf.one_hot(range(num_classes), depth=num_classes)
input_img_size = 256
max_epochs = 1

# def to_png():
# 	for img in glob.glob("C:/dataset_pneumonia/bmp/train/*/*.bmp"):
# 		PIL.Image.open(img).save(img.replace("bmp", "png"))
# to_png()
# exit(0)
# import cv2
# def convert_to_grayscale(image_path, output_path):
# 	image = cv2.imread(image_path)
# 	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	cv2.imwrite(output_path, grayscale_image)

# image_dir = dataset_root + "/train/1"
# for filename in os.listdir(image_dir):
# 	if filename.endswith(".png"):
# 		image_path = os.path.join(image_dir, filename)
# 		convert_to_grayscale(image_path, image_path.replace("png32", "png8"))

# exit(0)


def load_dataset(data_dir):
	def parse_image(filename):
		image_string = tf.io.read_file(filename)
		image = tf.image.decode_png(image_string)
		image = tf.cast(image, tf.float32) / 255.0
		image = tf.reshape(image, shape=(-1,input_img_size,input_img_size,1))
		label_folder = tf.strings.split(filename, sep='\\')[-2]
		label = tf.strings.to_number(label_folder, out_type=tf.int32)
		label = tf.gather(class_labels, label)
		label = tf.reshape(label, shape=(1, num_classes))
		
		return image, label

	file_paths = tf.io.gfile.glob(data_dir + "/*/*.png") 
	dataset = tf.data.Dataset.from_tensor_slices(file_paths)
	dataset = dataset.map(parse_image)

	return dataset

train_dataset = load_dataset(dataset_root + "/train")
test_dataset = load_dataset(dataset_root + "/test")

from keras import backend as K
K.set_image_data_format("channels_last")

leaky_relu = "relu"
my_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0001)

model_name = "model2"

model_path = dataset_root + "/" + model_name + ".keras"
print("Loading", model_name, "from", model_path)
try:
	model = tf.keras.models.load_model(model_path)
	print("Model loaded")
except ValueError:
	print("No model found - creating new model")
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Input(shape=(input_img_size, input_img_size, 1)))
	model.add(tf.keras.layers.Conv2D(10, (5, 5), activation=leaky_relu, data_format='channels_last'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
	model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=leaky_relu, data_format='channels_last'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
	model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=leaky_relu, data_format='channels_last'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
	model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=leaky_relu, data_format='channels_last'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
	model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=leaky_relu, data_format='channels_last'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(32, activation=leaky_relu))
	model.add(tf.keras.layers.Dense(num_classes, activation=leaky_relu))
	model.compile(loss="mse", optimizer=my_optimizer, metrics=["accuracy"])

print(model.summary())

def get_test_evaluation():
	test_loss, test_accuracy = model.evaluate(test_dataset)
	print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

get_test_evaluation()
history = model.fit(x=train_dataset, validation_data=test_dataset, epochs=1, steps_per_epoch=1000)
get_test_evaluation()

model.save(model_path)
print(history.history["loss"])
print(history.history["val_loss"])
