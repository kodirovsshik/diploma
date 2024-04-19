import tensorflow as tf

image_dir = "E:/backup/archive(1)/images"
num_classes = 15
epochs = 5

class_labels = tf.one_hot(range(num_classes), depth=num_classes)

def decode_image_with_label(filename):
	image = tf.image.decode_png(tf.io.read_file(filename), channels=1)

	split_path = tf.strings.split(filename, sep="/")
	class_folder = split_path[-2]
	label = tf.gather(class_labels, tf.strings.to_number(class_folder, out_type=tf.int32))

	return image, label

train_dataset = tf.data.Dataset.list_files(f"{image_dir}/train/*/*.png").map(decode_image_with_label)
test_dataset = tf.data.Dataset.list_files(f"{image_dir}/test/*/*.png").map(decode_image_with_label)

leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.1)
my_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(1024, 1024, 1)))
model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=leaky_relu, name="conv1"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool1"))
model.add(tf.keras.layers.Conv2D(40, (5, 5), activation=leaky_relu, name="conv2"))
model.add(tf.keras.layers.MaxPooling2D((2, 2), name="pool2"))
model.add(tf.keras.layers.Conv2D(80, (5, 5), activation=leaky_relu, name="conv3"))
model.add(tf.keras.layers.MaxPooling2D((3, 3), name="pool3"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=leaky_relu, name="dense1"))
model.add(tf.keras.layers.Dense(64, activation=leaky_relu, name="dense2"))
model.add(tf.keras.layers.Dense(32, activation=leaky_relu, name="dense3"))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_out"))
model.compile(loss="categorical_crossentropy", optimizer=my_optimizer, metrics=["loss", "accuracy"])

train_loss_history = []
test_loss_history = []

for epoch in range(epochs):
	print(f"--- Epoch {epoch+1} / {epochs} ---")

	history = model.fit(train_dataset, epochs=1)
	loss, accuracy = history.history["loss"][0], history.history["accuracy"][0]

	test_loss, test_accuracy = model.evaluate(test_dataset)

	print(f"Train Loss: {loss:.4f}, Train Accuracy: {accuracy:.4f}")
	print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
	
	train_loss_history.append(loss)
	test_loss_history.append(test_loss)

print("Training complete!")
