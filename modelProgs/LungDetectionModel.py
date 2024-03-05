import keras
import tensorflow as tf

BATCH_SIZE = 16

print("\nLoading training data...")

training_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05)

training_iterator = training_data_generator.flow_from_directory(
    'chest_xray/train',
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE)

print("\nLoading validation data...")

validation_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_iterator = validation_data_generator.flow_from_directory(
    'chest_xray/test',
    class_mode='categorical',
    color_mode='grayscale',
    batch_size=BATCH_SIZE)

print("\nBuilding model...")

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))
model.add(tf.keras.layers.Conv2D(2, 7, strides=1, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.summary()

print("\nCompiling model...")

model.compile(
   optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005),
   loss=tf.keras.losses.CategoricalCrossentropy(),
   metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

print("\nDone.")

print("\nTraining model...")

model.fit(
       training_iterator,
       steps_per_epoch=training_iterator.samples/BATCH_SIZE,
       epochs=5,
       validation_data=validation_iterator,
       validation_steps=validation_iterator.samples/BATCH_SIZE)

model.save('/Users/hrishid/TensorFlowProjects/models/lungModel.keras')