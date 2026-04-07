import tensorflow as tf

print(tf.__version__)  # should be 2.10

model = tf.keras.models.load_model("your_model.h5")
model.save("model_fixed.h5")