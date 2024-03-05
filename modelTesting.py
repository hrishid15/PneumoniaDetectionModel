from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
import numpy as np

model = load_model('models/lungModel.keras')


image_path = 'bad1.jpeg'
img_to_disp = imageio.v2.imread(image_path)

# Load and preprocess the new image
new_image = image.load_img(image_path, target_size=(256, 256))
new_image = image.img_to_array(new_image)
grayscale_image = (new_image[:,:,0] + new_image[:,:,1] + new_image[:,:,2]) / 3
grayscale_image = grayscale_image / 255.0  # Assuming normalization during training
grayscale_image = np.expand_dims(grayscale_image, axis=0)  # Add batch dimension


# Make a prediction
prediction = model.predict(grayscale_image)

# Interpret the results
predicted_class = np.argmax(prediction)  # Assuming highest probability wins

print("\n\n\n")

if predicted_class == 0:
    print("Predicted class: Normal")
elif predicted_class == 1:
    print("Predicted class: Pneumonia")

print("\n\n\n")

plt.imshow(img_to_disp)
plt.show()


