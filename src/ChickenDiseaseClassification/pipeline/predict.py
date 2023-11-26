import numpy as np
import tensorflow as tf
import os


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    # def predict(self):
    #     # load model
    #     model = tf.keras.models.load_model(os.path.join("artifacts","training", "model.h5"))

    #     imagename = self.filename
    #     test_image = tf.keras.preprocessing.image.load_img(imagename, target_size = (224,224))
    #     test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    #     test_image = np.expand_dims(test_image, axis = 0)
    #     result = np.argmax(model.predict(test_image), axis=1)
    #     print(result)

    #     if result[0] == 1:
    #         prediction = 'Healthy'
    #         return [{ "image" : prediction}]
    #     # elif result[1] == 1:
    #     #     prediction = 'Coccidiosis'
    #     #     return [{ "image" : prediction}]
    #     # elif result[2] == 1:
    #     #     prediction = 'New Castle Disease'
    #     #     return [{ "image" : prediction}]
    #     # elif result[3] == 1:
    #     #     prediction = 'Coccidiosis'
    #     #     return [{ "image" : prediction}]
    #     # elif result[4] == 1:
    #     #     prediction = 'Healthy'
    #     #     return [{ "image" : prediction}]
    #     # elif result[5] == 1:
    #     #     prediction = 'New Castle Disease'
    #     #     return [{ "image" : prediction}]
    #     # elif result[6] == 1:
    #     #     prediction = 'Salmonella'
    #     #     return [{ "image" : prediction}]
    #     else:
    #         prediction = 'Salmonella'
    #         return [{ "image" : prediction}]
import os
import numpy as np
import tensorflow as tf

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = tf.keras.models.load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size=(224, 224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Get the predicted probabilities for each class
        predictions = model.predict(test_image)

        # Get the class index with the highest probability
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Mapping of class indices to class labels
        class_labels = {
            0: 'Coccidiosis',
            1: 'Healthy',
            2: 'New Castle Disease',
            3: 'Salmonella',
    
            # Add more mappings as needed
        }

        # Get the predicted class label
        predicted_label = class_labels.get(predicted_class_index, 'Unknown Class')
        print(predicted_class_index)
        print(predicted_label)

        return [{"image": predicted_label}]

