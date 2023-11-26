import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from ChickenDiseaseClassification.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    
    def get_base_model(self):
        self.model = tf.keras.applications.efficientnet.EfficientNetB3(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            pooling='max'
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # if freeze_all:
        #     for layer in model.layers:
        #         model.trainable = False
        # elif (freeze_till is not None) and (freeze_till > 0):
        #     for layer in model.layers[:-freeze_till]:
        #         model.trainable = False

        # flatten_in = tf.keras.layers.Flatten()(model.output)
        # prediction = tf.keras.layers.Dense(
        #     units=classes,
        #     activation="softmax"
        # )(flatten_in)

        # full_model = tf.keras.models.Model(
        #     inputs=model.input,
        #     outputs=prediction
        # )

        # full_model.compile(
        #     optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=["accuracy"]
        # )

        # full_model.summary()
        # return full_model
    
        full_model = tf.keras.Sequential([
            model,
            tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.016), activity_regularizer=tf.keras.regularizers.l1(0.006),
                  bias_regularizer=tf.keras.regularizers.l1(0.006), activation='relu'),
            tf.keras.layers.Dropout(rate=0.45, seed=123),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
