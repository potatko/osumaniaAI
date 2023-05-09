import numpy as np
import cv2
from tensorflow.keras.models import load_model

class Agent:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def act(self, obs):
        # perform necessary preprocessing on the observation
        obs = preprocess_frame(obs)
        obs = np.expand_dims(obs, axis=0)

        # make prediction using the model
        action_prob = self.model.predict(obs)[0]
        action = np.argmax(action_prob)

        return action
