import numpy as np
import cv2
from preprocessing import preprocess_data
from model import build_model, train_model
from agent import Agent

# Collect training data
video_path = "<path-to-video>"
output_path = "<output-directory-path>"
preprocess_data(video_path, output_path)

# Preprocess data
data_path = "<preprocessed-data-path>"
frames = np.load(data_path)

# Train model
input_shape = frames.shape[1:]
num_keys = 4 # or however many keys there are in your osu!mania game
model = build_model(input_shape, num_keys)
train_model(model, X_train, y_train, X_val, y_val)

# Save model
model_path = "<path-to-save-model>"
model.save(model_path)

# Play game
model_path = "<path-to-saved-model>"
agent = Agent(model_path)
# use agent.act(observation) to get the next action to take in the game
