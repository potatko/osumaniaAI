import cv2
import numpy as np


def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Resize to 84x84
    resized = cv2.resize(thresh, (84, 84), interpolation=cv2.INTER_AREA)

    # Reshape to 84x84x1
    reshaped = np.reshape(resized, (84, 84, 1))

    return reshaped


def preprocess_data(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60.0, (84, 84), isColor=False)

    frames = []
    with open(video_path + '.txt', 'r') as f:
        key_presses = [line.split() for line in f]

    # Get the frame rate and calculate the number of key presses per frame
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    keys_per_frame = [[] for i in range(int(frame_rate / 60))]

    for key in key_presses:
        time, key_id = key
        frame_num = int(float(time) * frame_rate)

        if frame_num >= len(keys_per_frame):
            break

        keys_per_frame[frame_num].append(int(key_id))

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            if frame_num % int(frame_rate / 60) == 0:
                keys = keys_per_frame[frame_num // int(frame_rate / 60)]

                if len(keys) > 0:
                    frames.append(preprocess_frame(frame))

                for key in keys:
                    out.write(np.array([key], dtype=np.uint8))

            frame_num += 1
        else:
            break

    cap.release()
    out.release()

    X = np.array(frames)
    y = np.loadtxt(output_path, dtype=np.uint8)

    return X, y
