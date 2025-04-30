"""
Face Detection using OpenCV DNN Module

This script detects faces using a pre-trained deep learning model (Caffe-based).
It works with a webcam or video file.

Requirements:
- deploy.prototxt
- res10_300x300_ssd_iter_140000_fp16.caffemodel
- opencv-python
- opencv-contrib-python (for dnn module)

To run:
    python face_detect.py              # Uses webcam
    python face_detect.py video.mp4   # Uses video file

Name: Naheed Anjum
SID: 21104002
"""

import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve


def download_and_unzip(url, save_path):
    """Download and unzip the model files if not already present."""
    print("Downloading and extracting assets...", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.dirname(save_path))
        print("Done.")
    except Exception as e:
        print("\nInvalid ZIP file:", e)


def main():
    # Download model assets if not present
    URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_12.zip")
    if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)

    # Check model files
    model_txt = "deploy.prototxt"
    model_bin = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    if not os.path.exists(model_txt) or not os.path.exists(model_bin):
        print("Required model files not found.")
        sys.exit(1)

    # Load input source (webcam by default or video file from argv)
    source_input = 0
    if len(sys.argv) > 1:
        source_input = sys.argv[1]

    source = cv2.VideoCapture(source_input)
    if not source.isOpened():
        print("Cannot open video source.")
        sys.exit(1)

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    # Load DNN model
    net = cv2.dnn.readNetFromCaffe(model_txt, model_bin)

    # Set model input details
    in_width, in_height = 300, 300
    mean = [104, 117, 123]
    conf_threshold = 0.7

    while cv2.waitKey(1) != 27:  # ESC key to exit
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        # Create blob & run inference
        blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Draw detection boxes
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Confidence: {confidence:.4f}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Show inference time
        t, _ = net.getPerfProfile()
        fps_label = f"Inference time: {t * 1000.0 / cv2.getTickFrequency():.2f} ms"
        cv2.putText(frame, fps_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv2.imshow("Camera Preview", frame)

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
