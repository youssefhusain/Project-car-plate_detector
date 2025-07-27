
from transformers import pipeline
qwen_pipe = pipeline("image-text-to-text", model="Qwen/Qwen2.5-VL-3B-Instruct")



import string
import cv2
import torch
import numpy as np
import json
import os
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import easyocr
from PIL import Image



# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def write_json(results, output_path):
    """Write only frames with license plates to a JSON file."""
    json_results = {}
    for frame_nmr in results:
        filtered_frame_data = {}

        for car_id in results[frame_nmr]:

            if car_id == 'scene_description':
                continue

            car_data = results[frame_nmr][car_id]

            if isinstance(car_data, dict) and 'license_plate' in car_data:
                filtered_frame_data[car_id] = {
                    'car': {
                        'bbox': car_data['car']['bbox']
                    },
                    'license_plate': {
                        'bbox': car_data['license_plate']['bbox'],
                        'text': car_data['license_plate']['text'],
                        'bbox_score': float(car_data['license_plate']['bbox_score']),
                        'text_score': float(car_data['license_plate']['text_score'])
                    }
                }


        if filtered_frame_data:
            filtered_frame_data['scene_description'] = results[frame_nmr].get('scene_description', "")
            json_results[frame_nmr] = filtered_frame_data

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=4)


def license_complies_format(text):
    """Check if the license plate text complies with the required format."""
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    return False

def format_license(text):
    """Format the license plate text by converting characters using the mapping dictionaries."""
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char,
               5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int,
               3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop):
    """Read the license plate text from the given cropped image."""
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """Retrieve the vehicle coordinates and ID based on the license plate coordinates."""
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]
    return -1, -1, -1, -1, -1

def linear_assignment(cost_matrix):
    """Solve the linear assignment problem using Hungarian algorithm."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """Compute IOU between two sets of bounding boxes."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
        (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """Convert bounding box to [x,y,s,r] format."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert [x,y,s,r] format to [x1,y1,x2,y2] format."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

class KalmanBoxTracker:
    """Track individual objects using Kalman Filter."""
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update the state vector with observed bbox."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Advance the state vector and return the predicted bounding box."""
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return the current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Assign detections to tracked objects."""
    if len(trackers) == 0:
        return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    """SORT tracker implementation."""
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """Update tracker with new detections."""
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

def process_frame(frame, frame_nmr, coco_model, license_plate_detector, mot_tracker):
    """Process a single frame for license plate recognition."""
    results = {}
    results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in [2, 3, 5, 7]:  # vehicle classes
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }
    return results

question = "Describe the traffic scene in detail."



def get_scene_description(pipe, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    try:
        result = pipe(messages)
        return result[0]["generated_text"][1]["content"]
    except Exception as e:
        return f"Error generating description: {e}"

def main():
    video_path = '/content/drive/MyDrive/Colab Notebooks/WhatsApp Video 2025-07-27 at 1.16.07 AM.mp4'
    output_path = 'results.json'

    # Initialize tracker
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('/content/drive/MyDrive/Colab Notebooks/yolov8n.pt')
    license_plate_detector = YOLO('/content/drive/MyDrive/Colab Notebooks/license_plate_detector.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coco_model.to(device)
    license_plate_detector.to(device)
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_nmr = 0
    all_results = {}
    description=""
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break


        results = process_frame(frame, frame_nmr, coco_model, license_plate_detector, mot_tracker)

        # Convert frame to PIL image for Qwen
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Get scene description
        if not description:
          description = get_scene_description(qwen_pipe, image_pil, question)

        # Merge results
        all_results[frame_nmr] = results.get(frame_nmr, {})
        all_results[frame_nmr]['scene_description'] = description

        print(f"\nFrame {frame_nmr} description:\n{description}\n{'-'*50}")

        frame_nmr += 1

    cap.release()

    # Save to JSON
    if all_results:
        write_json(all_results, output_path)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save")

if __name__ == '__main__':
    main()




