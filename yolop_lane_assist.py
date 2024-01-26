import math
import random
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image
import os
import pygetwindow as gw
import dxcam
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from pynput.keyboard import Key, Controller
keyboard = Controller()
import win32api, win32con
from skimage.draw import line


#-------------Parameters----------------
hard_steer_diff = 130
hard_steer_time = 0.2
soft_steer_time = 0.1
distance_threshold = 320
line_detector_width = 70
line_detector_height = 350
game_title = "Euro Truck Simulator"
#---------------------------------------
category_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

category_colors = {category_id: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for category_id in category_map.keys()}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_yolo(frame,model,middle):
    pil_frame = Image.fromarray(original_image.astype(np.uint8))
    results = model(pil_frame)

    results = results.xyxy[0]

    distances = []
    bbox_middle = []
    colors = []

    for pred in results:
        x1, y1, x2, y2, conf, class_idx = pred[:6]
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        conf = conf.item()
        class_idx = int(class_idx.item())

        local_center_x = (x1 + x2) / 2
        local_center_y = (y1 + y2) / 2

        distances.append(math.sqrt((local_center_x - middle[0]) ** 2 + (local_center_y - middle[1]) ** 2))
        bbox_middle.append((local_center_x,local_center_y))
        colors.append(category_colors[class_idx + 1])

        if class_idx+1 in [1,2,3,4,7,6,8,10,13]:
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), category_colors[class_idx + 1], thickness)
            cv2.putText(frame, f"{category_map[class_idx + 1]}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, category_colors[class_idx + 1], thickness)

    if len(distances) > 0:
        best_dist_id = distances.index(max(distances))
        if distances[best_dist_id] <= distance_threshold:
            return frame, bbox_middle[best_dist_id], colors[best_dist_id]
        else:
            return frame, None, None
    else:
        return frame,None, None

def run_yolop(frame,model):
    w,h,ch = frame.shape
    det_out, da_seg_out, ll_seg_out = model(img)
    da_seg_out = generate_mask(da_seg_out,w,h)
    ll_seg_out = generate_mask(ll_seg_out,w,h)
    frame = visualize_results(frame, ll_seg_out, da_seg_out)

    return frame

def generate_mask(model_result,w,h):
    if not isinstance(model_result, np.ndarray):
        model_result = model_result.detach().cpu().numpy()
    model_result = np.squeeze(model_result, axis=0)
    model_result = model_result.reshape((2, h, w))
    model_result = model_result.transpose((1, 2, 0))
    model_result = (model_result[:, :, 1] > 0.5).astype(np.uint8)
    model_result = model_result[:h, :]
    return model_result

def visualize_results(frame,ll_mask,semseg_mask):

    binary_ll_mask = (ll_mask > 0).astype(np.uint8)
    binary_semseg_mask = (semseg_mask > 0).astype(np.uint8)

    alpha_ll = binary_ll_mask * 255
    alpha_semseg = binary_semseg_mask * 255

    colored_ll_mask = frame.copy()
    colored_ll_mask[binary_ll_mask == 1] = [255, 0, 0]

    colored_semseg_mask = frame.copy()

    colored_semseg_mask[binary_semseg_mask == 1] = [0, 255, 0]

    result = cv2.addWeighted(frame, 0.2, colored_ll_mask, 0.8, 0)
    result = cv2.addWeighted(result, 0.7, colored_semseg_mask, 0.3, 0)

    result_with_transparency = cv2.merge([result, alpha_ll | alpha_semseg])


    return result_with_transparency


def apply_steer(diff,key):
    keyboard.press(key)
    if diff < hard_steer_diff:
        keyboard.press('s')
        time.sleep(hard_steer_time)
        keyboard.release('s')
        keyboard.release(key)
    else:
        time.sleep(soft_steer_time)
        keyboard.release(key)

def bresenham_line(start, end):
    rr, cc = line(start[0], start[1], end[0], end[1])
    return list(zip(rr, cc))

def check_line_hits_mask(line_points, mask):
    for point in line_points:
        if mask[point[1]-1, point[0]-1] == 1:
            return True, point  # Return True if a hit is found and the position
    return False, None  # Return False if no hit is found

def check_line_hit(mask,start_point, end_point):
    line_points = bresenham_line(start_point, end_point)
    hit, position = check_line_hits_mask(line_points, mask)
    if hit:
        return True,position
    else:
        return False,(0,0)

def calculate_midpoint(point1, point2):
    x_mid = (point1[0] + point2[0]) // 2
    y_mid = (point1[1] + point2[1]) // 2
    return (x_mid, y_mid)

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

camera = dxcam.create()
loop_time = time.time()
camera.start(target_fps=144)

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True).to(device)
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

open_windows = gw.getAllTitles()
for title in open_windows:
    print(title)
game_window = gw.getWindowsWithTitle(game_title)[0]
game_window.activate()
left, top, right, bottom, width, height = game_window.left, game_window.top, game_window.right,game_window.bottom, game_window.width, game_window.height

start = False
last_steer = None
state_left = win32api.GetKeyState(0x01)
is_mouse_pressed = False
loop_time = time.time()

while True:
    try:
        left, top, right, bottom, width, height = game_window.left, game_window.top, game_window.right, game_window.bottom, game_window.width, game_window.height
        img = camera.grab(region=(left, top, right, bottom))
        if img is None:
            continue
        img = img[:480, :640, :]
        h, w, c = img.shape

        original_image = img
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)
        img = torch.from_numpy(img).to(device)

        det_out, da_seg_out, ll_seg_out = model(img)

        da_seg_out = generate_mask(da_seg_out, w, h)
        ll_seg_out = generate_mask(ll_seg_out, w, h)

        height, width = ll_seg_out.shape[:2]
        bottom_middle_coordinates = (width//2, height)

        original_image, nearest_bbox, nearest_color = run_yolo(original_image, yolo,bottom_middle_coordinates)

        start_point = bottom_middle_coordinates
        #if start == False:
        color_middle = (0, 255, 0)
        if nearest_bbox == None:
            end_point = (bottom_middle_coordinates[0], bottom_middle_coordinates[1] - 150)
        else:
            end_point = (int(nearest_bbox[0]),int(nearest_bbox[1]))
            color_middle = nearest_color
        color = (0, 255, 0)
        thickness = 2
        cv2.line(original_image, start_point, end_point, color_middle, thickness)

        left_start_point = (start_point[0]+line_detector_width,start_point[1])
        if start == False:
            left_end_point = (left_start_point[0],left_start_point[1]-line_detector_height)
        else:
            left_end_point = (left_start_point[0],last_steer[1])

        right_start_point = (start_point[0]-line_detector_width,start_point[1])
        if start == False:
            right_end_point = (right_start_point[0],right_start_point[1]-line_detector_height)
        else:
            right_end_point = (right_start_point[0], last_steer[1])

        left_check,left_pos = check_line_hit(ll_seg_out,left_start_point,left_end_point)
        right_check,right_pos = check_line_hit(ll_seg_out,right_start_point,right_end_point)

        if left_check:
            cv2.line(original_image, left_start_point,left_pos, color, thickness)
        if right_check:
            cv2.line(original_image, right_start_point,right_pos, color, thickness)


        a = win32api.GetKeyState(0x01)
        if a != state_left:
            state_left = a
            print(a)
            if a < 0:
                is_mouse_pressed = True
            else:
                is_mouse_pressed = False


        original_image = visualize_results(original_image, ll_seg_out, da_seg_out)

        if is_mouse_pressed:

            if left_check and right_check == False:
                diff = abs(left_pos[1] - left_start_point[1])
                apply_steer(diff,'a')

            if right_check and left_check == False:
                diff = abs(right_pos[1] - right_start_point[1])
                apply_steer(diff,'d')

            if right_check and left_check:
                if left_pos[1] > right_pos[1]:
                    diff = abs(left_pos[1] - left_start_point[1])
                    apply_steer(diff,'a')
                else:
                    diff = abs(right_pos[1] - right_start_point[1])
                    apply_steer(diff,'d')

        if nearest_bbox is not None:
            angle = angle_between(bottom_middle_coordinates,(int(nearest_bbox[0]),int(nearest_bbox[1])))
            if angle>= 10 and angle<=20:
                keyboard.press('s')
            else:
                keyboard.release('s')
        else:
            keyboard.release('s')

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Overlay', original_image)
        cv2.waitKey(1)
        start = False

        print('FPS {}'.format(1 / (time.time() - loop_time)))
        loop_time = time.time()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }

        print(
            f"Exception at line {traceback_details['lineno']}: {traceback_details['type']}: {traceback_details['message']}")
camera.stop()
