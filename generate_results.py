import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
from natsort import natsorted
from ultralytics import YOLO
from pose_evaluation_utils import *

MODEL = "yolo11n.pt"


fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

# Not trained model
if MODEL == "yolo11n.pt":
    #0:person, 1:bicycle, 2:car, 3:motorcycle
    DETECTED_CLASSES = [0, 1, 2, 3]
    CAR_CLASS = [2]

# trained model 
elif MODEL == "best.pt":
    #0:Car, 1:Pedestrian, 2:Van, 3:Cyclist, 4:Truck, 5:Misc
    DETECTED_CLASSES = [0, 1, 2, 3, 4, 5]
    CAR_CLASS = [0]

model = YOLO(MODEL)


def generate_poses(seq:str, N:int, use_YOLO:bool, Filter_static_objs:bool = False):
    """
    Main function

    1. Read N imgs of Kitti sequence
    2. If use_YOLO == True, create a mask that removes detected objects
    3. If Filter_static_objs == True, re-add static objects
    4. Match points, considering the mask
    5. Calculate pose (R,t)
    6. Write pose in a .txt (Kitti format)
    """

    # prevent "use_YOLO = False" + "Filter_static_objs = True"
    if use_YOLO == False: Filter_static_objs = False

    # list of imgs in seq and updated value of N
    img_list, N = get_N_images_from_folder(seq,N)

    # output file (depends if using or not YOLO)
    output_file = get_output_file(seq,use_YOLO)

    # prepare to enter the for loop...
    curr_img = cv2.imread(img_list[0])
    curr_R = np.eye(3)
    curr_t = np.array([[0], [0], [0]])

    # for all images...
    for i in range(1,N):
        
        # imgs     
        prev_img = cv2.imread(img_list[i-1])
        curr_img = cv2.imread(img_list[i])   

        # prepare masks... at fist everything is 1 (considered to pose estimation)
        prev_mask = np.ones([prev_img.shape[0], prev_img.shape[1]], dtype=np.uint8)*255
        curr_mask = np.ones([curr_img.shape[0], curr_img.shape[1]], dtype=np.uint8)*255

        # if using YOLO, regions with detected objects = 0
        if (use_YOLO):
            detect_objects(prev_img, curr_img, prev_mask, curr_mask)

        # if "Filter_static_objs" activated, make static objects = 1 again (only cars)
        if (Filter_static_objs):
            disconsider_static_objects(prev_img, curr_img, prev_mask, curr_mask)
            
        # find matched points
        pts1,pts2 = find_points(prev_img,curr_img,nfeatures=6000,mask1=prev_mask,mask2=curr_mask)

        # Find R and t between two frames
        R,t = calculate_R_and_t(pts1, pts2)

        # update global t and R
        curr_t = np.matmul(curr_R, t) + curr_t
        curr_R = np.matmul(curr_R, R)
        
        # write line in output file
        write_pose(curr_R,curr_t,output_file)



def generate_images_with_boxes(seq):


    input_folder = './data/data_odometry_color/dataset/sequences/'+seq+'/image_2'
    output_folder = './results/yolo_boxes/'+seq

    os.makedirs(output_folder, exist_ok=True)


    image_files = sorted(f for f in os.listdir(input_folder) if f.endswith('.png'))

    for filename in image_files:
        filepath = os.path.join(input_folder, filename)

        img = cv2.imread(filepath)

        results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        result_img = results[0].plot()

        save_path = os.path.join(output_folder, filename)
        plt.imsave(save_path, result_img)


def generate_images_with_boxes_filtered(seq):
    input_folder = './data/data_odometry_color/dataset/sequences/'+seq+'/image_2'
    output_folder = './results/yolo_boxes/'+seq
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(f for f in os.listdir(input_folder) if f.endswith('.png'))

    prev_img = None
    prev_car_boxes = None

    for filename in image_files:

        filepath = os.path.join(input_folder, filename)
        img = cv2.imread(filepath)

        # Run YOLO
        results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        names = results[0].names if hasattr(results[0], "names") else getattr(model, "names", {})

        # Split detections: cars vs others
        car_boxes = []
        other_boxes = []
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = box.int().tolist()
            c = int(cls); cf = float(conf)
            if c == 2:  #(car)
                car_boxes.append((x1, y1, x2, y2, c, cf))
            else:
                other_boxes.append((x1, y1, x2, y2, c, cf))

        # Decide which cars are moving (skip on first frame)
        moving_idxs = set()
        if prev_img is not None and len(car_boxes) > 0:
            moving_idxs = classify_moving_boxes(
                prev_img, img,
                car_boxes_curr=car_boxes,
                nfeatures=6000,
                inlier_ratio_drop=0.5
            )

        # Draw: all non-car objects; only moving cars
        vis = img.copy()

        def draw_box(b, color=(0, 255, 0)):
            x1, y1, x2, y2, c, cf = b
            label = f"{names.get(c, str(c))} {cf:.2f}" if isinstance(names, dict) else f"{c} {cf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(y1, th + 3)
            cv2.rectangle(vis, (x1, y_text - th - 3), (x1 + tw + 2, y_text + base - 3), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # others (persons, bikes, etc.)
        for b in other_boxes:
            draw_box(b, color=(255, 0, 0))  # blue-ish for non-car classes

        # cars: first frame -> draw all; otherwise draw only moving ones
        if prev_img is None:
            for b in car_boxes:
                draw_box(b, color=(0, 255, 0))  # green
        else:
            for idx, b in enumerate(car_boxes):
                if idx in moving_idxs:
                    draw_box(b, color=(0, 255, 0))  # green for moving car
                # else: skip non-moving car (no box)

        # Save
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, vis)  # keep BGR correct on disk

        # Update previous frame state
        prev_img = img
        prev_car_boxes = car_boxes

# create video with imgs created by generate_images_with_boxes
def generate_video(seq,fps=10):
    folder_imgs = "./results/yolo_boxes/" + seq

    folder_out = "./results/videos"

    if not os.path.exists(folder_out): os.makedirs(folder_out)

    folder_out += "/video"+seq+".mp4"

    files = [img for img in os.listdir(folder_imgs) if img.endswith((".jpg", ".png"))]

    files = natsorted(files)

    img_example = cv2.imread(os.path.join(folder_imgs, files[0]))
    h, w, _ = img_example.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video = cv2.VideoWriter(folder_out, fourcc, fps, (w, h))

    for file in files:

        full_path = os.path.join(folder_imgs, file)

        img = cv2.imread(full_path)
        
        video.write(img)

    video.release()

# --------------------------------------------------------------------------- #
#                             Utility functions                               #
# --------------------------------------------------------------------------- #

# calculate iou
def iou(boxA, boxB):
    """
    box: (x1,y1,x2,y2)
    """
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)

# find matched points 
def find_points(prev_img,curr_img,nfeatures,mask1,mask2):
    """
    imgs: BGR
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, mask1)
    kp2, des2 = orb.detectAndCompute(curr_gray, mask2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts_prev = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_curr = np.float32([kp2[m.trainIdx].pt for m in matches])

    return pts_prev,pts_curr

# make regions with detected objects = 0
def detect_objects(prev_img, curr_img, prev_mask, curr_mask):

    # apply YOLO model
    prev_results = model(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
    curr_results = model(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))

    # build (x1,y1,x2,y2,cls,conf) lists 
    prev_boxes = boxes_to_xyxy(prev_results, DETECTED_CLASSES)
    curr_boxes = boxes_to_xyxy(curr_results, DETECTED_CLASSES)

    # update masks 
    for (x1, y1, x2, y2, cls, conf) in prev_boxes:
        prev_mask[y1:y2, x1:x2] = 0
    for (x1, y1, x2, y2, cls, conf) in curr_boxes:
        curr_mask[y1:y2, x1:x2] = 0
    
    return prev_mask, curr_mask

# make regions with static objects = 1 again
# we will evaluate only cars (class number = 2) 
def disconsider_static_objects(prev_img, curr_img, prev_mask, curr_mask):

    # apply YOLO model
    curr_results = model(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))

    # build (x1,y1,x2,y2,cls,conf) lists only for cars
    curr_boxes = boxes_to_xyxy(curr_results, CAR_CLASS)

    # find moving boxes
    moving_idxs = classify_moving_boxes(prev_img,curr_img,curr_boxes)

    # Disconsider not moving cars 
    for idx, (x1, y1, x2, y2, cls, conf) in enumerate(curr_boxes):
        if idx not in moving_idxs:
            curr_mask[y1:y2, x1:x2] = 1

    return prev_mask, curr_mask

# return ixs of moving objects
def classify_moving_boxes(prev_img, curr_img, boxes_curr,
                          nfeatures=6000, 
                          inlier_ratio_drop=0.5):
    """
    Return the indices of moving cars

    Strategy:
      1) Compute ORB matches on the whole frame (no mask) and estimate Essential matrix with RANSAC.
      2) For each current car box, compute the fraction of matches that are RANSAC inliers.
         - If this box inlier ratio << global inlier ratio => moving.
    """

    # matches
    pts_prev,pts_curr = find_points(prev_img,curr_img,nfeatures)
    pts_curr_int = np.round(pts_curr).astype(int)

    # inliers
    E, inlier_mask = cv2.findEssentialMat(
        pts_prev, pts_curr, focal=fx, pp=(cx, cy),
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    # fraction of inliers
    global_inlier_ratio = inlier_mask.mean() 
    
    # prepare to enter the for loop...
    moving_idxs = set()
    
    # for each box...
    for idx, (x1, y1, x2, y2, cls, conf) in enumerate(boxes_curr):

        # Points inside box
        # ex: inside = [ True, True, False, True ], for each match in curr img
        inside = (pts_curr_int[:,0] > x1) & (pts_curr_int[:,0] < x2) & \
                 (pts_curr_int[:,1] > y1) & (pts_curr_int[:,1] < y2)
                
        # ex: idxs = [ 0, 1, 3 ]
        idxs = np.where(inside)[0]

        # fraction of inliers inside box
        box_inliers = inlier_mask[idxs].mean()

        # moving?
        if box_inliers < global_inlier_ratio * inlier_ratio_drop:
            moving_idxs.add(idx)

    return moving_idxs

# return a list with N imgs of a seq 
def get_N_images_from_folder(seq,N):

    img_data_dir = './data/data_odometry_color/dataset/sequences/' + seq + '/image_2/'
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)
    if N>num_frames: N = num_frames
    
    return img_list, N

# get output file name (depends if using YOLO or not)
def get_output_file(seq,use_YOLO):

    if (use_YOLO):
        output_file = "./results/txt_results/resultsYOLO" + seq + ".txt"
    else:
        output_file = "./results/txt_results/results" + seq + ".txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    return output_file

# convert from ultralytics format (results) to (x1,y1,x2,y2,cls,conf)
def boxes_to_xyxy(results, wanted_classes):
    boxes = []
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if int(cls) in wanted_classes:
            x1, y1, x2, y2 = box.int().tolist()
            boxes.append((x1, y1, x2, y2, int(cls), float(conf)))
    return boxes

# calculate poses
def calculate_R_and_t(pts1, pts2):
    """
    R and t between two frames with matched points
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))
    R = R.transpose()
    t = -np.matmul(R, t)
    return R,t

# write rotation/translation in a line
def write_pose(R,t,output_file):

    [tx, ty, tz] = [t[0], t[1], t[2]]
        
    R11, R12, R13 = R[0]
    R21, R22, R23 = R[1]
    R31, R32, R33 = R[2]
    
    with open(output_file, 'a') as f:
        f.write('%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n' % 
        (R11, R12, R13, tx, R21, R22, R23, ty, R31, R32, R33, tz))

# Read R and t from a whole file
def read_file(n_lines,file):
    R_all = []
    t_all = []
    lines = range(0,n_lines)
    for line in lines:
        R,t = read_pose(line,file)
        R_all.append(R)
        t_all.append(t)
    return R_all,t_all

# Read R and t from a single line
def read_pose(line,file):

    with open(file, 'r') as f:
        lines = f.readlines()
    values = list(map(float, lines[line].strip().split()))

    R = np.array([
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]]
    ])

    t = np.array([
        [values[3]],
        [values[7]],
        [values[11]]
    ])

    return R,t