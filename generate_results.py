import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from natsort import natsorted
from ultralytics import YOLO
from pose_evaluation_utils import *
from IPython.display import clear_output
import shutil
import subprocess
from scipy.spatial.transform import Rotation as R

MODEL = "best.pt"


fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

# Not trained model
if MODEL == "yolo11n.pt":
    CLASSES_NAMES = {0:"person",1:"bicycle",2:"car",3:"motorcycle"}
    DETECTED_CLASSES = [0, 1, 2, 3]
    CAR_CLASS = [2]
    #person:gray, bicycle:gray, car:red, motorcycle:gray (BGR)
    COLORS_CLASSES = {0:(128, 128, 128),1:(128, 128, 128),2:(0,0,255),3:(128, 128, 128)}

# trained model 
elif MODEL == "best.pt":
    #0:Car, 1:Pedestrian, 2:Van, 3:Cyclist, 4:Truck, 5:Misc
    CLASSES_NAMES = {0:"Car",1:"Pedestrian",2:"Van",3:"Cyclist",4:"Truck",5:"Misc"}
    DETECTED_CLASSES = [0, 1, 2, 3, 4, 5]
    CAR_CLASS = [0]
    #Car:red, Pedestrian:gray, Van:gray, Cyclist:gray, Truck:gray, Misc:gray (BGR)
    COLORS_CLASSES = {0:(0,0,255),1:(128, 128, 128),2:(128, 128, 128),3:(128, 128, 128),4:(128, 128, 128),5:(128, 128, 128)}

model = YOLO(MODEL)

# main function -> does the odometry pipeline
def generate_poses(seq:str, frame_start:int, frame_end:int, use_YOLO:bool, 
                   Filter_static_objs:bool = False, inlier_ratio:float = 1):
    """
    Main function:
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
    img_list = get_images_from_folder(seq)

    # output file (depends if using or not YOLO)
    output_file = get_output_file(seq,frame_start,frame_end,use_YOLO,Filter_static_objs,inlier_ratio)

    # clean file
    if os.path.exists(output_file): os.remove(output_file)

    # prepare to enter the for loop...
    curr_R = np.eye(3)
    curr_t = np.array([[0], [0], [0]])
    write_pose(curr_R,curr_t,output_file)

    # for all images...
    for i in range(frame_start+1,frame_end+1):
        
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
            disconsider_static_objects(prev_img, curr_img, prev_mask, curr_mask, inlier_ratio)
            
        # find matched points
        pts1,pts2 = find_points(prev_img,curr_img,mask1=prev_mask,mask2=curr_mask)

        # Find R and t between two frames
        R,t = calculate_R_and_t(pts1, pts2)

        # update global t and R
        curr_t = np.matmul(curr_R, t) + curr_t
        curr_R = np.matmul(curr_R, R)
        
        # write line in output file
        write_pose(curr_R,curr_t,output_file)
    
# create images with boxes around detected objects
def generate_images_with_boxes(seq:str, frame_start:int, frame_end:int,
                               Filter_static_objs:bool = False, inlier_ratio = 1):

    # get N KITTI sequency imgs
    img_list = get_images_from_folder(seq)

    # output folder
    output_folder = f'./results/yolo_boxes/'

    os.makedirs(output_folder, exist_ok=True)

    for i in range(frame_start,frame_end+1):

        # imgs     
        curr_img = cv2.imread(img_list[i]) 

        # detections
        results = model(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
        clear_output(wait=False)

        # build (x1,y1,x2,y2,cls,conf) lists 
        curr_boxes = boxes_to_xyxy(results, DETECTED_CLASSES)

        moving_idxs = []

        # if we want to remove boxes in static objects (cars)
        if(Filter_static_objs and i>0):

            # get previous img
            prev_img = cv2.imread(img_list[i-1])

            # find moving boxes
            moving_idxs,_,_ = classify_moving_boxes(prev_img,curr_img,curr_boxes,inlier_ratio=inlier_ratio)

        # draw boxes
        for idx,b in enumerate(curr_boxes):

            # moving -> red, static -> green
            if idx in moving_idxs:
                draw_box(curr_img, b, (0,0,255))
            else:
                draw_box(curr_img, b, (0,255,0))

        # save img
        plt.imsave(output_folder + seq + "_" + os.path.basename(img_list[i]), cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))

# create video with imgs created by generate_images_with_boxes
def generate_video(map,Filter_static_objs = False,inlier_ratio = 1,fps=10):
    """
    map (dict): {"00":[(start,end),(start,end)], "01":[(start,end)], ...}
    """

    # video_name: 0.9_00_3_01_5_05_1.mp4
    if Filter_static_objs:
        video_name = str(inlier_ratio) 
    else: 
        video_name = "N"

    
    for seq,intervals in map.items():
        video_name += "_" + seq + "_" + str(len(intervals)) 
    
        for interval in intervals:
            frame_start,frame_end = interval
            generate_images_with_boxes(seq, frame_start, frame_end,Filter_static_objs, inlier_ratio)
    

    video_name += ".mp4"
    folder_imgs = "./results/yolo_boxes"
    folder_out = "./results/videos"

    if not os.path.exists(folder_out): os.makedirs(folder_out)

    folder_out += "/" + video_name 
    files = [img for img in os.listdir(folder_imgs) if img.endswith((".jpg", ".png"))]
    
    files = natsorted(files)
    print(files)

    img_example = cv2.imread(os.path.join(folder_imgs, files[0]))
    h, w, _ = img_example.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video = cv2.VideoWriter(folder_out, fourcc, fps, (w, h))

    for file in files:

        full_path = os.path.join(folder_imgs, file)

        img = cv2.imread(full_path)

        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))

        video.write(img)

    video.release()

    shutil.rmtree(folder_imgs)

# compare metrics of two different methods
def compare_metrics(map,method1,method2):
    """
    method: (yolo:bool, filter:bool, ir:int)
    example: (True,True,1)
    """

    # get methods infos
    yolo1,filter1,ir1 = method1
    yolo2,filter2,ir2 = method2

    # name output_file based on methods
    output_file = name_comparison_file(method1,method2)

    # loop through all excerpts
    for seq,intervals in map.items():

        # write seq
        with open(output_file,"a") as f: f.write(f"Seq {seq}\n")

        for interval in intervals:
            frame_start,frame_end = interval

            # gt excerpt
            generate_gt_files(seq, frame_start, frame_end)

            # get metrics
            ape_t1,rpe_t1,rpe_r1 = calculate_metrics(seq,frame_start,frame_end,yolo1,filter1,ir1)
            ape_t2,rpe_t2,rpe_r2 = calculate_metrics(seq,frame_start,frame_end,yolo2,filter2,ir2)

            # bold best metrics (smallest)
            if float(ape_t1)<=float(ape_t2): ape_t1 = f"\\textbf{{{ape_t1}}}" 
            else: ape_t2 = f"\\textbf{{{ape_t2}}}"
            if float(rpe_t1)<=float(rpe_t2): rpe_t1 = f"\\textbf{{{rpe_t1}}}" 
            else: rpe_t2 = f"\\textbf{{{rpe_t2}}}"
            if float(rpe_r1)<=float(rpe_r2): rpe_r1 = f"\\textbf{{{rpe_r1}}}" 
            else: rpe_r2 = f"\\textbf{{{rpe_r2}}}"

            # write results
            with open(output_file,"a") as f:
                f.write(f"{frame_start} to {frame_end} & {ape_t1} & {rpe_t1} & {rpe_r1} " \
                        f"& {ape_t2} & {rpe_t2} & {rpe_r2} \\\\\n")
    
        # skip line
        with open(output_file,"a") as f: f.write(f"\n")

# graph (evo)
def generate_graph(seq,frame_start,frame_end,estimation1_file_path,estimation2_file_path):

    gt_file_path = f"./results/txt_results/{seq}/gt/{frame_start}_to_{frame_end}.txt"
    cmd = f"evo_traj kitti {estimation1_file_path} {estimation2_file_path} --ref={gt_file_path} -p --plot_mode=xz -as"
    subprocess.run(cmd, shell=True)

# graphs for all excerpts
def generate_all_graphs(map, methods):
    """
    methods: [method1,method2,...], method: (yolo:bool, filter:bool, ir:int)
    exemple: [(True,True,1), (False,True,2), (True,False,3)]
    """

    plt.rc("text", usetex=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14
    })

    output_folder = "results/plots_results/"

    for seq, intervals in map.items():
        for frame_start, frame_end in intervals:

            output_folder_full = os.path.join(output_folder, f"{seq}/{frame_start}_to_{frame_end}")
            os.makedirs(output_folder_full, exist_ok=True)
            generate_graph_mine(seq, frame_start, frame_end, methods, output_folder_full)

# plots traj, translation, rotation
def generate_graph_mine(seq, frame_start, frame_end, methods, output_folder):

    estimation_paths = []
    method_names = []
    for (yolo, flt, ir) in methods:
        estimation_paths.append(get_output_file(seq, frame_start, frame_end, yolo, flt, ir))
        
        if not yolo: method_names.append("No YOLO")
        elif not flt: method_names.append("YOLO")
        else: method_names.append("YOLO filtered")
    
    gt_file_path = f"./results/txt_results/{seq}/gt/{frame_start}_to_{frame_end}.txt"
    N = frame_end - frame_start + 1    
    os.makedirs(output_folder, exist_ok=True)

    # --- Read GT ---
    R_gt_all, t_gt_all = read_file(N, gt_file_path)
    t_gt = np.hstack(t_gt_all)  # (3, N)

    # --- Read and align each method ---
    methods_R, methods_t = [], []
    for est_path in estimation_paths:
        R_all, t_all = read_file(N, est_path)
        t = np.hstack(t_all)

        # Align with GT
        c, R_align, t_align = umeyama(t, t_gt)
        t_aligned = [c * (R_align @ ti) + t_align for ti in t_all]
        R0_gt = R_gt_all[0]
        R0_est = R_all[0]
        R0_correction = R0_gt @ (R_align @ R0_est).T
        R_aligned = [R0_correction @ (R_align @ Ri) for Ri in R_all]

        methods_R.append(R_aligned)
        methods_t.append(t_aligned)

    lines = range(0, N)

    # --- Roll, Pitch, Yaw ---
    euler_gt = [mat2rpy(M) for M in R_gt_all]
    euler_methods = [[mat2rpy(M) for M in R_all] for R_all in methods_R]

    for i, ax_name in zip(range(0, 3), ['roll', 'pitch', 'yaw']):
        angles_gt = [comp[i] for comp in euler_gt]
        plt.figure()
        plt.grid()
        for idx, euler_m in enumerate(euler_methods):
            errors = [wrap_angle_deg(angles_gt[j] - euler_m[j][i]) for j in range(N)]
            plt.plot(lines, errors, linewidth=2, label=method_names[idx])
        plt.title(rf"Error {ax_name.capitalize()} ($^\circ$)")
        plt.legend(framealpha=1.0)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{ax_name}.png", format="png", dpi=400, bbox_inches="tight")
        plt.close()

    # --- Translations ---
    for i, ax_name in zip(range(0, 3), ['x','y','z']):
        plt.figure()
        plt.grid()
        for idx, t_all in enumerate(methods_t):
            errors = [t_gt_all[j][i] - t_all[j][i] for j in range(N)]
            plt.plot(lines, errors, linewidth=2, label=method_names[idx])
        plt.title(rf"Error {ax_name.upper()} (m)")
        plt.legend(framealpha=1.0)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{ax_name}.png", format="png", dpi=400, bbox_inches="tight")
        plt.close()

    # --- Trajectory (XZ plane) ---
    x_gt = [t[0][0] for t in t_gt_all]
    z_gt = [t[2][0] for t in t_gt_all]

    plt.figure()
    plt.grid()
    plt.plot(x_gt, z_gt, linewidth=2, label="GT")
    for idx, t_all in enumerate(methods_t):
        x_m = [t[0][0] for t in t_all]
        z_m = [t[2][0] for t in t_all]
        plt.plot(x_m, z_m, linewidth=2, label=method_names[idx])
    plt.title(r"Trajectory (XZ plane)")
    plt.xlabel(r"X (m)")
    plt.ylabel(r"Z (m)")
    plt.axis("equal")
    plt.legend(framealpha=1.0)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/trajectory_xz.png", format="png", dpi=400, bbox_inches="tight")
    plt.close()


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
def find_points(prev_img,curr_img,nfeatures=6000,mask1=None,mask2=None):
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
    clear_output(wait=False)

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
def disconsider_static_objects(prev_img, curr_img, prev_mask, curr_mask, inlier_ratio):

    # apply YOLO model
    curr_results = model(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
    clear_output(wait=False)

    # build (x1,y1,x2,y2,cls,conf) lists only for cars
    curr_boxes = boxes_to_xyxy(curr_results, CAR_CLASS)

    # find moving boxes
    moving_idxs,_,_ = classify_moving_boxes(prev_img,curr_img,curr_boxes,inlier_ratio=inlier_ratio)

    # Disconsider not moving cars 
    for idx, (x1, y1, x2, y2, cls, conf) in enumerate(curr_boxes):
        if idx not in moving_idxs:
            curr_mask[y1:y2, x1:x2] = 1

    return prev_mask, curr_mask

# return ixs of moving objects
def classify_moving_boxes(prev_img, curr_img, boxes_curr,
                          nfeatures=6000, 
                          inlier_ratio=1):
    """
    Strategy:
      1) Compute ORB matches on the whole frame (no mask) and estimate Essential matrix with RANSAC.
      2) For each current car box, compute the fraction of matches that are RANSAC inliers.
         - If box inlier ratio < global inlier ratio * inlier_ratio => moving.
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
    global_inliers = inlier_mask.mean() 
    
    # prepare to enter the for loop...
    moving_idxs = []
    box_inliers_all = []
    
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
        box_inliers_all.append(box_inliers)

        # moving?
        if box_inliers < global_inliers * inlier_ratio:
            moving_idxs.append(idx)

    return moving_idxs, box_inliers_all, global_inliers

# return a list with N imgs of a seq 
def get_images_from_folder(seq):

    img_data_dir = './data/data_odometry_color/dataset/sequences/' + seq + '/image_2/'
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    
    return img_list

# get output file name (depends if using YOLO or not)
def get_output_file(seq,frame_start,frame_end,use_YOLO,Filter_static_objs,inlier_ratio):

    output_path = f"./results/txt_results/{seq}/"

    if not use_YOLO:
        output_path += "noYOLO/"
    else:
        if Filter_static_objs:
            output_path += f"{MODEL.removesuffix(".pt")}/filter{inlier_ratio}/"
        else:
            output_path += f"{MODEL.removesuffix(".pt")}/noFilter/"

    os.makedirs(output_path,exist_ok=True)

    filename = f"{frame_start}_to_{frame_end}.txt"
    output_path += filename

    return output_path

# convert from yolo format (results) to (x1,y1,x2,y2,cls,conf)
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

# draw yolo box
def draw_box(img, b, color):

    # box attributes
    x1, y1, x2, y2, c, cf = b

    # dict with detectable classes
    names = {i: name for i, name in model.names.items() if i in DETECTED_CLASSES}

    # box label
    label = f"{names.get(c, str(c))} {cf:.2f}" 

    # box color
    #color = COLORS_CLASSES[c]

    # drawing
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1, th + 3)
    cv2.rectangle(img, (x1, y_text - th - 3), (x1 + tw + 2, y_text + base - 3), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 1, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

# get metrics
def select_metrics(log_file):
    """
    log file: 
        file with the outputs of: ape, rpe (t) and rpe (r)
    """

    with open(log_file,"r") as f:
        all_metrics = f.readlines()

    ape_t = all_metrics[7].strip().split()[-1]
    rpe_t = all_metrics[16].strip().split()[-1]
    rpe_r = all_metrics[28].strip().split()[-1]

    return ape_t,rpe_t,rpe_r

# return: ape_t,rpe_t,rpe_r
def calculate_metrics(seq,frame_start,frame_end,use_YOLO,Filter_static_objs,inlier_ratio):

    # get txt result corresponding to seq,frame_start,...
    estimation_file_path = get_output_file(
        seq,frame_start,frame_end,use_YOLO,Filter_static_objs,inlier_ratio)

    # get corresponding gt excerpt file 
    gt_file_path = f"./results/txt_results/{seq}/gt/{frame_start}_to_{frame_end}.txt"

    # temp.txt with commands outputs
    log_file =  f"results/evaluation/temp.txt"

    # execute comands
    cmd1 = f"evo_ape kitti {gt_file_path} {estimation_file_path} -as --pose_relation=trans_part"
    cmd2 = f"evo_rpe kitti {gt_file_path} {estimation_file_path} -as --pose_relation=trans_part"
    cmd3 = f"evo_rpe kitti {gt_file_path} {estimation_file_path} -as --pose_relation=angle_deg"
    with open(log_file, "w") as f:
        subprocess.run(cmd1, shell=True, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(cmd2, shell=True, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(cmd3, shell=True, stdout=f, stderr=subprocess.STDOUT)

    # get metrics
    ape_t,rpe_t,rpe_r = select_metrics(log_file)

    # remove temps
    os.remove(log_file)

    return ape_t,rpe_t,rpe_r

# create a dict with seqs excerpts
def read_seq_excerpts_file():

    map = {}
    with open("data/seq_excerpts.txt","r") as f:
        lines = f.readlines()
        n = len(lines)
        i = 0
        for i in range(n):
            line = lines[i].strip()
            if line.isdecimal():
                j = 1
                intervals = []
                while i+j < n and lines[i+j].strip():
                    start,_,end = lines[i+j].split("-")[-1].strip().partition(",")
                    start = int(start)
                    end = int(end)
                    intervals.append((start,end))
                    j += 1
                if intervals:
                    map[line] = intervals
            i += 1

    return map

# name output_file of compare_metrics based on methods
def name_comparison_file(method1,method2):

    # get methods infos
    yolo1,filter1,ir1 = method1
    yolo2,filter2,ir2 = method2

    output_folder = "results/evaluation/"
    output_name = ""
    if not yolo1: output_name += "noYolo"
    elif filter1: output_name += f"YoloFilt{ir1}"
    else: output_name += "YoloNoFilt"
    output_name += "_vs_"
    if not yolo2: output_name += "noYolo"
    elif filter2: output_name += f"YoloFilt{ir2}"
    else: output_name += "YoloNoFilt"
    output_name += ".txt"
    output_file = output_folder+output_name
    if os.path.exists(output_file): os.remove(output_file)

    return output_file

# 3x3 rotation matrix to roll, pitch, yaw (degrees)
def mat2rpy(M):
    r = R.from_matrix(M)
    return r.as_euler('xyz', degrees=True)  # roll, pitch, yaw

# umeyama algorithm
def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t

# create ground truth excerpts
def generate_gt_files(seq:str, frame_start:int, frame_end:int):

    # read gt 
    gt_complete = f"data/ground_truth/dataset/poses/{seq}.txt"
    with open(gt_complete,"r") as f: content = f.readlines()

    # prepare output file
    gt_excerpt = f"./results/txt_results/{seq}/gt/{frame_start}_to_{frame_end}.txt"
    os.makedirs(os.path.dirname(gt_excerpt),exist_ok=True)

    # write selected lines
    with open(gt_excerpt,"w") as f:
        for line in content[frame_start:frame_end+1]:
            f.write(line)

# wrap angle to [-180, 180] range
def wrap_angle_deg(angle):
    
    return (angle + 180) % 360 - 180

