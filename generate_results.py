import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
from natsort import natsorted
from ultralytics import YOLO
from pose_evaluation_utils import *


model = YOLO("best.pt")


width = 1241.0
height = 376.0
fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]

#0: person
#1: bicycle
#2: car
#3: motorcycle
mask_classes = [0, 1, 2, 3]

def generate_poses(seq:str, YOLO:bool, N:int):


    img_data_dir = './data/data_odometry_color/dataset/sequences/' + seq + '/image_2/'
    img_list = glob(img_data_dir + '/*.png')
    img_list.sort()
    num_frames = len(img_list)

    if N>num_frames: N = num_frames


    if (YOLO):
        output_file = "./results/txt_results/resultsYOLO" + seq + ".txt"
    else:
        output_file = "./results/txt_results/results" + seq + ".txt"


    if os.path.exists(output_file):
        os.remove(output_file)
    

    for i in range(N):
        
        curr_img = cv2.imread(img_list[i])

        if i == 0:
            curr_R = np.eye(3)
            curr_t = np.array([[0], [0], [0]])
        else:
            prev_img = cv2.imread(img_list[i-1])

            orb = cv2.ORB_create(nfeatures=6000)
            
            #------YOLO--------
            if (YOLO):
                #prev_results = model(cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR))
                #prev_results = model(prev_img)
                prev_results = model(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
                prev_mask = np.ones([prev_img.shape[0], prev_img.shape[1]], dtype=np.uint8)*255
                prev_boxes = prev_results[0].boxes
                #curr_results = model(cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR))
                #curr_results = model(curr_img)
                curr_results = model(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
                curr_mask = np.ones([curr_img.shape[0], curr_img.shape[1]], dtype=np.uint8)*255
                curr_boxes = curr_results[0].boxes
                for box,cls in zip(curr_boxes.xyxy,curr_boxes.cls):
                    if cls in mask_classes:
                        x1, y1, x2, y2 = box.int().tolist()
                        curr_mask[y1:y2, x1:x2] = 0
                for box,cls in zip(prev_boxes.xyxy,prev_boxes.cls):
                    if cls in mask_classes:
                        x1, y1, x2, y2 = box.int().tolist()
                        prev_mask[y1:y2, x1:x2] = 0
                kp1, des1 = orb.detectAndCompute(prev_img, prev_mask)
                kp2, des2 = orb.detectAndCompute(curr_img, curr_mask)
            #------YOLO--------

            else:
                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(prev_img, None)
                kp2, des2 = orb.detectAndCompute(curr_img, None)

            # use brute-force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match ORB descriptors
            matches = bf.match(des1, des2)

            # Sort the matched keypoints in the order of matching distance
            # so the best matches came to the front
            matches = sorted(matches, key=lambda x: x.distance)


            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # compute essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))


            # Get camera motion !!!
            # before : image motion
            R = R.transpose()
            t = -np.matmul(R, t)


            curr_t = np.matmul(curr_R, t) + curr_t
            curr_R = np.matmul(curr_R, R)
            
            
        [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
        
        R11, R12, R13 = curr_R[0]
        R21, R22, R23 = curr_R[1]
        R31, R32, R33 = curr_R[2]
        


        with open(output_file, 'a') as f:
            f.write('%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n' % 
            (R11, R12, R13, tx, R21, R22, R23, ty, R31, R32, R33, tz))




#--------------------------------------------------------------------------

def read_pose(line,file):



    with open(file, 'r') as f:
        linhas = f.readlines()

    valores = list(map(float, linhas[line].strip().split()))

    R = np.array([
        [valores[0], valores[1], valores[2]],
        [valores[4], valores[5], valores[6]],
        [valores[8], valores[9], valores[10]]
    ])

    t = np.array([
        [valores[3]],
        [valores[7]],
        [valores[11]]
    ])

    return R,t


def read_file(n_lines,file):
    R_all = []
    t_all = []
    lines = range(0,n_lines)
    for line in lines:
        R,t = read_pose(line,file)
        R_all.append(R)
        t_all.append(t)
    return R_all,t_all





def generate_graphs(seq:str):

    with open("./results/txt_results/results" + seq + ".txt", 'r') as f:
        N = sum(1 for _ in f) 

    if not os.path.exists("results/plots_results/" + seq):
        os.makedirs("results/plots_results/" + seq)


    R_vo_all,t_vo_all = read_file(N,"./results/txt_results/results" + seq + ".txt")
    R_gt_all,t_gt_all = read_file(N,"./data/ground_truth/dataset/poses/" + seq + ".txt")
    R_yolo_all,t_yolo_all = read_file(N,"./results/txt_results/resultsYOLO" + seq + ".txt")


    lines = range(0,N)


    euler_vo   = [mat2euler(M) for M in R_vo_all]
    euler_gt   = [mat2euler(M) for M in R_gt_all]
    euler_yolo = [mat2euler(M) for M in R_yolo_all]
    for i,ax in zip(range(0,3),['z','y','x']):
        angles_vo = [component[i]*180/np.pi for component in euler_vo]
        angles_gt = [component[i]*180/np.pi for component in euler_gt]
        angles_yolo = [component[i]*180/np.pi for component in euler_yolo]
        plt.plot(lines,angles_vo, color='red', label='vo')
        plt.plot(lines,angles_gt, color='green', label='gt')
        plt.plot(lines,angles_yolo, color='blue', label='yolo')
        plt.title("Rotation around "+ax+" (deg)"), plt.grid(True), plt.legend()
        plt.savefig(f"results/plots_results/{seq}/rotation_"+ax+".png")
        plt.close()   

    """
    for i in range(0,3):
        for j in range(0,3):
            r_vo = [R_vo_all[k][i][j] for k in range(0,len(R_vo_all))]
            r_gt = [R_gt_all[k][i][j] for k in range(0,len(R_gt_all))]
            r_yolo = [R_yolo_all[k][i][j] for k in range(0,len(R_yolo_all))]

            plt.plot(lines,r_vo, color='red', label='vo')
            plt.plot(lines,r_gt, color='green', label='gt')
            plt.plot(lines,r_yolo, color='blue', label='yolo')
            plt.title("R"+str(i)+str(j)), plt.grid(True), plt.legend()
            plt.savefig(f"results/plots_results/{seq}/R{i}{j}.png")
            plt.close()
    """


    for i in range(0,3):
        t_vo = [t_vo_all[k][i] for k in range(0,len(t_vo_all))]
        t_gt = [t_gt_all[k][i] for k in range(0,len(t_gt_all))]
        t_yolo = [t_yolo_all[k][i] for k in range(0,len(t_yolo_all))]

        plt.plot(lines,t_vo, color='red', label='vo')
        plt.plot(lines,t_gt, color='green', label='gt')
        plt.plot(lines,t_yolo, color='blue', label='yolo')
        plt.title("t"+str(i)), plt.grid(True), plt.legend()
        plt.savefig(f"results/plots_results/{seq}/t{i}.png")
        plt.close()




def generate_images_with_boxes(seq):


    input_folder = './data/data_odometry_color/dataset/sequences/'+seq+'/image_2'
    output_folder = './results/yolo_boxes/'+seq

    os.makedirs(output_folder, exist_ok=True)


    image_files = sorted(f for f in os.listdir(input_folder) if f.endswith('.png'))

    for filename in image_files:
        filepath = os.path.join(input_folder, filename)

        img = cv2.imread(filepath)

        #results = model(img)
        results = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        result_img = results[0].plot()

        save_path = os.path.join(output_folder, filename)
        plt.imsave(save_path, result_img)

        print(f"Salvo: {save_path}")


def generate_video(seq,fps=10):
    pasta_imagens = "./results/yolo_boxes/" + seq

    video_saida = "./results/videos"

    if not os.path.exists(video_saida):
        os.makedirs(video_saida)

    video_saida += "/video"+seq+".mp4"

    arquivos = [img for img in os.listdir(pasta_imagens) if img.endswith((".jpg", ".png"))]

    arquivos = natsorted(arquivos)

    img_exemplo = cv2.imread(os.path.join(pasta_imagens, arquivos[0]))
    altura, largura, _ = img_exemplo.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_saida, fourcc, fps, (largura, altura))

    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta_imagens, arquivo)
        img = cv2.imread(caminho_completo)
        if img is None:
            print(f"Erro ao carregar {arquivo}")
            continue
        video.write(img)

    video.release()
    print("Vídeo criado com sucesso:", video_saida)