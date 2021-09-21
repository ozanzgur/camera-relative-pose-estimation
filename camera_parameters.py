import cv2
import numpy as np

POINTS_PATH_2D = 'data/vr2d.npy'
POINTS_PATH_3D = 'data/vr3d.npy'

IMG_SIZE = (1920, 1080) # w, h
CX = 960
CY = 540
DIST_COEFFS = (CX, CY)
INTRINSIC_FOCAL_GUESS = 100
CAMERA_MATRIX = np.array([
    [INTRINSIC_FOCAL_GUESS , 0, CX],
    [0, INTRINSIC_FOCAL_GUESS, CY],
    [0, 0, 1]
    ])

class CameraParameters:
    def __init__(self, display = True):
        points2d = np.load(POINTS_PATH_2D)
        points3d = np.load(POINTS_PATH_3D)
        
        # Calculate camera parameters
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            [points3d], [points2d], IMG_SIZE, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        self.ret = ret
        self.matrix = matrix
        self.distortion = distortion
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs

        self.fx = matrix[0, 0]
        self.fy = matrix[1, 1]
        self.cx = CX
        self.cy = CY
        self.size = (self.cx, self.cy)

        if display:
            # Displaying required output
            print(" Camera matrix:")
            print(matrix)