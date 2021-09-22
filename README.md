# Camera Relative Pose Estimation
Estimate relative camera pose in 2 images relative to a reference frame and plot the trajectory.
- Outputs can be found in "estimate_pose.ipynb"

### Steps:
#### 1- Camera Calibration
cv2.calibrateCamera is called in camera_parameters.py to calculate intrinsic parameters.
CAMERA_MATRIX contains the intrinsic guess with focal_length = 100
```python
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            [points3d], [points2d], IMG_SIZE, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
cp = CameraParameters()
```
#### 2- Get Feature Coordinates in the First Frame
Detect features from the first frame and get their coordinates. These features will be matched
with the ones in the next frame for pose estimation.
cv2.FastFeatureDetector is used for feature detection.
```
detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
px_ref = detector.detect(frame)
```
#### 3- Match Features in the Next Frame
```
kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
```
#### 4- Calculate Essential Matrix
```
E, mask = cv2.findEssentialMat(px_cur, self.px_ref, focal = self.cp.fx,
            pp = self.cp.size, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
```
#### 5- Estimate Pose
Estimate pose from the essential matrix using the method in:
[David Nistér. An efficient solution to the five-point relative pose problem.](https://dl.acm.org/doi/10.1109/TPAMI.2004.17)
```
_, cur_R, cur_t, mask = cv2.recoverPose(E, px_cur, self.px_ref, focal=self.cp.fx, pp = self.cp.size)
```

#### References

[https://github.com/uoip/monoVO-python](https://github.com/uoip/monoVO-python)
[David Nistér. An efficient solution to the five-point relative pose problem.](https://dl.acm.org/doi/10.1109/TPAMI.2004.17)
