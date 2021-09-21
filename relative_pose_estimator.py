import numpy as np
import cv2
import camera_parameters
import utils
import pose_utils

class RelativePoseEstimator:
	def __init__(self, camera_parameters : camera_parameters.CameraParameters, first_frame : np.array):
		self.px_ref = pose_utils.get_feature_points(first_frame)
		self.cp = camera_parameters
		self.last_frame = first_frame
		self.n_get_new_features = 1500

	def process_next(self, new_frame : np.array):
		# Process second frame and estimate pose
		self.px_ref, px_cur = pose_utils.featureTracking(self.last_frame, new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(px_cur, self.px_ref, focal = self.cp.fx, pp = self.cp.size, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
		_, cur_R, cur_t, mask = cv2.recoverPose(E, px_cur, self.px_ref, focal=self.cp.fx, pp = self.cp.size)

		# When there are not enough features to track, get new features
		if px_cur.shape[0] < self.n_get_new_features:
			self.get_new_features()

		self.last_frame = new_frame
		self.px_ref = px_cur
		R_euler = utils.mat2euler(cur_R)
		return cur_t[:, 0], np.array(R_euler)

	def get_new_features(self):
		self.px_ref = pose_utils.get_feature_points(self.last_frame)