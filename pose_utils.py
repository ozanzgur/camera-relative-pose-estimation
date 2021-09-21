import cv2
import numpy as np

detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def get_feature_points(frame):
    px_ref = detector.detect(frame)
    px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    return px_ref

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2