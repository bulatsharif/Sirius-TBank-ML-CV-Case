import numpy as np
import cv2
from typing import List

MIN_MATCH_COUNT = 4
TBANK_LOGO_PATH = "app/models/tbank_logo.png"


# Pre-load logo templates
TBANK_LOGO = cv2.imread(TBANK_LOGO_PATH, cv2.IMREAD_GRAYSCALE)


def detect_image_on_another_SIFT_RANSAC(query_image: np.ndarray, search_image: np.ndarray) -> List[List[float]]:
  # --- Load inputs (grayscale) ---------------------------------------------------
  # We read both the logo ("query") and the scene ("search") as GRAYSCALE because
  # SIFT operates on intensity gradients; color is not required for SIFT to work.
  # Grayscale also saves memory and time without hurting descriptor quality.
  if query_image is None or search_image is None:
      return None

  # --- Detect local features (SIFT) ---------------------------------------------
  # Create a SIFT extractor. For each image, SIFT returns:
  #   - keypoints (kp_*): points with location (x,y), scale, and orientation;
  #   - descriptors (desc_*): a 128-D vector per keypoint encoding local gradient
  #     patterns. These descriptors are approximately invariant to in-plane
  #     rotation and uniform scale changes around each keypoint.
  # We will match descriptors between the query and search images to hypothesize
  # corresponding points that likely belong to the same physical logo parts.
  sift = cv2.SIFT_create(
      nfeatures=5000,              
      contrastThreshold=0.02,      
      edgeThreshold=20,            
      sigma=1.2                     
  )

  kp_q, desc_q = sift.detectAndCompute(query_image, None)
  kp_s, desc_s = sift.detectAndCompute(search_image, None)
  
  def rootsift(desc):
    if desc is None: return None
    eps = 1e-7
    desc = desc.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + eps)
    return np.sqrt(desc)

  desc_q_rs = rootsift(desc_q)
  desc_s_rs = rootsift(desc_s)


  if desc_q is None or desc_s is None or len(desc_q) < 2 or len(desc_s) < 2:
    return None
  # --- Build a descriptor matcher (FLANN + KD-tree) ------------------------------
  # We now search, for each query descriptor, its nearest neighbors in the search
  # image. FLANN (Fast Library for Approximate Nearest Neighbors) provides an
  # efficient KD-tree index for floating-point descriptors like SIFT.
  # 'trees=5' builds an ensemble of KD-trees (more trees -> better recall).
  # 'checks=50' is the number of leaf checks at query time (more -> better accuracy).
  # We ask for the 2 nearest neighbors (k=2) so we can apply the Lowe ratio test.
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
  matches = bf.knnMatch(desc_q_rs, desc_s_rs, k=2)


  # flann = cv2.FlannBasedMatcher(index_params, search_params)
  # try:
  #     matches = flann.knnMatch(desc_q, desc_s, k=2)
  # except cv2.error:
  #     return None
  # --- Lowe ratio test: keep distinctive matches ---------------------------------
  # Each 'matches[i]' contains the best (m) and second-best (n) candidate in the
  # search image for the i-th query descriptor. If m is only slightly better than n,
  # the match is ambiguous and we discard it. The classic threshold is 0.7–0.8; here
  # 0.7 is stricter (fewer, cleaner matches), which helps downstream geometry.
  good_matches = []
  for pair in matches:
    if len(pair) < 2:
      continue
    m, n = pair
    if m.distance < 0.8 * n.distance:
      good_matches.append(m)
  # --- Geometric verification + localization (Homography via RANSAC) -------------
  # Descriptor matches alone are not enough: some are outliers. We robustly fit a
  # single global planar transform (a homography) that many matches agree on.
  # If enough good matches exist (MIN_MATCH_COUNT), we estimate H using RANSAC:
  #   - Input: 2D correspondences between query (src_pts) and search (dst_pts).
  #   - Output: 3x3 homography matrix M and an inlier mask (which matches agree).
  # With M in hand, we can project the rectangle of the logo into the search image,
  # obtaining the quadrilateral that localizes the logo under perspective.
  if len(good_matches) >= MIN_MATCH_COUNT:
    # Build Nx1x2 arrays of matched 2D points in query (src) and search (dst).
    src_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate a planar projective transform with RANSAC.
    # The reprojection threshold is 5.0 pixels (tune with image resolution).
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0, maxIters=5000, confidence=0.999)
    # 'mask' flags which matches are inliers (consistent with M).

    if M is None or mask is None or mask.sum() < 4:
      matchesMask = None
      return None
    else:
      matchesMask = mask.ravel().tolist()

      # Map the four corners of the query image through the homography to the search.
      # This yields the logo's outline as a quadrilateral in the search image.
      h, w = query_image.shape
      pts = np.float32([ [0,0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
      dst = cv2.perspectiveTransform(pts, M)
      points_for_area = np.asarray(dst).reshape(-1, 2).astype(np.float32)
      area_region_result = abs(cv2.contourArea(points_for_area))
      
      bboxes = dst.reshape(-1, 2).tolist()
      if bboxes:
        x_coords = [p[0] for p in bboxes]
        y_coords = [p[1] for p in bboxes]
        width = search_image.shape[1]
        height = search_image.shape[0]

        x_min = int(max(0, min(x_coords)))
        y_min = int(max(0, min(y_coords)))
        x_max = int(min(width, max(x_coords)))
        y_max = int(min(height, max(y_coords)))
        
        
        area_region_rectangle = (x_max - x_min) * (y_max - y_min)
        # print(area_region_result, area_region_rectangle)  
        # print(height * width)

        if area_region_rectangle == 0:
            return None
        
        if area_region_result / area_region_rectangle < 0.1 or area_region_rectangle < (height * width * 0.0005):
          return None
        
      return bboxes
  else:
    # If we don't have enough high-quality matches, homography estimation would
    # be unstable. Report how many we had versus the minimum required.
    return None

def detect_logo_SIFT_RANSAC(image: np.ndarray) -> List[List[float]]:
    if image is None:
        return None

    bboxes_tbank = detect_image_on_another_SIFT_RANSAC(TBANK_LOGO, image)

    if bboxes_tbank:
        return bboxes_tbank
    else:
        return None