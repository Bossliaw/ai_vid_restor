import numpy as np
import cv2


def align_src_img_to_dst(src_img, dst_img, fitter, **kwargs):
    map_x, map_y = fitter.fit_target_to_source(dst_img, src_img)
    # interp: image pixel sampling interpolation method
    reg_img = cv2.remap(src_img, map_x, map_y, kwargs.get("interp", cv2.INTER_LANCZOS4))

    unregister_absdiff = cv2.absdiff(dst_img, src_img)
    registered_absdiff = cv2.absdiff(dst_img, reg_img)
    unregister_sad = np.sum(unregister_absdiff)
    registered_sad = np.sum(registered_absdiff)
    reg_successful = unregister_sad > registered_sad

    # TODO: draw alignment vector flow
    dbg_viz_size = kwargs.get("dbg_viz_size", (0, 0))
    if dbg_viz_size != (0, 0):
        unregister_overlay = cv2.addWeighted(dst_img, 0.5, src_img, 0.5, 0)
        registered_overlay = cv2.addWeighted(dst_img, 0.5, reg_img, 0.5, 0)

        viz_aligned = np.hstack((unregister_overlay, registered_overlay))
        viz_absdiff = np.hstack((unregister_absdiff, registered_absdiff))

        viz_resized = cv2.resize(np.vstack((viz_aligned, viz_absdiff)), dbg_viz_size)
        cv2.imshow("Registration", viz_resized)
        cv2.waitKey(delay=20)

    dbg_msg = "SAD {} from {:10d} to {:10d}, registration {}".format(
        "decreased" if reg_successful else "increased",
        unregister_sad,
        registered_sad,
        "OK!" if reg_successful else "NG!",
    )
    print(dbg_msg)
    return reg_img if reg_successful else None


class ImageRegistrationAligner:
    def __init__(self):
        pass

    def fit_target_to_source(self, dst_img, src_img):
        # no-op / identity implementation
        return np.indices(dst_img.shape[:2], dtype=np.float32)


# Method 1: feature keypoint matching + global homography transformation fitting
# See: https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
# See: https://stackoverflow.com/questions/46520123
class OrbFeatureAligner(ImageRegistrationAligner):
    def __init__(self):
        self.orb = cv2.ORB_create(
            edgeThreshold=15,
            patchSize=31,
            nlevels=8,
            fastThreshold=20,
            scaleFactor=1.2,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            firstLevel=0,
            nfeatures=1500,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # self.flann = cv2.FlannBasedMatcher_create()

    def fit_target_to_source(self, dst_img, src_img):
        src_keypts, src_feats = self.orb.detectAndCompute(src_img, mask=None)
        dst_keypts, dst_feats = self.orb.detectAndCompute(dst_img, mask=None)
        matches = self.bf.match(src_feats, dst_feats)
        # matches = self.flann.knnMatch(src_feats, dst_feats, k=2)

        # Store all the good matches as per Lowe's ratio test.
        # map_pairs = []
        # for m, n in matches:
        #    if m.distance < 0.7 * n.distance:
        #        map_pairs.append(m)
        map_pairs, num_pairs = matches, len(matches)
        # dbg_msg = "src_feats: {}, dst_feats: {}, map_pairs: {}".format(
        #     len(src_feats),
        #     len(dst_feats),
        #     num_pairs,
        # )
        # print(dbg_msg)

        if num_pairs < 4:
            raise ValueError("less than 4 pairs, unable to estimate homography")

        # Extract matched keypoints for later homography estimation
        src_good_pts = np.float32(
            [src_keypts[m.queryIdx].pt for m in map_pairs]
        ).reshape(-1, 1, 2)
        dst_good_pts = np.float32(
            [dst_keypts[m.trainIdx].pt for m in map_pairs]
        ).reshape(-1, 1, 2)
        dst_to_src_H, mask = cv2.findHomography(
            dst_good_pts, src_good_pts, cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if dst_to_src_H is None:
            raise ValueError("fail to reach consensus homography via random sampling")

        # Draw matching images
        # viz_params = dict(
        #     matchColor=(0, 255, 0),  # draw matches in green color
        #     singlePointColor=None,
        #     matchesMask=mask.ravel().tolist(),  # draw only inliers
        #     flags=2,
        # )
        # dbg_viz = cv2.drawMatches(
        #     src_img, src_keypts, dst_img, dst_keypts, map_pairs, None, **viz_params
        # )

        # Compute entire warping lookup table from homography matrix
        # See: https://stackoverflow.com/questions/46520123
        # create indices of the destination image and linearize them
        h, w = dst_img.shape[:2]
        indy, indx = np.indices((h, w), dtype=np.float32)
        lin_homg_ind = np.array(
            [indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()]
        )

        # warp the coordinates of src to those of true_dst
        map_ind = dst_to_src_H.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1] / map_ind[-1]  # ensure homogeneity
        map_x = map_x.reshape(h, w).astype(np.float32)
        map_y = map_y.reshape(h, w).astype(np.float32)

        return map_x, map_y


# Method 2: motion estimation & compensation
# See: https://stackoverflow.com/questions/43512039
class OpticalFlowAligner(ImageRegistrationAligner):
    def __init__(self):
        pass

    def fit_target_to_source(self, dst_img, src_img):
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        cv2.calcOpticalFlowFarneback(dst_gray, src_gray)
        # ... TODO ...

        # no-op / identity implementation
        return np.indices(dst_img.shape[:2], dtype=np.float32)


# Method 3: other registration techniques (median threshold bitmap, etc ...)
# NOTE: cv2.reg requires extra modules ...
class MedianThresholdBitmapAligner(ImageRegistrationAligner):
    pass  # ... TODO ...
