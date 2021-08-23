import sys
import pathlib
import numpy as np
import cv2

# print(cv2.getBuildInformation())

# Load original/restored video
src_vid_path = pathlib.Path(sys.argv[1]).resolve()
ref_vid_path = pathlib.Path(sys.argv[2]).resolve()
original_vid = cv2.VideoCapture(str(src_vid_path))  # source
restored_vid = cv2.VideoCapture(str(ref_vid_path))  # target
original_vid_length = int(original_vid.get(cv2.CAP_PROP_FRAME_COUNT))
restored_vid_length = int(restored_vid.get(cv2.CAP_PROP_FRAME_COUNT))
print("original_vid_length:", original_vid_length)
print("restored_vid_length:", restored_vid_length)
if original_vid_length != restored_vid_length:
    raise ValueError("Unable to do video registration on different length videos")

src_frame_width = int(original_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
src_frame_height = int(original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
ref_frame_width = int(restored_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
ref_frame_height = int(restored_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_frame_size = (src_frame_width, src_frame_height)
ref_frame_size = (ref_frame_width, ref_frame_height)
if src_frame_size != ref_frame_size:
    raise ValueError("Unable to do video registration on different frame sizes")


# Prepare registration algorithms. Candidate approaches:
#
# 1. feature keypoint matching + global transformation matrix estimation
# ====> ??
# 2. motion estimation & compensation
# ====> Will try later
# 3. other registration techniques (median threshold bitmap, cv2.reg, etc ...)
#    (NOTE: cv2.reg requires extra modules ...)


class ImageRegistrationAligner:
    def __init__(self):
        pass

    def computeSrcToDstMap(src_img, dst_img):
        raise NotImplementedError


# Method 1.
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
        self.flann = cv2.FlannBasedMatcher_create()

    def computeSrcToDstMap(self, src_img, dst_img):
        src_keypts, src_feats = self.orb.detectAndCompute(src_img, mask=None)
        dst_keypts, dst_feats = self.orb.detectAndCompute(dst_img, mask=None)
        print("src_feats:", len(src_feats))
        print("dst_feats:", len(dst_feats))
        matches = self.bf.match(src_feats, dst_feats)
        # matches = self.flann.knnMatch(src_feats, dst_feats, k=2)

        # Store all the good matches as per Lowe's ratio test.
        # map_pairs = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         map_pairs.append(m)
        map_pairs, num_pairs = matches, len(matches)

        # print("num_pairs:", num_pairs)
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
        # draw_params = dict(
        #     matchColor=(0, 255, 0),  # draw matches in green color
        #     singlePointColor=None,
        #     matchesMask=mask.ravel().tolist(),  # draw only inliers
        #     flags=2,
        # )
        # viz_matches = cv2.drawMatches(
        #     src_img, src_keypts, dst_img, dst_keypts, map_pairs, None, **draw_params
        # )
        viz_matches = None

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

        return map_x, map_y, viz_matches


# Method 2. (implementing)
# See: https://stackoverflow.com/questions/43512039
class MotionEstimatedAligner(ImageRegistrationAligner):
    def __init__(self):
        pass

    def computeSrcToDstMap(src_img, dst_img):
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        cv2.calcOpticalFlowFarneback(src_img, dst_img)

        raise NotImplementedError


# For each frame pair, estimate registration and then compensate to minimize
# frame pair residual/difference
img_reg = OrbFeatureAligner()
num_frames = original_vid_length = restored_vid_length
frame_size = src_frame_size = ref_frame_size
w, h = frame_size
shrink_2x = w > 1920 or h > 1000
viz_size = (w // 2 if shrink_2x else w, h // 2 if shrink_2x else h)

num_compen_ok = 0
for i in range(num_frames):
    src_ok, src_img = original_vid.read()
    dst_ok, dst_img = restored_vid.read()
    if not src_ok or not dst_ok:
        break
    print("Aligning frame {:4d} ...".format(i))

    map_x, map_y, viz_matches = img_reg.computeSrcToDstMap(src_img, dst_img)
    map_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LANCZOS4)

    no_comp = cv2.addWeighted(dst_img, 0.5, src_img, 0.5, 0)
    comp_on = cv2.addWeighted(dst_img, 0.5, map_img, 0.5, 0)
    diff_no_comp = cv2.absdiff(dst_img, src_img)
    diff_comp_on = cv2.absdiff(dst_img, map_img)
    sad_no_comp, sad_comp_on = np.sum(diff_no_comp), np.sum(diff_comp_on)

    print(
        "no compensation SAD:",
        sad_no_comp,
        "compensation on SAD:",
        sad_comp_on,
        "Result:",
        "OK" if sad_comp_on < sad_no_comp else "NG",
    )
    if sad_comp_on < sad_no_comp:
        num_compen_ok += 1

    viz_aligned = np.hstack((no_comp, comp_on))
    viz_absdiff = np.hstack((diff_no_comp, diff_comp_on))
    viz_resized = cv2.resize(np.vstack((viz_aligned, viz_absdiff)), viz_size)

    cv2.imshow("video registration", viz_resized)
    cv2.waitKey(delay=40)

print(
    "registration success rate: {:.2f}% (={:4d}/{:4d})".format(
        100 * num_compen_ok / num_frames, num_compen_ok, num_frames
    )
)

# def decode_fourcc(cc):
#     # See: https://stackoverflow.com/questions/49138457
#     return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

# def encode_fourcc(c1, c2, c3, c4):
#     return cv2.VideoWriter_fourcc(c1, c2, c3, c4)

# codec_string = decode_fourcc(original_vid.get(cv2.CAP_PROP_FOURCC))  # 'acpn'

# We can read 'acpn' = Apple ProRes 422 Standard Definition
# But we cannot write 'acpn' file due to Apple-specific codec ....
# NOTE: 'acpn' is often used in film industry due to its default encoding preset
#       is good at preserving color quality to be used in post-production
#       editing software, where h.264/h.265 requires fine tuned encoding
#       parameters ...
#
# NOTE: So our output video is merely a visualization check ...
# TODO: implement quad view of alignment check

# frame_rate = int(original_vid.get(cv2.CAP_PROP_FPS))
# frame_width = int(original_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(original_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_size = (frame_width, frame_height)
# fourcc_codec = encode_fourcc(*"AVdh")

# dst_vid_path = src_vid_path.with_stem(f"{src_vid_path.stem}-aligned")
# orig_aligned = cv2.VideoWriter(str(dst_vid_path), fourcc_codec, frame_rate, frame_size)
