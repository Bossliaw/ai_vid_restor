import numpy as np
import cv2

import vid_frame_reg as img_reg
import multiprocessing


def align_frame_pair(original_vid, restored_vid, fitter, frame_idx, vid_length):
    if frame_idx >= vid_length:
        return None, "Unable to read frame beyond video length"
    original_vid.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    restored_vid.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    src_ok, src_img = original_vid.read()
    dst_ok, dst_img = restored_vid.read()
    if not src_ok or not dst_ok:
        return None, "Unable to read frame from either video source"

    reg_img = img_reg.align_src_img_to_dst(src_img, dst_img, fitter)
    if reg_img is None:
        return src_img, "Bad alignment"
    else:
        return reg_img, "OK"


# DAMN: multi-processing is not trivial to parallel cv2.VideoCapture ...
class VideoPairFitBatchReader:
    def __init__(self, src_vid_path, ref_vid_path, fitter, batch_size=1):
        self.batch_size = 1 if batch_size < 1 else batch_size
        self.fitter = fitter

        self.src_vids = []
        self.ref_vids = []
        for i in range(self.batch_size):
            self.src_vids.append(cv2.VideoCapture(src_vid_path))
            self.ref_vids.append(cv2.VideoCapture(ref_vid_path))

        src_vid_dim = (
            int(self.src_vids[0].get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.src_vids[0].get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.src_vids[0].get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        ref_vid_dim = (
            int(self.ref_vids[0].get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.ref_vids[0].get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.ref_vids[0].get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        print("original video dimension:", src_vid_dim)
        print("restored video dimension:", ref_vid_dim)
        if src_vid_dim != ref_vid_dim:
            raise ValueError(
                "Unable to do registration due to different video dimension (length, width, height)"
            )
        self.vid_length = src_vid_dim[0]

        # Set up multi-process
        self.frame_pos = 0
        self.pool = multiprocessing.Pool()

    def get_video_length(self):
        return self.vid_length

    def get_current_frame_idx(self):
        return self.frame_pos

    def get_batch_size(self):
        return self.batch_size

    def pop_aligned_frame(self):
        # SUCK: preparing parameters in good old loop ...
        map_params = []
        for i in range(self.batch_size):
            map_params.append(
                (
                    self.src_vids[i],
                    self.ref_vids[i],
                    self.fitter,
                    self.frame_pos + i,
                    self.vid_length,
                )
            )
        aligned_frames = self.pool.map(align_frame_pair, map_params)
        self.frame_pos += self.batch_size
        return aligned_frames
