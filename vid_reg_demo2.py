import sys
import pathlib
import numpy as np
import cv2

import vid_frame_reg
import vid_batch_reg

# Load original/restored video
src_vid_path = str(pathlib.Path(sys.argv[1]).resolve())
ref_vid_path = str(pathlib.Path(sys.argv[2]).resolve())
vid_pair_reg = vid_batch_reg.VideoPairFitBatchReader(
    src_vid_path, ref_vid_path, vid_frame_reg.OrbFeatureAligner(), batch_size=2
)


# For each frame pair, estimate registration and then compensate to minimize
# frame pair residual/difference
num_frames = vid_pair_reg.get_video_length()
num_batches = num_frames // vid_pair_reg.get_batch_size()
num_compen_ok = 0
for i in range(num_batches):
    print("Batch align {} ...".format(i))
    for img, msg in vid_pair_reg.pop_aligned_frame():
        if img is None:
            break
        num_compen_ok += 1
        cv2.imshow("registration", cv2.resize(img, (1024, 778)))
        cv2.waitKey(delay=20)

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
