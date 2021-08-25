import sys
import pathlib
import numpy as np
import cv2

import vid_frame_reg as img_reg

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

h, w = src_frame_height, ref_frame_width
while w > 1920 or h > 1000:
    h, w = h // 2, w // 2
dbg_viz_size = (w, h)

# For each frame pair, estimate registration and then compensate to minimize
# frame pair residual/difference
img_fitter = img_reg.OrbFeatureAligner()
num_frames = original_vid_length = restored_vid_length
num_compen_ok = 0
for i in range(num_frames):
    src_ok, src_img = original_vid.read()
    dst_ok, dst_img = restored_vid.read()
    if not src_ok or not dst_ok:
        break

    print("Aligning frame {:4d} ...".format(i))
    reg_img = img_reg.align_src_img_to_dst(
        src_img, dst_img, img_fitter, dbg_viz_size=dbg_viz_size
    )

    if reg_img is not None:
        num_compen_ok += 1

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
