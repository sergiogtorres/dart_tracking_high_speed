import time
import pyrealsense2 as rs
import numpy as np
import cv2
import glfw
from OpenGL.GL import *
from matplotlib import pyplot as plt
import matplotlib
import utils
import openGL_utils

matplotlib.use("QtAgg")

W = 848
H = 100
N = 1000
# --- data recording
file_name = "arrays_300_us_sunlight_len_4_20250604_164058.npz"#"arrays_500_len_4_20250604_141405.npz"#"arrays_len_4_20250604_120527.npz"
loaded = np.load(f'F:/projects/4. learning/Intel_realsense_tutorials/300_fps_testing/{file_name}')

all_frames_ir_left = loaded['arr_0']
all_frames_ir_right = loaded['arr_1']
all_frames_ir_depth = loaded['arr_2']
timestamps = loaded['arr_3']

playback_x = 100
frame_num_min = 900#2150
frame_num_max = 999#2200

EQUALIZE_IMAGES = True


if EQUALIZE_IMAGES:
    all_frames_ir_left = utils.histogram_equalization(all_frames_ir_left)
    all_frames_ir_right = utils.histogram_equalization(all_frames_ir_right)


# --- the arrays are circular, i.e., data is recorded and overwritten sequentially as we reach the end of the array
# --- roll the arrays so the first index corresponds to the earliest timestamp
timestamps = timestamps - np.min(timestamps)
tiemstamps_original = np.copy(timestamps)
zeroth_idx = np.argmin(timestamps)
timestamps = np.roll(timestamps, -zeroth_idx)
delta_ts = timestamps[1:] - timestamps[:-1]

assert np.all(delta_ts >= 0)
all_frames_ir_left = np.roll(all_frames_ir_left, -zeroth_idx, axis=0)
all_frames_ir_right = np.roll(all_frames_ir_right, -zeroth_idx, axis=0)
all_frames_ir_depth = np.roll(all_frames_ir_depth, -zeroth_idx, axis=0)
# ----------------------------------


# --- GLFW INIT ---
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")
side_by_side_frames = 3
window_width = W * side_by_side_frames
window_height = H
window = glfw.create_window(window_width, window_height, "RealSense IR + Depth Viewer", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create window")
glfw.make_context_current(window)
glfw.swap_interval(0)

# --- OPENGL TEXTURE SETUP ---

tex_ir_left = openGL_utils.create_texture()
tex_ir_right = openGL_utils.create_texture()
tex_depth = openGL_utils.create_texture()



frame_count = 0
time_prev = time.time()



len_frames = frame_num_max-frame_num_min
prev_timestamp = None
# --- MAIN LOOP ---
try:
    while not glfw.window_should_close(window):


        frame_count += 1
        frame_count_wrapped = frame_num_min + frame_count % len_frames
        print(f"frame:{frame_count_wrapped}")
        if frame_count % 100 == 0:
            now = time.time()
            fps = 100 / (now - time_prev)
            print(f"Effective FPS: {fps:.2f}")
            time_prev = now

        # Convert IR frames (8-bit) to 3-channel RGB
        ir1_np = all_frames_ir_left[frame_count_wrapped].astype(np.uint8)
        ir2_np = all_frames_ir_right[frame_count_wrapped].astype(np.uint8)
        depth_np = all_frames_ir_depth[frame_count_wrapped].astype(np.uint16)
        timestamp = timestamps[frame_count_wrapped]
        ir1_rgb = cv2.cvtColor(ir1_np, cv2.COLOR_GRAY2RGB)
        ir2_rgb = cv2.cvtColor(ir2_np, cv2.COLOR_GRAY2RGB)


        # Convert depth frame to RGB using color map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)

        # Upload all images as textures
        openGL_utils.upload_image(tex_ir_left, ir1_rgb)
        openGL_utils.upload_image(tex_ir_right, ir2_rgb)
        openGL_utils.upload_image(tex_depth, depth_colormap)

        # Draw all 3 images side-by-side
        glClear(GL_COLOR_BUFFER_BIT)
        openGL_utils.draw_textured_quad(0.0, side_by_side_frames, tex_ir_left)
        openGL_utils.draw_textured_quad(1, side_by_side_frames, tex_ir_right)
        openGL_utils.draw_textured_quad(2, side_by_side_frames, tex_depth)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # --- TIMING CONTROL ---
        if prev_timestamp is not None:
            dt = playback_x * (timestamp - prev_timestamp) / 1000.0  # Convert to seconds
            time.sleep(max(dt, 0))  # sleep at least 0
        prev_timestamp = timestamp

        prev_timestamp = timestamp
except Exception as e:
    print("Error:", e)
finally:
    glfw.terminate()
