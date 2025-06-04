import time
from datetime import datetime
import pyrealsense2 as rs
import numpy as np
import cv2
import glfw
from OpenGL.GL import *



W = 848
H = 100
N = 1000
# --- data recording
all_frames_ir_left = np.zeros((N, H, W), dtype="uint8")
all_frames_ir_right = np.zeros((N, H, W), dtype="uint8")
all_frames_ir_depth = np.zeros((N, H, W), dtype="uint16")
timestamps = np.zeros(N)
data_arrays = [all_frames_ir_left, all_frames_ir_right, all_frames_ir_depth, timestamps]


# --- REALSENSE INIT ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 300)
config.enable_stream(rs.stream.infrared, 1, 848, 100, rs.format.y8, 300)
config.enable_stream(rs.stream.infrared, 2, 848, 100, rs.format.y8, 300)
profile = pipeline.start(config)


# --- SET MANUAL EXPOSURE ---
device = profile.get_device()
exposure_us = 500
extra_str = f"{exposure_us}_us_sunlight"
# Stereo Module
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.enable_auto_exposure):
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto
    depth_sensor.set_option(rs.option.exposure, exposure_us)  # Set exposure in microseconds (1000 us = 1 ms)

# --- GET EXPOSURE ---
device = profile.get_device()
for sensor in device.query_sensors():
    name = sensor.get_info(rs.camera_info.name)
    if sensor.supports(rs.option.exposure):
        exposure = sensor.get_option(rs.option.exposure)
        print(f"{name} sensor exposure: {exposure} µs")


# Enable emitter (IR projector)
device = profile.get_device()
depth_sensor = device.query_sensors()[0]  # Usually depth sensor is first

if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)  # 1 = ON, 0 = OFF
    print("IR projector enabled.")
else:
    print("Emitter control not supported on this device.")


# --- GLFW INIT ---
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")
window_width = W * 3
window_height = H
window = glfw.create_window(window_width, window_height, "RealSense IR + Depth Viewer", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create window")
glfw.make_context_current(window)
glfw.swap_interval(0)

# --- OPENGL TEXTURE SETUP ---
def create_texture():
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex_id


tex_ir_left = create_texture()
tex_ir_right = create_texture()
tex_depth = create_texture()


frame_count = 0
time_prev = time.time()
def upload_image(tex_id, image):
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0,
                 GL_RGB, GL_UNSIGNED_BYTE, image)

def draw_textured_quad(x_offset, tex_id):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(-1 + x_offset, -1)
    glTexCoord2f(1, 1); glVertex2f(-1 + x_offset + 2/3, -1)
    glTexCoord2f(1, 0); glVertex2f(-1 + x_offset + 2/3,  1)
    glTexCoord2f(0, 0); glVertex2f(-1 + x_offset,  1)
    glEnd()

# --- MAIN LOOP ---
try:
    while not glfw.window_should_close(window):
        frames = pipeline.wait_for_frames()
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)
        depth = frames.get_depth_frame()

        if not (ir1 and ir2 and depth):
            continue

        frame_count += 1
        if frame_count % 100 == 0:
            now = time.time()
            fps = 100 / (now - time_prev)
            print(f"Effective FPS: {fps:.2f}")
            time_prev = now

        # Convert IR frames (8-bit) to 3-channel RGB
        ir1_np = np.asanyarray(ir1.get_data())
        ir2_np = np.asanyarray(ir2.get_data())
        ir1_rgb = cv2.cvtColor(ir1_np, cv2.COLOR_GRAY2RGB)
        ir2_rgb = cv2.cvtColor(ir2_np, cv2.COLOR_GRAY2RGB)


        # Convert depth frame to RGB using color map
        depth_np = np.asanyarray(depth.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)

        # Save frames and timestamps into np arrays
        all_frames_ir_left[frame_count % N]     = ir1_np
        all_frames_ir_right[frame_count % N]    = ir2_np
        all_frames_ir_depth[frame_count % N]    = depth_np
        timestamps[frame_count % N]             = ir1.get_timestamp()

        # Upload all images as textures
        upload_image(tex_ir_left, ir1_rgb)
        upload_image(tex_ir_right, ir2_rgb)
        upload_image(tex_depth, depth_colormap)

        # Draw all 3 images side-by-side
        glClear(GL_COLOR_BUFFER_BIT)
        draw_textured_quad(0.0, tex_ir_left)
        draw_textured_quad(2/3, tex_ir_right)
        draw_textured_quad(4/3, tex_depth)

        glfw.swap_buffers(window)
        glfw.poll_events()
except Exception as e:
    print("Error:", e)
finally:
    pipeline.stop()
    glfw.terminate()
    datetime_str = timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arrays_{extra_str}_len_{len(data_arrays)}_{timestamp}.npz"
    np.savez(filename, *data_arrays)