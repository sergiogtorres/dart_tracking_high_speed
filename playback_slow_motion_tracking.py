import time
import pyrealsense2 as rs
import numpy as np
import cv2
import glfw
from OpenGL.GL import *
from matplotlib import pyplot as plt
import matplotlib
import tools.utils as utils
import tools.openGL_utils as openGL_utils

matplotlib.use("QtAgg")

W = 848
H = 100
N = 1000

SAFE_BORDER_MARGIN = 30
# --- data recording
file_name = "arrays_300_us_sunlight_len_4_20250604_164058.npz"#"arrays_500_len_4_20250604_141405.npz"#"arrays_len_4_20250604_120527.npz"
loaded = np.load(f'data/{file_name}')

all_frames_ir_left = loaded['arr_0']
all_frames_ir_right = loaded['arr_1']
all_frames_ir_depth = loaded['arr_2']
timestamps = loaded['arr_3']

K = np.array([[423.22164917,   0.        , 431.39187622],
              [  0.        , 423.22164917,  50.72911072],
              [  0.        ,   0.        ,   1.        ]])

K_inv = np.linalg.inv(K)

playback_x = 100
frame_num_min = 900#2150
frame_num_max = 999#931#2200

EQUALIZE_IMAGES = True
#equalization_func = utils.histogram_equalization
equalization_func = utils.histogram_equalization_simple

if EQUALIZE_IMAGES:
    all_frames_ir_left = equalization_func(all_frames_ir_left, extra=5)
    all_frames_ir_right = equalization_func(all_frames_ir_right, extra=5)


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
## -- Initialize currently_tracking variables
dart_uv = np.array([0, 0, 1])[:, np.newaxis]

##########################################################
# --- GLFW INIT ---
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

# I want them one on top of each other now
#side_by_side_frames = 4
#window_width = W * side_by_side_frames
#window_height = H
#window = glfw.create_window(window_width, window_height, "RealSense IR + Depth Viewer", None, None)

vertically_stacked_frames = 4
window_width = W
window_height = H * vertically_stacked_frames
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
tex_fgmask = openGL_utils.create_texture()
##################################################################


frame_count = 0
time_prev = time.time()

# --- Initialize background subtraction

fgbg = cv2.createBackgroundSubtractorMOG2()
n_close = 3
n_open = 30
kernel_open = np.ones((n_close, n_close), np.uint8)
kernel_close = np.ones((n_open, n_open), np.uint8)
# --------------------------------------------
len_frames = frame_num_max-frame_num_min
prev_timestamp = None
currently_tracking = False
has_been_tracked = False

fgmask = np.zeros((H, W), dtype="uint16")
fgmask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)

XYZ = None
vel = None
XYZ_array = np.zeros((3,1000))
# ____________________________
# --- Initialize Kalman Filter
# ____________________________
import es_ekf_tracking
import fast_plotter

DO_MEASUREMENT_UPDATE = True

# current_time = 0
dt_recorded = None
run = True
kalman_filterer = es_ekf_tracking.EsExKalmanFilter(debugging_mode=False, start_at_k=0)

plot_names = ["xy", "xz", "yz", "tx"]  # ["trajectory x", "trajectory y", "trajectory z", "xy", "xz", "yz"]
data_shapes = [None] * 4
xlims = [-1, 1]
ylims = [-1, 1]


plotter = fast_plotter.FastPlotter(plot_names=plot_names,
                      data_shapes=data_shapes,
                      window_name="plotting_simulation",
                      xlims=xlims, ylims=ylims,
                      nrows=1, ncols=4, figsize=(12, 2), dpi=100, mode=fast_plotter.MODE_PLOT,
                      desired_window_width = None)

# --- Other variables
erosion_mask_depth_contour = np.ones((5,5))
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

        # --- TIMING CONTROL ---
        if prev_timestamp is not None:
            dt_recorded = timestamp - prev_timestamp
            dt_playback = playback_x * (dt_recorded) / 1000.0  # Convert to seconds
            print(dt_playback)
            time.sleep(max(dt_playback, 0))  # sleep at least 0
        prev_timestamp = timestamp

        # getting foreground mask
        fgmask = fgbg.apply(ir1_np)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

        fgmask[:, :SAFE_BORDER_MARGIN] *= 0   # clipping anything outside the safe border area
        fgmask[:, -SAFE_BORDER_MARGIN:] *= 0

        # --- find contours in the fgmask
        fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        contours, hierarchy = cv2.findContours(fgmask, 1, 2)
        isolated_dart_mask = np.zeros_like(ir1_np, dtype="uint8")
        currently_tracking =    False
        area_in_range =         False
        aspect_ratio_in_range = False
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            #box_plot = [box.reshape(-1,1,2).astype("int32")]
            #fgmask_color_contour = cv2.drawContours(fgmask_color, box_plot,0,(0,0,255),2)
            #plt.figure("rotated bounding box")
            #plt.imshow(fgmask_color_contour)
            side_lengths = [
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[1] - box[2]),
                #np.linalg.norm(box[2] - box[3]),
                #np.linalg.norm(box[3] - box[0]),
            ]
            ratio = side_lengths[1] / side_lengths[0]

            # filter small detections
            area = cv2.contourArea(contour)#utils.polygon_area(box)
            print(f"area:, {area}")
            # rough way to check if it's a NERF dart
            if 200 < area < 5000:
                area_in_range = True
            else:
                area_in_range = False
            if ratio < 1:               # in case the axes are swapped
                ratio = 1/ratio
            #print(f"\t\t\t\tratio:{ratio}")
            if 4 < ratio < 8:
                # we found the nerf dart
                aspect_ratio_in_range = True
            else:
                aspect_ratio_in_range = False

            currently_tracking = area_in_range & aspect_ratio_in_range # check for both conditions

            if currently_tracking:
                has_been_tracked = True # This variable is set forever as True once tracking has started

                # if currently_tracking, we have found the projectile -> stop looking for other contours (break)
                break

        if currently_tracking:
            print("huh")
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            dart_uv[:2, 0] = cx, cy


            cv2.drawContours(isolated_dart_mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

            if ((SAFE_BORDER_MARGIN < cx < W - SAFE_BORDER_MARGIN) and
                (SAFE_BORDER_MARGIN/2 < cy < H - SAFE_BORDER_MARGIN/2)): # Ensure that we are within a safe image margin

                # -- Use found contour to get dart depth
                masked_depth = (isolated_dart_mask > 0) * depth_np
                #       Erode contour to ensure no outliers in depth
                #masked_depth = cv2.erode(masked_depth, erosion_mask_depth_contour)
                #avg_z_coord_NERF_dart = masked_depth[(masked_depth>0)].mean()
                #       OR, use median to remove outliers (possibly less robust, but less computationally expensive
                avg_z_coord_NERF_dart = np.median(masked_depth[(masked_depth>0)].flatten())

                # calculate the X, Y, Z position of the dart in 3D coordinates (camera frame of reference)
                XYZ = K_inv @ dart_uv
                XYZ /= XYZ[-1]
                XYZ *= (avg_z_coord_NERF_dart * 1e-3) # avg_z_coord_NERF_dart is in mm, we want m (SI)ç

                XYZ_array[:, frame_count] = XYZ[:,0]

                if frame_count > 0:
                    vel = ((XYZ_array[:, frame_count] - XYZ_array[:, frame_count-1]) / dt_recorded) * 1000 #timestamps in ms
                    vel = np.clip(vel, -20, 20)   # simple clipping to remove noise.
                                                   # Not robust, but good enough for debugging
                    # TODO: remove clipping of measured velocity

                print(f"depth:{avg_z_coord_NERF_dart}")
                #print(f"screen:  {cx, cy}")
                print(f"dart pos:{XYZ[:, 0]}")
                print(f"velocity:{vel}")

        fgmask_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(fgmask_rgb, [contour], contourIdx=-1, color=(255,0,0), thickness=1)
        cv2.drawContours(fgmask_rgb, [box.reshape(4,1,2).astype(np.int32)], contourIdx=-1, color=(0,255,0), thickness=1)

        # Convert depth frame to RGB using color map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)

        cv2.drawContours(depth_colormap, [contour], contourIdx=-1, color=(0,255,0), thickness=1)


        # Upload all images as textures
        openGL_utils.upload_image(tex_ir_left, ir1_rgb)
        openGL_utils.upload_image(tex_ir_right, ir2_rgb)
        openGL_utils.upload_image(tex_depth, depth_colormap)
        openGL_utils.upload_image(tex_fgmask, fgmask_rgb)

        # Draw all 3 images side-by-side
        glClear(GL_COLOR_BUFFER_BIT)
        #horizontal drawing
        #openGL_utils.draw_textured_quad(0, side_by_side_frames, tex_ir_left)
        #openGL_utils.draw_textured_quad(1, side_by_side_frames, tex_ir_right)
        #openGL_utils.draw_textured_quad(2, side_by_side_frames, tex_depth)
        #openGL_utils.draw_textured_quad(3, side_by_side_frames, tex_fgmask)

        openGL_utils.draw_textured_quad(0, vertically_stacked_frames, tex_ir_left, layout="vertical")
        openGL_utils.draw_textured_quad(1, vertically_stacked_frames, tex_ir_right, layout="vertical")
        openGL_utils.draw_textured_quad(2, vertically_stacked_frames, tex_depth, layout="vertical")
        openGL_utils.draw_textured_quad(3, vertically_stacked_frames, tex_fgmask, layout="vertical")

        glfw.swap_buffers(window)
        glfw.poll_events()

        # --------------------------------------------------------------------------------------------------------------
        # -- Kalman Filter stuff & trajectory plotting

        print(f"dt_recorded:{dt_recorded}\n"
              f"has_been_tracked:{has_been_tracked}\n"
              f"vel:{vel}")
        if (dt_recorded is not None and has_been_tracked and vel is not None):
            print("entered main if statement")
            kalman_filterer.update(dt_recorded/1000)
            # TODO: automate recording & use of units (ms, 100s of um, etc.) from realsense camera during capture

            if DO_MEASUREMENT_UPDATE:

                if currently_tracking: # if tracking is currently active, do measurement update

                    y_k = np.hstack((XYZ.squeeze(), vel))

                    ## Correction step with measurement

                    kalman_filterer.measurement_update(y_k)
                    # TODO: something is wrong with the measurement update. E.G.
                    #y_k_pred: [0.21825032 - 0.00217293  0.27747229  0.1580695 - 0.01175899  0.30510076]
                    #y_k: [1.17086510e-01 - 1.17631965e-02  3.38000000e-01 - 1.61060466e+01
                    #      - 7.28147377e-02  2.09223584e+00]
                    #error
                    #state:
                    #delta_p_k_hat: [-0.09345169 - 0.00885916  0.05591344]
                    #delta_v_k_hat: [-7.16015078e-05 - 3.68036540e-06  2.61698073e-05]
                    #true
                    #state:
                    #p_hat: [0.12479864 - 0.01103209  0.33338572]
                    #v_hat: [0.1579979 - 0.01176267  0.30512693]
                    #p_hat, v_hat, P_cov_hat = measurement_update(R_k, P_cov[k], y_k, p_est[k], v_est[k], H_k)
                    ##^^^here, using P_cov[k] since we are CORRECTING wrt the current iteration



            #xt = np.vstack((times[:k+1], p_est[:k+1, 0]))
            # yt = np.vstack((times[:k+1], p_est[:k+1, 1]))
            # zt = np.vstack((times[:k+1], p_est[:k+1, 2]))
            xy = kalman_filterer.p_est[:kalman_filterer.k + 1, [0, 1]].T
            xz = kalman_filterer.p_est[:kalman_filterer.k + 1, [0, 2]].T
            yz = kalman_filterer.p_est[:kalman_filterer.k + 1, [1, 2]].T

            tx = np.vstack((kalman_filterer.times[:kalman_filterer.k + 1], kalman_filterer.p_est[:kalman_filterer.k + 1, 0]))

            state = np.hstack((kalman_filterer.p_est[kalman_filterer.k], kalman_filterer.v_est[kalman_filterer.k]))
            figure_text1 = (f"last_measurement: pos: {np.round(y_k[:3], 2)}, vel:{np.round(y_k[3:], 2)}    "                           
                           f"currently_tracking:{currently_tracking}")
            figure_test2 = f"last_state      : pos: {np.round(state[:3], 2)}, vel:{np.round(state[3:], 2)}    "
            plot_img = plotter.update_plot([xy, xz, yz, tx], figure_text1, figure_test2)  # [xt, yt, zt, xy, xz, yz]
            print(f"plot updated up to index: {kalman_filterer.k}")
            print(f"x positions")
            print(f"{kalman_filterer.p_est[:kalman_filterer.k + 1, 0]}")
            print(f"x velocities")
            print(f"{kalman_filterer.v_est[:kalman_filterer.k + 1, 0]}")


            kalman_filterer.update_time_index(dt_recorded/1000) # Kalman Filter class uses time in s (SI)
        # --------------------------------------------------------------------------------------------------------------
        if kalman_filterer.k == 910:
            res = plot_img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('output.mp4', fourcc, 5, res)
        if kalman_filterer.k >= 910:
            writer.write(plot_img)
            #cv2.imwrite(plot_img)




except Exception as e:
    print("Error:", e)
finally:
    glfw.terminate()
    writer.release()
