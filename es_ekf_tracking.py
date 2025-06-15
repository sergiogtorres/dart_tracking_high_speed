import pickle
import time

import fast_plotter
#from utils import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

from fast_plotter import FastPlotter

# Testing with a simple model

# Linear process model:
#
#x_k = F_k_prev @ x_k_prev + G_k_prev @ u_k_prev + w_k_prev
#
#x_k: state
#u_k: inputs
#w_k: process noise
#F_k: prediction matrix
#G_k: control matrix
#
#Linear measurement model:
#
#y_k = H_k @ x_k + v_k
#
#y_k: measurement
#v_k: measurement noise
#H_k: measurement matrix


#EKF
# Linearized process/motion model
#x_k = f_k_prev(x_hat_k_prev, u_k_prev, 0) + F_k_prev @ (x_k_prev - x_hat_k_prev) + L_k_prev @ w_k_prev
#
# Linearized measurement model
#y_k = h_k(h_check_k,0) + H_k (x_k-x_check_k) + M_k @ v_k
#
# Prediction
# x_check_k = f_k_prev(x_hat_k_prev, u_k_prev, 0) (these zeros mean a 0 vector noise)
# P_k = F_k_prev @ P_hat_k_prev @ F_k_prev.T + L_k_prev @ Q_k_prev @ L_k_prev.T
#
# Optimal gain
# K_k = P_check_k @ H_k.T @ np.linalg.inv(H_k @ P_check_k @ H_k.T + m_k @ R_k @ M_k.T)
#
# Correction
# x_hat_k = x_check_k + K_k @ (y_k - h_k(x_check_k, 0))
# P_hat_k = (I - K_k @ H_k) @ P_check_k
#
# x_check_k: Prediction, from motion model at time k
# x_hat_k: Corrected prediction, with measurement at time k


# State defined as:
# x = [x, y, z, vx, vy, vz, ax, ay, az]
# Nonlinear process model f_k:
# x_k = f_k_prev(x_k_prev)
# x_k = [x, y, z] + [vx, vy, vz] * Δt
#
# Control vector: gravity
# + (1/2) * [ax, ay, az] * Δt**2
#
# Measurement model:
# y_k = [I 0] @ x_k + v_k (we only measure position and frame-frame velocity.
#       [0 I]
#
# M_k = dh_k / d_v_k |_x_check_k,0 (the Jacobian of the measurement model w.r.t. the noise vector v_k
#                                   evaluated at constant x=x_check_k and v=0)
#
#
#
#
def get_measurement(current_time, v0, g_accel):
    pos = v0 * current_time + 1 / 2 * g_accel * current_time ** 2
    v = v0 + g_accel * current_time
    return np.hstack((pos, v))

class EsExKalmanFilter:
    def __init__(self, debugging_mode=True, start_at_k = 1):

        self.debugging_mode = debugging_mode
        # Constants:
        # -- Variances:
        self.var_pos = (1e-3)**2 # 1mm std
        self.var_speed = (1)**2 # 1 m/s std
        self.var_accel = (0.5)**2 # 0.5 m/s^2 std
        self.g_accel = np.array([0, 0, -9.81])

        #Q_k = np.diag([var_pos]*3 + [var_speed]*3 + [var_accel]*3)

        # -- Motion model

        # -- Measurement model
        self.var_meas_pos   = self.var_pos
        self.var_meas_speed = self.var_speed
        self.R_k = np.diag([self.var_meas_pos]*3 + [self.var_meas_speed]*3)
        self.M_k = np.eye(6) # the measurement model is simply linear
        self.H_k = np.eye(6)
        self.L_k = np.eye(6)

        # -- Initializing variables:
        self.k = start_at_k
        self.N = 1000
        self.x = np.array([0, 0, 0, 0, 0, 0])
        self.p_est = np.zeros((self.N, 3))
        self.v_est = np.zeros((self.N, 3))
        self.v_est_uncorrected = np.zeros((self.N, 3))
        self.p_est_uncorrected = np.zeros((self.N, 3))
        self.p_obs = np.zeros((self.N, 3))
        self.v_obs = np.zeros((self.N, 3))
        self.P_cov = np.zeros((self.N, 6, 6))
        self.times = np.zeros(self.N)
        self.current_time = 0

        #self.v_est[0] = self.v_est_uncorrected[0] = v0

    def update(self, delta_t):

        self.Q_k = delta_t ** 2 * np.diag([self.var_speed] * 3 + [self.var_accel] * 3) ** 2

        # 1. Update state. If we had e.g. IMU inputs, we would use those as the control vector
        self.accel = self.g_accel + (np.random.random(3) - 0.5) * 10  # + get_drag(v_est[k-1], c)
        #               ^^^^^^^^^^^^^^^^^^^^^^^random component
        v_increment = self.accel * delta_t
        p_increment = self.v_est[self.k - 1] * delta_t + 0.5 * self.accel * delta_t ** 2
        #
        if self.debugging_mode:
            # to simulate some randomness
            random_factor_1 = 1 + (np.random.random() - 0.5) * 10
            random_factor_2 = 1 + (np.random.random() - 0.5) * 10
            #print(f"random factors:{random_factor_1}, {random_factor_2}")
            v_increment_randomized = v_increment * (random_factor_1)
            p_increment_randomized = p_increment * (random_factor_2)

            new_vel = self.v_est[self.k - 1] + v_increment_randomized + (np.random.random(3) - 0.5) * 1
            new_p = self.p_est[self.k - 1] + p_increment_randomized + (np.random.random(3) - 0.5) * 0.1
            #                                                              #^^^^^^^^^^^random component


            self.v_est_uncorrected[self.k] = new_vel
            self.p_est_uncorrected[self.k] = new_p
        else:
            new_vel = self.v_est[self.k - 1] + v_increment
            new_p = self.p_est[self.k - 1] + p_increment

            self.v_est_uncorrected[self.k] = new_vel
            self.p_est_uncorrected[self.k] = new_p

        # Apply linearized motion model and compute updated Jacobians

        self.F_k_prev = np.zeros((6, 6))
        self.F_k_prev[:3, :3] = np.eye(3)
        self.F_k_prev[:3, 3:6] = delta_t * np.eye(3)
        self.F_k_prev[3:6, 3:6] = np.eye(3)

        # Propagate uncertainty

        # P = Fx @ P @ Fx.T + Fi @ Qi @ Fi.T
        self.P_cov[self.k] = self.F_k_prev @ self.P_cov[self.k - 1] @ self.F_k_prev.T + self.L_k @ self.Q_k @ self.L_k.T

        (self.p_est[self.k],
         self.v_est[self.k],
         self.P_cov[self.k]) = (self.p_est_uncorrected[self.k],
                           self.v_est_uncorrected[self.k],
                           self.P_cov[self.k])
        # TODO: this actually makes it so p_est, etc., point to p_est_uncorrected, so it's nonsense. np.copy() or remove


    def update_time_index(self, delta_t):
        self.k += 1
        self.current_time += delta_t
        self.times[self.k] = self.current_time




    def measurement_update(self, y_k):
        """
        "check" variables stay inside the context of this function, and are used to update class parameters.
        :param self:
        :param y_k:
        :return:
        """

        self.p_obs[self.k] = y_k[:3]
        self.v_obs[self.k] = y_k[3:]

        p_check = self.v_est[self.k]
        v_check = self.p_est[self.k]
        P_cov_check = self.P_cov[self.k]
        self.H_k = np.eye(6)  # np.zeros((6, 9))
        # H_k[0:3, 0:3] = np.eye(3)

        self.K_k = P_cov_check @ self.H_k.T @ np.linalg.inv(self.H_k @ P_cov_check @ self.H_k.T + self.R_k)

        # Get error state
        y_k_pred = np.hstack(
            (p_check, v_check))  # the predicted measurement is simply the predicted position! since the
        # measurement is directly the position itself!
        # nonetheless, I prefer to call this y_k_pred to give the code generality
        # at the expense of little effort


        delta_x_k_hat = self.K_k @ (y_k - y_k_pred)

        delta_p_k_hat = delta_x_k_hat[:3]
        delta_v_k_hat = delta_x_k_hat[3:6]

        # True state = predicted state + δx
        p_hat = p_check + delta_p_k_hat
        v_hat = v_check + delta_v_k_hat

        print(f"measurement update:\n"
              f"y_k_pred:{y_k_pred}\n"
              f"y_k:{y_k}\n"
              f"error state:\n"
              f"delta_p_k_hat:{delta_p_k_hat}\n"
              f"delta_v_k_hat:{delta_v_k_hat}\n"
              f"true state:\n"
              f"p_hat:{p_hat}\n"
              f"v_hat:{v_hat}\n")
        # Get corrected covariance

        K_at_H = self.K_k @ self.H_k
        P_cov_hat = (np.eye(K_at_H.shape[0]) - K_at_H) @ P_cov_check

        # update class parameters
        self.p_est[self.k], self.v_est[self.k], self.P_cov[self.k] = p_hat, v_hat, P_cov_hat


if __name__ == "__main__":
    # For testing
    v0 = np.array([20,0,0])

    DO_MEASUREMENT_UPDATE = True

    #current_time = 0
    run = True
    kalman_filterer = EsExKalmanFilter()

    plot_names = ["xy", "xz_bad", "yz_bad", "xy", "xz",
                       "yz"]  # ["trajectory x", "trajectory y", "trajectory z", "xy", "xz", "yz"]
    data_shapes = [None] * 6
    xlims = [-10, 10]
    ylims = [-10, 10]

    plotter = FastPlotter(plot_names=plot_names,
                          data_shapes=data_shapes,
                          window_name="plotting_simulation",
                          xlims=xlims, ylims=ylims,
                          nrows=1, ncols=6, figsize=(18, 3), dpi=200, mode=fast_plotter.MODE_PLOT)


    delta_t = 1/300
    while run:
        t_prev = time.time()
        kalman_filterer.update(delta_t)

        if DO_MEASUREMENT_UPDATE:
            ## Assume there is always a measurement
            y_k = get_measurement(kalman_filterer.current_time, v0, kalman_filterer.g_accel)
            # Adding some noise
            y_k += np.hstack(((np.random.random(3) - 0.5)*0.1, (np.random.random(3) - 0.5)*1)) #error ~1 m, 10m/s

            ## Correction step with measurement

            kalman_filterer.measurement_update(y_k)
            #p_hat, v_hat, P_cov_hat = measurement_update(R_k, P_cov[k], y_k, p_est[k], v_est[k], H_k)
            ##^^^here, using P_cov[k] since we are CORRECTING wrt the current iteration

        else:
            pass

        kalman_filterer.update_time_index(delta_t)

        # xt = np.vstack((times[:k+1], p_est[:k+1, 0]))
        # yt = np.vstack((times[:k+1], p_est[:k+1, 1]))
        # zt = np.vstack((times[:k+1], p_est[:k+1, 2]))
        xy = kalman_filterer.p_est[:kalman_filterer.k + 1, [0, 1]].T
        xz = kalman_filterer.p_est[:kalman_filterer.k + 1, [0, 2]].T
        yz = kalman_filterer.p_est[:kalman_filterer.k + 1, [1, 2]].T
        print(xy)
        xy_bad = kalman_filterer.p_est_uncorrected[:kalman_filterer.k + 1, [0, 1]].T
        xz_bad = kalman_filterer.p_est_uncorrected[:kalman_filterer.k + 1, [0, 2]].T
        yz_bad = kalman_filterer.p_est_uncorrected[:kalman_filterer.k + 1, [1, 2]].T

        plotter.update_plot([xy_bad, xz_bad, yz_bad, xy, xz, yz], f"")  # [xt, yt, zt, xy, xz, yz]
        #print("add p_obs and v_obs to plots")

        t_now = time.time()
        delta_t = t_now - t_prev
        #time.sleep(delta_t)
