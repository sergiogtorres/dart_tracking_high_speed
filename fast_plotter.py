from matplotlib import pyplot as plt, cm
import numpy as np
import cv2

MODE_IMSHOW = 0
MODE_PLOT = 1     #for some reason, it's plotting on a matplotlib windows as well, together with opencv.
MODE_SCATTER = 2
class FastPlotter:
    def __init__(self, plot_names, data_shapes, window_name, xlims, ylims, nrows=2, ncols=4, figsize=(10,5), dpi=200, mode=MODE_IMSHOW):
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
        self.pl_list = []
        self.window_name = window_name
        self.prev_txt = None
        self.mode = mode

        if type(self.axs) == type([]):
            for ax_row, title_row in zip(self.axs, plot_names, xlims, ylims):
                for ax, title in zip(ax_row, title_row):
                    ax.set_title(title)
                    ax.set_xlim([current_xmin, current_xmax])
                    ax.set_ylim([current_ymin, current_ymax])


        else:
            for ax, title, data_shape in zip(self.axs, plot_names, data_shapes):
                ax.set_title(title)
                if self.mode == MODE_IMSHOW:
                    random_arr = np.random.random(data_shape)
                    self.pl_list.append(ax.imshow(random_arr, cmap=cm.coolwarm))  # for some reason, if it's a 0 array it doesn't work
                elif self.mode == MODE_PLOT:
                    random_arr = np.array([0,0])
                    self.pl_list += (ax.plot(random_arr,random_arr))  # plot comes out as a 1 element list already
                elif self.mode == MODE_SCATTER:
                    random_arr = np.array([0, 0])
                    self.pl_list += [ax.scatter(random_arr, random_arr)]  # plot comes out as a 1 element list already



        self.fig.canvas.draw()
        self.bg_axs_list = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axs]

        buf = self.fig.canvas.buffer_rgba()
        plot = np.asarray(buf)
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, plot)
        cv2.waitKey(1)

    def update_plot(self, data, figure_text):
        pct_big = 0.2
        pct_small = 0.1 #to check increase the xlim and ylim of the axes. Big and small to include hysteresis

        for plot, datum, ax, bg_ax in zip(self.pl_list, data, self.axs, self.bg_axs_list):

            if self.mode==MODE_SCATTER:
                plot.set_offsets(datum)
            else:
                plot.set_data(datum)

            self.fig.canvas.restore_region(bg_ax)
            current_xmin, current_xmax = ax.get_xlim()
            current_ymin, current_ymax = ax.get_ylim()
            xmax = datum[0,:].max()
            xmin = datum[0,:].min()
            range_x = xmax - xmin
            ymax = datum[1,:].max()
            ymin = datum[1,:].min()
            range_y = ymax - ymin

            # There has to be a smarter way, but this is good for now
            if current_xmin > xmin-pct_small*range_x:
                current_xmin = xmin - pct_big*range_x
            if current_xmax < xmax+pct_small*range_x:
                current_xmax = xmax + pct_big*range_x

            if current_ymin > ymin-pct_small*range_y:
                current_ymin = ymin - pct_big*range_y
            if current_ymax < ymax+pct_small*range_y:
                current_ymax = ymax + pct_big*range_y

            range_x_updated = current_xmax - current_xmin
            range_y_updated = current_ymax - current_ymin

            # --- ensure the plot is square (again, there's probably a better way to do this)
            diff = range_x_updated - range_y_updated
            if diff > 0:    # if y is smaller, enlarge y
                current_ymin -= diff / 2
                current_ymax += diff / 2
            else:           # if x is smaller, enlarge x
                diff = np.abs(diff)
                current_xmin -= diff / 2
                current_xmax += diff / 2

            ax.set_xlim([current_xmin, current_xmax])
            ax.set_ylim([current_ymin, current_ymax])
            ax.draw_artist(plot)
            self.fig.canvas.blit(ax.bbox)

        buf = self.fig.canvas.buffer_rgba()
        plot = np.asarray(buf)
        if self.prev_txt is not None:
            cv2.putText(plot, self.prev_txt, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.putText(plot, figure_text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, plot)
        cv2.waitKey(1)
        #cv2.waitKey(int(0.02 * 1000))
        self.prev_txt = figure_text
        return plot

