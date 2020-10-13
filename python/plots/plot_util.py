import numpy as np
import matplotlib.pyplot as plt

def draw_2d_covariance(mean, covmatrix, chisquare_val=2.4477, return_raw=False, *args, **kwargs):
    (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covmatrix)
    phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

    a = chisquare_val * np.sqrt(largest_eigval)
    b = chisquare_val * np.sqrt(smallest_eigval)

    ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi, num=200))
    ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi, num=200))

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
    if return_raw:
        return mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1]
    else:
        return plt.plot(mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1], *args, **kwargs)


def build_to_cov(ews, angle):
    rot_mat = np.array([[np.cos(angle), - np.sin(angle)],
                        [np.sin(angle),   np.cos(angle)]])
    return rot_mat @ np.eye(2) * ews @ rot_mat.T