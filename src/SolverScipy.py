import bz2
import numpy as np
from scipy.sparse import lil_matrix


def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def name(filename):
  k = 0
  while filename[k] != '/':
    k += 1
  l = k + 8
  while filename[l] != 'p':
    l += 1
  return filename[: k-1] + filename[k + 8 : l - 2]

import time
from scipy.optimize import least_squares

tab = np.empty((1,6), dtype=object)
dic_status= {-1 : "error", 0 : "max funct eval", 1 : "||Jtr|| < gtol", 2 : "dF < ftol * F", 3 : "||d|| < xtol * (xtol + ||x||)", 4 : "xtol and ftol"}
problems = ["LadyBug/problem-49-7776-pre.txt.bz2",
              "LadyBug/problem-73-11032-pre.txt.bz2"]
# problems = ["LadyBug/problem-49-7776-pre.txt.bz2",
#               "LadyBug/problem-73-11032-pre.txt.bz2",
#               "LadyBug/problem-138-19878-pre.txt.bz2",
#               "LadyBug/problem-318-41628-pre.txt.bz2",
#               "LadyBug/problem-460-56811-pre.txt.bz2",
#               "LadyBug/problem-646-73584-pre.txt.bz2",
#               "LadyBug/problem-810-88814-pre.txt.bz2",
#               "LadyBug/problem-1031-110968-pre.txt.bz2",
#               "LadyBug/problem-1235-129634-pre.txt.bz2",
#               "Dubrovnik/problem-202-132796-pre.txt.bz2",
#               "LadyBug/problem-1723-156502-pre.txt.bz2",
#               "Dubrovnik/problem-273-176305-pre.txt.bz2",
#               "Dubrovnik/problem-356-226730-pre.txt.bz2",
#               "Venice/problem-427-310384-pre.txt.bz2",
#                "Venice/problem-1350-894716-pre.txt.bz2"]
nb_pb = length(problems)

for k in range(nb_pb):
  camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data("../Data/" + problems[k])
  n_cameras = camera_params.shape[0]
  n_points = points_3d.shape[0]
  x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
  A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

  t0 = time.time()
  res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, gtol=1e-6, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
  t1 = time.time()

  tab[k, 1] = name(problems[k])
  tab[k, 1] = str(res.cost)
  tab[k, 2] = str(t1 - t0)
  tab[k, 3] = str(res.nfev)
  tab[k, 4] = dic_status[res.status]
  tab[k, 5] = str(res.optimality) + " \\ \n"

np.savetxt('scipy_results', tab, delimiter=' & ', fmt="%s")
