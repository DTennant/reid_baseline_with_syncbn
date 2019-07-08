import numpy as np
from PIL import Image
import cv2
from os.path import dirname as ospdn
import os.path as osp
import os

def may_make_dir(path):
  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)

def add_border(im, border_width, value):
  """Add color border around an image. The resulting image size is not changed.
  Args:
    im: numpy array with shape [3, im_h, im_w]
    border_width: scalar, measured in pixel
    value: scalar, or numpy array with shape [3]; the color of the border
  Returns:
    im: numpy array with shape [3, im_h, im_w]
  """
  assert (im.ndim == 3) and (im.shape[0] == 3)
  im = np.copy(im)

  if isinstance(value, np.ndarray):
    # reshape to [3, 1, 1]
    value = value.flatten()[:, np.newaxis, np.newaxis]
  im[:, :border_width, :] = value
  im[:, -border_width:, :] = value
  im[:, :, :border_width] = value
  im[:, :, -border_width:] = value

  return im

def make_im_grid(ims, n_rows, n_cols, space, pad_val):
  """Make a grid of images with space in between.
  Args:
    ims: a list of [3, im_h, im_w] images
    n_rows: num of rows
    n_cols: num of columns
    space: the num of pixels between two images
    pad_val: scalar, or numpy array with shape [3]; the color of the space
  Returns:
    ret_im: a numpy array with shape [3, H, W]
  """
  assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
  assert len(ims) <= n_rows * n_cols
  h, w = ims[0].shape[1:]
  H = h * n_rows + space * (n_rows - 1)
  W = w * n_cols + space * (n_cols - 1)
  if isinstance(pad_val, np.ndarray):
    # reshape to [3, 1, 1]
    pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
  ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
  for n, im in enumerate(ims):
    r = n // n_cols
    c = n % n_cols
    h1 = r * (h + space)
    h2 = r * (h + space) + h
    w1 = c * (w + space)
    w2 = c * (w + space) + w
    ret_im[:, h1:h2, w1:w2] = im
  return ret_im


def read_im(im_path):
  # shape [H, W, 3]
  im = np.asarray(Image.open(im_path))
  # Resize to (im_h, im_w) = (256, 256)
  resize_h_w = (384, 292)
  if (im.shape[0], im.shape[1]) != resize_h_w:
    im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
  # shape [3, H, W]
  im = im.transpose(2, 0, 1)
  return im


def save_im(im, save_path):
  """im: shape [3, H, W]"""
  may_make_dir(ospdn(save_path))
  im = im.transpose(1, 2, 0)
  Image.fromarray(im).save(save_path)


def rank_list_to_im(rank_list, same_id, q_im_path,
                         g_im_paths, 
                         n_row=-1, n_col=-1):
  """Save a query and its rank list as an image.
  Args:
    rank_list: a list, the indices of gallery images to show
    same_id: a list, len(same_id) = rank_list, whether each ranked image is
      with same id as query
    q_im_path: query image path
    g_im_paths: ALL gallery image paths
  """
  ims = [read_im(q_im_path)]
  for idx, (ind, sid) in enumerate(zip(rank_list, same_id)):
    im = read_im(g_im_paths[ind])
    # Add green boundary to true positive, red to false positive
    color = np.array([0, 255, 0]) if sid else np.array([255, 0, 0])
    im = add_border(im, 3, color)
    ims.append(im)

  if n_row == -1 and n_col == -1:
    n_row, n_col = 1, len(rank_list) + 1
  assert n_row * n_col == len(rank_list) + 1
  
  im = make_im_grid(ims, n_row, n_col, 8, 255)
  #save_im(im, osp.join(save_path, 'all.png'))

  return Image.fromarray(im.transpose(1,2,0))
