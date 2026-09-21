# -*- coding: utf-8 -*-
"""
PART 1 - collect the crop features, one npz per image.

WHY IT TAKES `dets` AND NOT THE RAW MODEL OUTPUT
    The txt files are written by decode_detections(extract_dets_from_outputs(...)).
    If we decoded the boxes ourselves we could easily end up with a slightly
    different geometry and a different row order. So this file reads the SAME
    `dets` array and uses the SAME calibration objects, and repeats exactly the
    arithmetic decode_detections does - minus the score threshold, because we
    want all 50 rows, not only the surviving ones.

    Result: row i of the npz is line i of the txt file, always.

Layout of one saved file (50 rows, same order as the txt):

    crop        (50, 256, 7, 7)  float16   raw feature map, cropped per box
    prob        (50,   1, 7, 7)  float16   RSH mask, same crop
    extra       (50, 31)         float32   scalars, see below
    query_idx   (50,)            int16     0..49
    score       (50,)            float32   the class score, to check alignment

    enhanced = crop * prob  -> rebuild later, costs nothing to store

EXTRA COLUMNS
     0  1    window width, height in image pixels, before the resize
     2  3    window centre u, v in image pixels
     4       depth
     5       sigma factor exp(-sigma_raw), as extract_dets_from_outputs gives it
     6       class score
     7 - 10  2D box minus projected 3D box, per edge, divided by the box size
    11 - 26  the 8 corners projected, expressed relative to the window
    27 28 29 h, w, l minus the kitti mean car size
    30       ry
"""

import os
import numpy as np
import torch
from torchvision.ops import roi_align


# =====================================================================
# SETTINGS
# =====================================================================
CROP_SIZE = 7          # output grid
WINDOW_SCALE = 1.5     # window = this many times the projected 3D box
FEATURE_STRIDE = 8     # srcs[0] is 1/8 of the image
SAMPLING_RATIO = 2

MEAN_H, MEAN_W, MEAN_L = 1.53, 1.63, 3.88   # kitti mean car size, reference only

N_EXTRA = 31

# column layout of `dets` produced by extract_dets_from_outputs
# [labels, scores, xs2d, ys2d, size_2d(2), depth, heading(24), size_3d(3),
#  xs3d, ys3d, sigma]
C_LABEL, C_SCORE = 0, 1
C_XS2D, C_YS2D = 2, 3
C_SIZE2D = slice(4, 6)
C_DEPTH = 6
C_HEADING = slice(7, 31)
C_SIZE3D = slice(31, 34)
C_XS3D, C_YS3D = 34, 35
C_SIGMA = 36


# ---------------------------------------------------------------------
def box3d_corners_np(h, w, l, x, y, z, ry):
    """8 corners of one kitti 3D box. y is the box CENTRE height here,
    matching what decode_detections produces after locations[1] += h/2.
    Returns (3, 8)."""
    xc = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    yc = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2])
    zc = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(ry), np.sin(ry)
    xr = c * xc + s * zc
    zr = -s * xc + c * zc
    return np.stack([xr + x, yc + y, zr + z], axis=0)


def project_np(corners, P2):
    """corners (3, 8), P2 (3, 4) -> u (8,), v (8,)."""
    hom = np.vstack([corners, np.ones((1, corners.shape[1]))])
    p = P2 @ hom
    d = np.clip(p[2], 1e-6, None)
    return p[0] / d, p[1] / d


# =====================================================================
class CropFeatureSaver(object):

    def __init__(self, out_dir, crop_size=CROP_SIZE,
                 window_scale=WINDOW_SCALE, stride=FEATURE_STRIDE):
        self.out_dir = out_dir
        self.crop_size = crop_size
        self.window_scale = window_scale
        self.stride = stride
        os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def save_batch(self, outputs, dets, info, calibs, cls_mean_size):
        """
        outputs       : the model dict, must carry feat_map and region_prob
        dets          : numpy array from extract_dets_from_outputs, already
                        moved to cpu, shape (B, 50, 37)
        info          : the numpy dict, needs img_id and img_size
        calibs        : the LIST of Calibration objects, the same list
                        decode_detections receives
        cls_mean_size : dataloader.dataset.cls_mean_size
        """
        feat = outputs['feat_map']           # (B, 256, Hf, Wf)
        prob = outputs['region_prob']        # (B,   1, Hf, Wf)
        device = feat.device
        B, Q = dets.shape[0], dets.shape[1]

        for i in range(B):
            img_id = int(info['img_id'][i])
            W_img = float(info['img_size'][i][0])
            H_img = float(info['img_size'][i][1])
            calib = calibs[i]
            P2 = np.asarray(calib.P2, dtype=np.float64)

            win = np.zeros((Q, 4), dtype=np.float32)
            extra = np.zeros((Q, N_EXTRA), dtype=np.float32)
            score = np.zeros(Q, dtype=np.float32)

            for j in range(Q):
                d = dets[i, j]
                cls_id = int(d[C_LABEL])
                score[j] = float(d[C_SCORE])

                # --- 2D box, exactly as decode_detections ---------------
                x = d[C_XS2D] * W_img
                y = d[C_YS2D] * H_img
                w2 = d[C_SIZE2D][0] * W_img
                h2 = d[C_SIZE2D][1] * H_img
                box2d = np.array([x - w2 / 2, y - h2 / 2,
                                  x + w2 / 2, y + h2 / 2])

                # --- 3D size: the head predicts a RESIDUAL --------------
                dims = d[C_SIZE3D].copy() + cls_mean_size[cls_id]
                h3, w3, l3 = float(dims[0]), float(dims[1]), float(dims[2])

                # --- 3D location, exactly as decode_detections ----------
                depth = float(d[C_DEPTH])
                x3d = d[C_XS3D] * W_img
                y3d = d[C_YS3D] * H_img
                loc = calib.img_to_rect(x3d, y3d, depth).reshape(-1)
                loc[1] += h3 / 2.0        # now loc is the box CENTRE

                # --- heading, exactly as decode_detections --------------
                from lib.helpers.decode_helper import get_heading_angle
                alpha = get_heading_angle(d[C_HEADING])
                ry = calib.alpha2ry(alpha, x)

                # --- project the 3D box ---------------------------------
                corners = box3d_corners_np(h3, w3, l3,
                                           loc[0], loc[1], loc[2], float(ry))
                u, v = project_np(corners, P2)
                p3d = np.array([u.min(), v.min(), u.max(), v.max()])

                # --- window --------------------------------------------
                cx = (p3d[0] + p3d[2]) / 2
                cy = (p3d[1] + p3d[3]) / 2
                bw = max(p3d[2] - p3d[0], 1.0)
                bh = max(p3d[3] - p3d[1], 1.0)
                ww = bw * self.window_scale
                wh = bh * self.window_scale
                win[j] = [cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2]

                # --- extras --------------------------------------------
                extra[j, 0] = ww
                extra[j, 1] = wh
                extra[j, 2] = cx
                extra[j, 3] = cy
                extra[j, 4] = depth
                extra[j, 5] = float(d[C_SIGMA])
                extra[j, 6] = float(d[C_SCORE])

                extra[j, 7] = (box2d[0] - p3d[0]) / bw
                extra[j, 8] = (box2d[1] - p3d[1]) / bh
                extra[j, 9] = (box2d[2] - p3d[2]) / bw
                extra[j, 10] = (box2d[3] - p3d[3]) / bh

                extra[j, 11:19] = (u - win[j, 0]) / ww
                extra[j, 19:27] = (v - win[j, 1]) / wh

                extra[j, 27] = h3 - MEAN_H
                extra[j, 28] = w3 - MEAN_W
                extra[j, 29] = l3 - MEAN_L
                extra[j, 30] = ry

            # --- ROI Align, all 50 windows at once ----------------------
            # roi_align interpolates, it does NOT round to whole cells.
            # That is the entire point: two boxes a fraction of a cell apart
            # come out as different crops.
            rois = torch.cat([torch.zeros(Q, 1),
                              torch.from_numpy(win)], dim=1).to(device).float()

            crop = roi_align(feat[i:i + 1], rois,
                             output_size=(self.crop_size, self.crop_size),
                             spatial_scale=1.0 / self.stride,
                             sampling_ratio=SAMPLING_RATIO, aligned=True)
            pcrop = roi_align(prob[i:i + 1], rois,
                              output_size=(self.crop_size, self.crop_size),
                              spatial_scale=1.0 / self.stride,
                              sampling_ratio=SAMPLING_RATIO, aligned=True)

            np.savez_compressed(
                os.path.join(self.out_dir, '{:06d}.npz'.format(img_id)),
                crop=crop.detach().cpu().numpy().astype(np.float16),
                prob=pcrop.detach().cpu().numpy().astype(np.float16),
                extra=extra,
                query_idx=np.arange(Q, dtype=np.int16),
                score=score,
            )
