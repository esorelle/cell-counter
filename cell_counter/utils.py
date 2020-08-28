import cv2
import numpy as np
import glob
import os
from PIL import Image
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image as sk_image
import matplotlib.pyplot as plt


fid_ref_path = 'cell-counter/resources/fiducial_ref_v3.tif'
fid_ref = Image.open(fid_ref_path)
fid_ref = np.asarray(fid_ref)

dig_ref_dir = 'cell-counter/resources/dig_ref_v3'


dig_refs = []
for _ref_digit in range(10):
    dig_tif_files = glob.glob(os.path.join(dig_ref_dir, str(_ref_digit), '*.tif'))
    dig_ref = Image.open(dig_tif_files[0])
    dig_ref = np.asarray(dig_ref)
    dig_refs.append(dig_ref)

kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_cross_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# get contour of a single empty apartment for masking
apt_ref_path = 'cell-counter/resources/apt_ref_v3.tif'
apt_ref_mask = Image.open(apt_ref_path)
apt_ref_mask = np.asarray(apt_ref_mask)
# apt_ref_mask = cv2.erode(apt_ref_mask, kernel_rect_3, iterations=3)
apt_ref_mask = apt_ref_mask.astype(np.bool)
apt_ref_area = apt_ref_mask.sum()


def save_image(img_array, directory, file_name):
    file_path = os.path.join(directory, file_name + '.tif')
    cv2.imwrite(file_path, img_array)


def flip_horizontal(input_img):
    flipped_img = input_img[:, ::-1]
    return flipped_img


# defined rotation function
def rotate(point, origin=(0, 0), degrees=0):
    angle = np.deg2rad(-degrees)
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


# identify digits from templates
def identify_digit(dig_region, digit_candidates=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
    # padding is crucial & needs to be about 1/2  of the template width/height
    # changes from median padding to avoid error...see if this works
    dig_region_pad = np.pad(dig_region, 10, mode='constant', constant_values=(128, 128))
    scores = []
    for i, ref_digit in enumerate(dig_refs):
        if i not in digit_candidates:
            scores.append(0.0)
            continue
        res = cv2.matchTemplate(dig_region_pad, ref_digit, cv2.TM_CCOEFF_NORMED)
        scores.append(res.max())

    return np.argmax(scores), np.max(scores)


def filter_contours_by_size(mask, min_size, max_size=None):
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if max_size is None:
        max_size = mask.shape[0] * mask.shape[1]

    filtered_mask = thresh.copy()

    for i, c in enumerate(contours):
        # test the bounding area first, since it's quicker than rendering each contour
        rect = cv2.boundingRect(c)
        rect_area = rect[2] * rect[3]

        if rect_area > max_size or rect_area <= min_size:
            # Using a new blank mask to calculate actual size of each contour
            # because cv2.contourArea calculates the area using the Green
            # formula, which can give a different result than the number of
            # non-zero pixels.
            tmp_mask = np.zeros(mask.shape)
            cv2.drawContours(tmp_mask, contours, i, 1, cv2.FILLED, hierarchy=hierarchy)

            true_area = (tmp_mask > 0).sum()

            if true_area > max_size or true_area <= min_size:
                # Black-out contour pixels
                # TODO: Should probably collect the "bad" contours and erase them all at once
                cv2.drawContours(filtered_mask, contours, i, 0, cv2.FILLED)

    return filtered_mask


def split_multi_cell(signal_img, multi_cell_mask, max_cell_area, plot=False):
    contour_tree, hierarchy = cv2.findContours(
        multi_cell_mask.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )
    single_cell_contours = []
    sub_multi_contours_idx = []
    for c_idx, sub_c in enumerate(contour_tree):
        h = hierarchy[0][c_idx]
        if h[3] != -1:
            # it's a child contour, ignore it
            continue
        sub_c_area = cv2.contourArea(sub_c)
        if sub_c_area < max_cell_area * .5:
            # too small, some kind of fragment
            continue
        elif sub_c_area > max_cell_area * 1.2:
            # too big for a single cell, try to split it
            sub_multi_contours_idx.append(c_idx)
        else:
            # just right, probably a cell so save it
            single_cell_contours.append(sub_c)
            if plot:
                sc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
                cv2.drawContours(
                    sc_mask,
                    [sub_c],
                    -1,
                    255,
                    cv2.FILLED
                )
                plt.figure(figsize=(16, 16))
                plt.imshow(sc_mask)
                plt.axis('off')
                plt.show()

    for c_idx in sub_multi_contours_idx:
        mc_mask = np.zeros(signal_img.shape, dtype=np.uint8)
        cv2.drawContours(
            mc_mask,
            contour_tree,
            c_idx,
            255,
            cv2.FILLED,
            hierarchy=hierarchy
        )
        # Convert the image into a graph with the value of the gradient on the
        # edges.
        region_graph = sk_image.img_to_graph(
            signal_img,
            mask=mc_mask.astype(np.bool)
        )
        # Take a decreasing function of the gradient: we take it weakly
        # dependent from the gradient the segmentation is close to a voronoi
        region_graph.data = np.exp(-region_graph.data / region_graph.data.std())
        n_clusters = 2
        labels = spectral_clustering(
            region_graph,
            n_clusters=n_clusters,
            eigen_solver='arpack',
            n_init=10
        )
        label_im = np.full(mc_mask.shape, -1.)
        label_im[mc_mask.astype(np.bool)] = labels
        if plot:
            plt.figure(figsize=(16, 16))
            plt.imshow(label_im)
            plt.axis('off')
            plt.show()
        for label in range(n_clusters):
            new_mask = label_im == label
            single_cell_contours.extend(
                split_multi_cell(signal_img, new_mask, max_cell_area, plot=plot)
            )
    return single_cell_contours
