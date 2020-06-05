import numpy as np
from skimage import filters
from skimage.measure import label, regionprops


def light_correction(input_img, gauss_blur_sigma, window_thresh):
    img_blur = filters.gaussian(input_img, gauss_blur_sigma)
    back_map = filters.threshold_local(img_blur, window_thresh, offset=0)
    img_scaled = np.divide(img_blur, back_map) * 255

    return img_scaled


def filter_blobs_by_extent(input_blobs, min_blob_extent):
    refined_blobs = np.zeros(np.shape(input_blobs))

    for blob in regionprops(label(input_blobs)):
        if blob.extent > min_blob_extent:
            tmp_north, tmp_west, tmp_south, tmp_east = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]
            # new condition based on bounding box dimensions -- remove if needed
            if np.logical_and(70 < tmp_east - tmp_west < 100, 180 < tmp_south - tmp_north < 240):
                hull = blob.convex_image
                refined_blobs[tmp_north:tmp_south, tmp_west:tmp_east] = hull

    return refined_blobs


def filter_blobs_near_edge(input_blobs):
    refined_blobs = np.zeros(np.shape(input_blobs))

    north, west, south, east = [], [], [], []
    true_north = []

    for blob in regionprops(input_blobs):
        # edge border of image -- was 50 pixels
        if abs(blob.centroid[0] - np.shape(input_blobs)[0]) > 70 and blob.centroid[0] > 70:
            if abs(blob.centroid[1] - np.shape(input_blobs)[1]) > 70 and blob.centroid[1] > 70:
                tmp_north, tmp_west, tmp_south, tmp_east = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]
                north.append(tmp_north)
                west.append(tmp_west - 3)
                south.append(tmp_south)
                east.append(tmp_east + 3)
                true_north.append(tmp_south - 212)
                hull = blob.convex_image
                refined_blobs[tmp_north:tmp_south, tmp_west:tmp_east] = hull

    blob_boundaries = {
        'north': north,
        'east': east,
        'west': west,
        'south': south,
        'true_north': true_north
    }

    return refined_blobs, blob_boundaries


def make_rectangle_mask(input_blobs, blob_boundaries):
    rect_mask = np.zeros(np.shape(input_blobs))
    for i in range(len(blob_boundaries['north'])):
        south = blob_boundaries['south'][i]
        west = blob_boundaries['west'][i]
        east = blob_boundaries['east'][i]
        rect_mask[south - 216:south, west - 5:east + 5] = 1

    return rect_mask


def find_rows(blob_boundaries):
    anchors_x = [np.int(np.round((blob_boundaries['east'][i] + blob_boundaries['west'][i]) / 2)) for i in range(len(blob_boundaries['east']))]
    anchors_y = [np.int(np.round(blob_boundaries['south'][i])) for i in range(len(blob_boundaries['south']))]  # analog of centers_y in Scott's code

    c_centers = [(anchors_x[i], anchors_y[i]) for i in range(len(anchors_y))]
    assigned_idx = []
    centers_y = np.array(anchors_y)

    row_dist = 110  # rows are separated by roughly 220px
    rows = []
    for i, cy in enumerate(centers_y):
        if i in assigned_idx:
            continue
        row_min = cy - row_dist
        row_max = cy + row_dist
        in_row = np.logical_and(centers_y > row_min, centers_y < row_max)
        row_membership = np.where(in_row)
        row_members = list(row_membership[0])
        rows.append(row_members)
        assigned_idx.extend(row_members)

    return rows, c_centers
