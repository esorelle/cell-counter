import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import ndimage as ndi
from skimage import filters, morphology, util
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


from cell_counter import utils


def light_correction(input_img):
    img_blur = cv2.GaussianBlur(input_img, (15, 15), 1.5)
    img_corr = input_img / img_blur

    # Translate to zero, then normalize to 8-bit range
    img_corr = img_corr - img_corr.min()
    img_corr = np.floor((img_corr / img_corr.max()) * 255.0)
    img_corr = img_corr.astype(np.uint8)

    return img_corr


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


def save_blob_plots(img_name, edges_0, edges_1, blobs_0, blobs_1, blobs_2, blobs_4, blobs_6, blobs_7, refined_blobs):
    image_save_path = img_name + '_process_steps'
    os.chdir(image_save_path)

    plt.imshow(edges_0, cmap='gray')
    plt.title('001_initial_edges')
    plt.tight_layout()
    plt.savefig(img_name + '001_initial_edges' + '.png')
    plt.close()

    plt.imshow(edges_1, cmap='gray')
    plt.title('002_dilated_edges')
    plt.tight_layout()
    plt.savefig(img_name + '002_dilated_edges' + '.png')
    plt.close()

    plt.imshow(blobs_0, cmap='gray')
    plt.title('003_initial_blobs')
    plt.tight_layout()
    plt.savefig(img_name + '003_initial_blobs' + '.png')
    plt.close()

    plt.imshow(blobs_1, cmap='gray')
    plt.title('004_max_area_filtered_blobs')
    plt.tight_layout()
    plt.savefig(img_name + '004_max_area_filtered_blobs' + '.png')
    plt.close()

    plt.imshow(blobs_2, cmap='gray')
    plt.title('005_closed_blobs')
    plt.tight_layout()
    plt.savefig(img_name + '005_closed_blobs' + '.png')
    plt.close()

    plt.imshow(blobs_4, cmap='gray')
    plt.title('006_extent_filtered_opened_blobs')
    plt.tight_layout()
    plt.savefig(img_name + '006_extent_filtered_opened_blobs' + '.png')
    plt.close()

    plt.imshow(blobs_6, cmap='gray')
    plt.title('007_min_area_filtered_blobs')
    plt.tight_layout()
    plt.savefig(img_name + '007_min_area_filtered_blobs' + '.png')
    plt.close()

    plt.imshow(blobs_7, cmap='nipy_spectral')
    plt.title('008_labeled_proto_chambers')
    plt.tight_layout()
    plt.savefig(img_name + '008_labeled_proto_chambers' + '.png')
    plt.close()

    plt.imshow(label(refined_blobs, connectivity=2), cmap='nipy_spectral')
    plt.title('009_labeled_refined_chambers')
    plt.tight_layout()
    plt.savefig(img_name + '009_labeled_refined_chambers' + '.png')
    plt.close()

    os.chdir('..')


def find_blobs(input_img, scaling_thresh, min_area, max_area, min_extent, save_plots=False, plot_dir=None):
    edges_0 = (input_img < scaling_thresh)
    # Changing rectangle structuring element controls connectivity of dense cells.
    # Can lead to false negatives for very full apartments
    # while retaining connection of apartment entry points.
    edges_1 = morphology.dilation(edges_0, morphology.selem.rectangle(1, 4))

    blobs_0 = util.invert(edges_1)
    blobs_1 = utils.remove_large_objects(blobs_0, max_area)
    # increase rect structuring element to increase blob grouping of dense cells as contiguous apartment
    blobs_2 = morphology.closing(blobs_1, morphology.selem.rectangle(5, 1))
    blobs_3 = filter_blobs_by_extent(blobs_2, min_extent)
    blobs_4 = morphology.opening(blobs_3, morphology.selem.disk(3))
    blobs_5 = blobs_4 > 0
    blobs_6 = morphology.remove_small_objects(blobs_5, min_area)
    blobs_7 = label(blobs_6, connectivity=2)

    refined_blobs, blob_boundaries = filter_blobs_near_edge(blobs_7)

    if save_plots:
        save_blob_plots(plot_dir, edges_0, edges_1, blobs_0, blobs_1, blobs_2, blobs_4, blobs_6, blobs_7, refined_blobs)

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
    anchors_x = [
        np.int(np.round((blob_boundaries['east'][i] + blob_boundaries['west'][i]) / 2))
        for i in range(len(blob_boundaries['east']))
    ]
    # analog of centers_y in Scott's code
    anchors_y = [np.int(np.round(blob_boundaries['south'][i])) for i in range(len(blob_boundaries['south']))]

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


def rotate_image(input_img, rows, c_centers):
    if len(rows) == 0:
        raise ValueError("rows list cannot be empty")

    r_degs = []
    for r in rows:
        # linregress doesn't work well for 2 points, so we skip rows with fewer than 3 points
        if len(r) <= 2:
            continue

        gradient, intercept, r_value, p_value, std_err = stats.linregress(c_centers[r[0]:r[-1] + 1])
        if gradient < 1:              # 2020-05-13: override large angle adjustments (observed bug)
            r_deg = np.degrees(np.arctan(gradient))
            r_degs.append(r_deg)

    r_deg_mean = np.mean(r_degs)
    img_8b = np.uint8(input_img / (2**8 + 1))   # new for rotational matrix calc
    n_rows, n_cols = np.shape(img_8b)
    rot_mat = cv2.getRotationMatrix2D((n_cols/2., n_rows/2.), r_deg_mean, 1)
    img_rot = cv2.warpAffine(img_8b, rot_mat, (n_cols, n_rows))
    new_img = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2RGB)

    return new_img, img_rot, r_deg_mean, n_cols, n_rows


def get_key_regions(new_img, img_rot, c_centers, r_deg_mean, n_cols, n_rows):
    row_text_regions = []
    col_text_regions = []
    apartment_mask = np.zeros(np.shape(new_img))  # changed from input_img 2020-06-05
    ordered_apartments = []

    for c_center in c_centers:
        rot_c = utils.rotate(c_center, origin=(n_cols/2., n_rows/2.), degrees=r_deg_mean)
        c_int_tup = tuple(np.round(rot_c).astype(np.int))

        # rect for row number
        row_rect_vert1 = (c_int_tup[0] + 44, c_int_tup[1] - 171)    # -10, -128
        row_rect_vert2 = (c_int_tup[0] + 95, c_int_tup[1] - 139)    # +40, -100
        row_text_regions.append(img_rot[c_int_tup[1] - 171:c_int_tup[1] - 139, c_int_tup[0] + 44:c_int_tup[0] + 95])

        # rect for col number
        col_rect_vert1 = (c_int_tup[0] - 97, c_int_tup[1] - 71)    # -148, -30
        col_rect_vert2 = (c_int_tup[0] - 40, c_int_tup[1] - 39)      # -98, -2
        col_text_regions.append(img_rot[c_int_tup[1] - 71:c_int_tup[1] - 39, c_int_tup[0] - 97:c_int_tup[0] - 40])

        # apt region
        apt_offset_x = c_int_tup[0] - utils.apt_ref_mask.shape[1] + 44    # -10
        apt_offset_y = c_int_tup[1] - utils.apt_ref_mask.shape[0] + 5    # +45
        apt_c = utils.apt_ref_c + [apt_offset_x, apt_offset_y]
        ordered_apartments.append(apt_c)
        cv2.circle(new_img, c_int_tup, 5, (60, 220, 60), -1)
        cv2.rectangle(new_img, row_rect_vert1, row_rect_vert2, (186, 85, 211), 2)
        cv2.rectangle(new_img, col_rect_vert1, col_rect_vert2, (190, 160, 65), 2)
        cv2.drawContours(new_img, [apt_c], 0, (65, 105, 255), 2)
        cv2.drawContours(apartment_mask, [apt_c], 0, (255, 255, 255), -1)

    return new_img, apartment_mask, row_text_regions, col_text_regions, ordered_apartments


def read_digits(row_text_regions, col_text_regions):
    row_numbers = []        # fill with read numbers
    row_num_avg_conf = []   # fill with mean score value from identify digits
    col_numbers = []        # fill with read numbers
    col_num_avg_conf = []   # fill with mean score value from identify digits
    single_digits = []

    for r in row_text_regions:
        row_confs = []
        if np.all(r > 0):                   # test any negative indices due to edge of image
            r_split = np.split(r, 3, axis=1)
            digits = []
            for sub_r in r_split:
                single_digits.append(sub_r)
                digits.append(str(utils.identify_digit(sub_r)[0]))
                row_confs.append(utils.identify_digit(sub_r)[1])
            if len(digits) < 2:
                digits = ['_', '_', '_']    # dummy numbering to prevent empty error
        else:
            digits = ['-Y', '-Y', '-Y']        # dummy numbering to prevent negative index error

        row_numbers.append(''.join(digits))
        row_num_avg_conf.append(np.mean(row_confs))

    for r in col_text_regions:
        col_confs = []
        if np.all(r > 0):
            r_split = np.split(r, 3, axis=1)
            digits = []
            for sub_r in r_split:           # test any negative indices due to edge of image
                digits.append(str(utils.identify_digit(sub_r)[0]))
                col_confs.append(utils.identify_digit(sub_r)[1])
            if len(digits) < 2:
                digits = ['_', '_', '_']     # dummy numbering to prevent empty error
        else:
            digits = ['-X', '-X', '-X']         # dummy numbering to prevent negative index error

        col_numbers.append(''.join(digits))
        col_num_avg_conf.append(np.mean(col_confs))

    return row_numbers, row_num_avg_conf, col_numbers, col_num_avg_conf


def detect_cells_tophat(input_img, tophat_selem):
    struct_elem = morphology.disk(tophat_selem)
    img_scaled = input_img / 255.

    b0 = morphology.white_tophat(img_scaled, struct_elem)
    b1 = b0 * (b0 > 0.035)
    b2 = filters.gaussian(b1, sigma=1.25)
    b3 = b2 > 0.05

    distance = ndi.distance_transform_edt(b3)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((11, 11)), labels=b3)
    markers = ndi.label(local_maxi)[0]

    labels = watershed(-distance, markers, mask=b3)

    return labels


def count_cells_tophat(blob_boundaries, x_coords, y_coords):
    chamber_cell_count_array = np.zeros(len(blob_boundaries['true_north']))
    for p in range(len(x_coords)):
        for c in range(len(blob_boundaries['true_north'])):
            if blob_boundaries['west'][c] < x_coords[p] < blob_boundaries['east'][c]:
                if blob_boundaries['south'][c] - 212 < y_coords[p] < blob_boundaries['south'][c]-15:
                    chamber_cell_count_array[c] += 1

    return chamber_cell_count_array


def detect_and_count_cell_contours(
        img_scaled,
        rect_mask,
        ordered_apartments,
        min_cell_area,
        max_cell_area
):
    gate_img = img_scaled * rect_mask  # changed from apartment_mask to better detect cells on apt edges
    # TODO: NEED TO REVIEW FOR PERFORMANCE ACROSS IMAGES -- HOW BEST TO SCALE FOR UNIFORM COUNTING?
    gate_img = gate_img > 0.8 * np.max(gate_img)
    contour_tree, hierarchy = cv2.findContours(
        gate_img.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = []
    for contour in contour_tree:
        area = cv2.contourArea(contour)
        if min_cell_area < area < max_cell_area:  # was 20-300 range
            filtered_contours.append(contour)

    chamber_cell_count_array_contours = np.zeros(len(ordered_apartments))
    contour_points = []

    for apt in range(len(ordered_apartments)):
        temp_mask = np.zeros(np.shape(gate_img))
        cv2.drawContours(temp_mask, [ordered_apartments[apt]], 0, (255, 255, 255), -1)
        temp_roi = label(temp_mask)

        for blob in regionprops(temp_roi):
            temp_roi_coords = [tuple(j) for j in blob.coords]
            for c in filtered_contours:
                moments = cv2.moments(c)
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                test_point = tuple([cy, cx])
                if test_point in temp_roi_coords:
                    chamber_cell_count_array_contours[apt] += 1
                    contour_points.append(test_point)  # use for scatter plot of counted cells

    chamber_cell_count_array_contours = [int(count) for count in chamber_cell_count_array_contours]

    return chamber_cell_count_array_contours, contour_points
