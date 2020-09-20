import os
from datetime import datetime as dt
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings


from cell_counter import utils


def light_correction(input_img, kernel_size=15):
    img_blur = cv2.GaussianBlur(input_img, (kernel_size, kernel_size), 0)
    img_corr = input_img / img_blur

    # Translate to zero, then normalize to 8-bit range
    img_corr = img_corr - img_corr.min()
    img_corr = np.floor((img_corr / img_corr.max()) * 255.0)
    img_corr = img_corr.astype(np.uint8)

    return img_corr


def find_fiducial_locations(input_img, threshold=0.6):
    res = cv2.matchTemplate(input_img, utils.fid_ref, cv2.TM_CCOEFF_NORMED)

    contours, hierarchy = cv2.findContours(
        (res > threshold).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fid_centers = []

    for c in contours:
        c_min_rect = cv2.minAreaRect(c)
        loc = np.array(c_min_rect[0])

        # The resulting array from matchTemplate is smaller than the
        # original image by half the template size, so translate back
        # to original image location
        loc += np.array(utils.fid_ref.shape) / 2.
        loc = np.round(loc).astype(np.uint)

        fid_centers.append(loc)

    return fid_centers


def make_rectangle_mask(input_blobs, blob_boundaries):
    rect_mask = np.zeros(np.shape(input_blobs))
    for i in range(len(blob_boundaries['north'])):
        south = blob_boundaries['south'][i]
        west = blob_boundaries['west'][i]
        east = blob_boundaries['east'][i]
        rect_mask[south - 216:south, west - 5:east + 5] = 1

    return rect_mask


def _find_rows(fiducial_locations):
    centers_y = [loc[1] for loc in fiducial_locations]

    # Rows are separated by roughly 220px, though we expect the image rotation
    # correction required to be < 2.5 degrees. The total image width is ~1400 pixels,
    # so the max separation for 2 fiducials on either end of a row is < 60 pixels.
    row_dist = 60
    rows = []
    assigned_idx = []

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

    return rows


def find_rotation_angle(fiducial_locations):
    row_membership = _find_rows(fiducial_locations)

    if len(row_membership) == 0:
        raise ValueError("failed to locate rows of fiducial markers")

    r_degs = []
    for r in row_membership:
        # stats linear regression function doesn't work well for 2 points, so skip rows with <3 points
        if len(r) < 3:
            continue

        gradient, intercept, r_value, p_value, std_err = stats.linregress(fiducial_locations[r[0]:r[-1] + 1])
        if gradient > 1:              # 2020-05-13: override large angle adjustments (observed bug)
            continue

        r_deg = np.degrees(np.arctan(gradient))
        r_degs.append(r_deg)

    r_deg_mean = np.mean(r_degs)

    return r_deg_mean


def render_fiducials(input_img, fiducial_locations):
    new_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)

    for fid_loc in fiducial_locations:
        x = fid_loc[0]
        y = fid_loc[1]

        #  draw a cross using 2 lines
        cv2.line(new_img, (x - 9, y), (x + 9, y), (60, 220, 60), 4, cv2.LINE_4)
        cv2.line(new_img, (x, y - 9), (x, y + 9), (60, 220, 60), 4, cv2.LINE_4)

    return new_img


def is_edge_fiducial(img_size, x, y):
    img_h, img_w = img_size

    fiducial_right_margin = 90
    fiducial_left_margin = 305
    fiducial_top_margin = 340
    fiducial_bottom_margin = 90

    if x > img_w - fiducial_right_margin or x < fiducial_left_margin:
        return True
    if y > img_h - fiducial_bottom_margin or y < fiducial_top_margin:
        return True

    return False


def identify_digits(sub_region, max_number=999, save_dir=None):
    """
    Identify digits in a image sub-region containing a 3 character address identifier
    :param sub_region: Image sub-array, should be equally divisible by 3
    :param max_number: the maximum 3-digit number possible in the sub-region
    :param save_dir: optional directory to save individual digit image regions
    :return: List of tuples containing the best matching digit and the corresponding matching score for each
        of the 3 digits
    """
    # determine digit candidates for each position
    max_num_str = str(max_number)
    max_num_len = len(max_num_str)
    dig_position_candidates = []
    if max_num_len == 3:
        dig_position_candidates.append(tuple(range(int(max_num_str[0]) + 1)))
        dig_position_candidates.append(tuple(range(10)))
        dig_position_candidates.append(tuple(range(10)))
    elif max_num_len == 2:
        dig_position_candidates.append(tuple(range(1)))
        dig_position_candidates.append(tuple(range(int(max_num_str[0]) + 1)))
        dig_position_candidates.append(tuple(range(10)))
    elif max_num_len == 1:
        dig_position_candidates.append(tuple(range(1)))
        dig_position_candidates.append(tuple(range(1)))
        dig_position_candidates.append(tuple(range(int(max_num_str[0]) + 1)))
    else:
        raise ValueError("max_number must be a positive number less than 999: given %d" % max_number)

    # split region into 3 equal parts, one per digit
    split_regions = np.split(sub_region, 3, axis=1)
    digits = []
    scores = []

    for i, sub_r in enumerate(split_regions):
        if save_dir is not None:
            digit_file_name = 'digit_%s_%d' % (dt.now().strftime('%Y%m%d%H%M%S%f'), i)
            try:
                utils.save_image(sub_r, save_dir, digit_file_name)
            except cv2.error:
                warnings.warn("Failed to save digit sub-region", UserWarning)
            # sleep for a milli-second to avoid duplicate file names
            time.sleep(0.001)

        digit, score = utils.identify_digit(sub_r, digit_candidates=dig_position_candidates[i])
        digits.append(str(digit))
        scores.append(score)

    return digits, scores


def identify_apartments(img_path, flip_horizontal=False, digit_dir=None, fiducial_dir=None, apt_region_dir=None):
    """
    Takes input image (after rotation correction) and fiducial locations to find and return
    the row and column sub-regions containing the row and column addresses

    :param img_path: path to image to extract apartment data
    :param flip_horizontal: whether to flip the given image horizontally, default is False
    :param digit_dir: path for saving individual digit sub-regions, don't save if None (default)
    :param fiducial_dir: path for saving image of fiducial locations, don't save if None (default)
    :param apt_region_dir: path for saving individual apartment sub-regions, don't save if None (default)
    :return: Lists of extracted text regions (row list, col list), in same order as given fiducials
    """
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = input_img.shape
    img_base_name = os.path.splitext(os.path.basename(img_path))[0]

    if flip_horizontal:
        input_img = utils.flip_horizontal(input_img)

    img_light_corr = light_correction(input_img)
    fid_centers = find_fiducial_locations(img_light_corr)
    rot_degrees = find_rotation_angle(fid_centers)

    # rotate the input images, light corrected image & fiducials around the center of the image
    input_img_rot = utils.rotate_image(input_img, rot_degrees)
    img_light_corr_rot = utils.rotate_image(img_light_corr, rot_degrees)
    fid_centers_rotated = utils.rotate_points(fid_centers, (img_h / 2., img_w / 2.), rot_degrees)

    if fiducial_dir is not None:
        fid_img = render_fiducials(input_img_rot, fid_centers_rotated)
        fid_img_name = "_".join(['fiducial_fig', img_base_name])
        utils.save_image(fid_img, fiducial_dir, fid_img_name)

    # row/col addresses are indexed at 0 (000), columns end at 127 and rows end at 46
    # The "origin" is in the bottom right of the image (after flipping).
    max_col = 127
    max_row = 46

    # Both column and row identifiers are 3 characters in length.
    # From the fiducial location, the column address is found to the left,
    # and the row address is found above.
    # The character height is roughly 32 pixels.
    # The region width is roughly 54 pixels, and this number is chosen to
    # be equally divided by the 3 characters (so we can separate the three digits)
    char_height = 52
    char_width = 34
    char_width_3x = char_width * 3

    row_offset_x1 = 17
    row_offset_x2 = row_offset_x1 - char_width_3x
    row_offset_y1 = 209
    row_offset_y2 = row_offset_y1 + char_height

    col_offset_x1 = 305
    col_offset_x2 = col_offset_x1 - char_width_3x
    col_offset_y1 = 3
    col_offset_y2 = col_offset_y1 + char_height

    apt_w = 180
    apt_h = 449
    apt_offset_x1 = 200
    apt_offset_x2 = apt_offset_x1 - apt_w
    apt_offset_y1 = 356
    apt_offset_y2 = apt_offset_y1 - apt_h

    # List of dictionaries for apartment data
    apt_data = []

    for fid_loc in fid_centers_rotated:
        # determine if fiducial is too close to an edge of the image
        x = fid_loc[0]
        y = fid_loc[1]

        on_edge = is_edge_fiducial(input_img.shape, x, y)
        if on_edge:
            continue

        # extract row & column sub-regions from the rotation & light corrected image
        row_region = img_light_corr_rot[y - row_offset_y2:y - row_offset_y1, x - row_offset_x1:x - row_offset_x2]
        col_region = img_light_corr_rot[y - col_offset_y2:y - col_offset_y1, x - col_offset_x1:x - col_offset_x2]

        row_digits, row_scores = identify_digits(row_region, max_number=max_row, save_dir=digit_dir)
        col_digits, col_scores = identify_digits(col_region, max_number=max_col, save_dir=digit_dir)

        # extract apt region from rotated input image
        apt_region = input_img_rot[y - apt_offset_y1:y - apt_offset_y2, x - apt_offset_x1:x - apt_offset_x2]

        apt = {
            'image_name': img_base_name,
            'fid_x': fid_loc[0],
            'fid_y': fid_loc[1],
            'row_region': row_region,
            'row_address': ''.join(row_digits),
            'row_digits': row_digits,
            'row_scores': row_scores,
            'col_address': ''.join(col_digits),
            'col_region': col_region,
            'col_digits': col_digits,
            'col_scores': col_scores,
            'apt_region': apt_region
        }

        if apt_region_dir is not None:
            apt_region_file_name = "_".join(
                [
                    'apt_region',
                    img_base_name,
                    apt['row_address'],
                    apt['col_address']
                ]
            )

            utils.save_image(apt_region, apt_region_dir, apt_region_file_name)

        apt_data.append(apt)

    return apt_data


def extract_cell_data(
        apt_data,
        min_cell_area,
        max_cell_area,
):
    # count the cells in each chamber -- simple percent of apartment method
    for apt in apt_data:
        edge_contours, edge_mask, non_edge_contours, non_edge_mask = find_apartment_blobs(apt['apt_region'])

        edge_blob_area = (edge_mask > 0).sum()
        edge_blob_apt_ratio = edge_blob_area / utils.apt_ref_area
        edge_cell_count_min = round(edge_blob_area / max_cell_area)
        edge_cell_count_max = round(edge_blob_area / min_cell_area)

        non_edge_blob_area = (non_edge_mask > 0).sum()
        non_edge_blob_apt_ratio = non_edge_blob_area / utils.apt_ref_area
        # TODO: I'm reducing the cell sizes by 25% for the non-edge estimate,
        #       as the non-edge contours are typically under-sized. Maybe make
        #       this reduction scale value an input? Or do we dilate the contours
        #       more?
        non_edge_cell_count_min = round(non_edge_blob_area / (0.75 * max_cell_area))
        non_edge_cell_count_max = round(non_edge_blob_area / (0.75 * min_cell_area))

        apt['edge_blob_count'] = len(edge_contours)
        apt['edge_blob_area'] = edge_blob_area
        apt['edge_blob_apt_ratio'] = edge_blob_apt_ratio
        apt['edge_cell_count_min'] = edge_cell_count_min
        apt['edge_cell_count_max'] = edge_cell_count_max
        apt['non_edge_blob_count'] = len(non_edge_contours)
        apt['non_edge_blob_area'] = non_edge_blob_area
        apt['non_edge_blob_apt_ratio'] = non_edge_blob_apt_ratio
        apt['non_edge_cell_count_min'] = non_edge_cell_count_min
        apt['non_edge_cell_count_max'] = non_edge_cell_count_max
        apt['edge_contours'] = edge_contours
        apt['edge_mask'] = edge_mask
        apt['non_edge_contours'] = non_edge_contours
        apt['non_edge_mask'] = non_edge_mask

    return apt_data


def render_apartment(apt_dict):
    # determine number of columns to plot
    col_count = 2  # default has 2 columns: pre-proc image & row/col address regions (w/metadata)
    edge_mask = False
    non_edge_mask = False
    stdev_mask = False
    if 'edge_mask' in apt_dict:
        edge_mask = True
        col_count += 1
    if 'non_edge_mask' in apt_dict:
        non_edge_mask = True
        col_count += 1
    if 'stdev_mask' in apt_dict:
        stdev_mask = True
        col_count += 1

    fig = plt.figure(constrained_layout=True, figsize=(2.1 * col_count, 5))
    gs = fig.add_gridspec(ncols=col_count, nrows=5, height_ratios=[1, 1, 3, 3, 3])

    current_col = 0

    apt_reg_ax = fig.add_subplot(gs[:, current_col])
    apt_reg_ax.set_title('Apt Region', fontsize=11)
    apt_reg_ax.axes.get_xaxis().set_visible(False)
    apt_reg_ax.axes.get_yaxis().set_visible(False)
    apt_reg_ax.imshow(apt_dict['apt_region'], cmap='gray', vmin=0, vmax=255)

    current_col += 1

    if edge_mask:
        apt_reg_ax = fig.add_subplot(gs[:, current_col])
        apt_reg_ax.set_title('Edge Blobs', fontsize=11)
        apt_reg_ax.axes.get_xaxis().set_visible(False)
        apt_reg_ax.axes.get_yaxis().set_visible(False)
        apt_reg_ax.imshow(apt_dict['edge_mask'], cmap='gray', vmin=0, vmax=255)

        current_col += 1

    if non_edge_mask:
        apt_reg_ax = fig.add_subplot(gs[:, current_col])
        apt_reg_ax.set_title('Non-edge Blobs', fontsize=11)
        apt_reg_ax.axes.get_xaxis().set_visible(False)
        apt_reg_ax.axes.get_yaxis().set_visible(False)
        apt_reg_ax.imshow(apt_dict['non_edge_mask'], cmap='gray', vmin=0, vmax=255)

        current_col += 1

    if stdev_mask:
        apt_reg_ax = fig.add_subplot(gs[:, current_col])
        apt_reg_ax.set_title('Std Dev Blobs', fontsize=11)
        apt_reg_ax.axes.get_xaxis().set_visible(False)
        apt_reg_ax.axes.get_yaxis().set_visible(False)
        apt_reg_ax.imshow(apt_dict['stdev_mask'], cmap='gray', vmin=0, vmax=1)

        current_col += 1

    row_reg_ax = fig.add_subplot(gs[0, current_col])
    row_reg_ax.set_title('Row: %s' % ''.join(apt_dict['row_digits']), fontsize=11)
    row_reg_ax.axes.get_xaxis().set_visible(False)
    row_reg_ax.axes.get_yaxis().set_visible(False)
    row_reg_ax.imshow(apt_dict['row_region'], cmap='gray', vmin=0, vmax=255)

    col_reg_ax = fig.add_subplot(gs[1, current_col])
    col_reg_ax.set_title('Col: %s' % ''.join(apt_dict['col_digits']), fontsize=11)
    col_reg_ax.axes.get_xaxis().set_visible(False)
    col_reg_ax.axes.get_yaxis().set_visible(False)
    col_reg_ax.imshow(apt_dict['col_region'], cmap='gray', vmin=0, vmax=255)

    if 'edge_cell_count_min' in apt_dict:
        edge_text_str = '\n'.join(
            (
                r'Edge stats:',
                r'# blobs: %d' % apt_dict['edge_blob_count'],
                r'count min: %d' % apt_dict['edge_cell_count_min'],
                r'count max: %d' % apt_dict['edge_cell_count_max'],
                r'area: %d' % apt_dict['edge_blob_area'],
                r'percent: %.1f%%' % (apt_dict['edge_blob_apt_ratio'] * 100)
            )
        )
        non_edge_text_str = '\n'.join(
            (
                r'Non-edge stats:',
                r'# blobs: %d' % apt_dict['non_edge_blob_count'],
                r'count min: %d' % apt_dict['non_edge_cell_count_min'],
                r'count max: %d' % apt_dict['non_edge_cell_count_max'],
                r'area: %d' % apt_dict['non_edge_blob_area'],
                r'percent: %.1f%%' % (apt_dict['non_edge_blob_apt_ratio'] * 100)
            )
        )

        # place a text box w/ stats in lower right subplot
        edge_text_ax = fig.add_subplot(gs[2, current_col])
        edge_text_ax.axis('off')
        edge_text_ax.text(0, 0.05, edge_text_str, fontsize=10, verticalalignment='bottom')

        non_edge_text_ax = fig.add_subplot(gs[3, current_col])
        non_edge_text_ax.axis('off')
        non_edge_text_ax.text(0, 0.95, non_edge_text_str, fontsize=10, verticalalignment='top')

    return fig


def find_apartment_blobs(apt_img):
    region_shape = apt_img.shape

    blur_median = cv2.medianBlur(apt_img, ksize=7)  # 7
    blur_bilateral = cv2.bilateralFilter(apt_img, d=5, sigmaColor=5, sigmaSpace=31)  # 7

    # Next, we perform a pseudo DoG, though it's not really a diff of
    # Gaussian's, but still a diff of blurs. The bilateral blur retains
    # edge features while blurring the background, where the median blur
    # is less selective.By subtracting the heavy median blur. The goal
    # is to remove the noise while retaining the edge features, and the
    # blur images are cast to  signed 16 bit to allow for negative values
    # from the result of the subtraction.
    dog_img = blur_bilateral.astype(np.int16) - blur_median.astype(np.int16)

    # Next, sample the background from a circle outside the apartment,
    # taking the min value of the background sample to use as a threshold
    bkg_mask = np.zeros(region_shape)
    bkg_mask = cv2.circle(bkg_mask, (11, 400), 9, 1, -1)
    bkg_mask = bkg_mask.astype(np.bool)

    bkg_min = dog_img[bkg_mask].min()

    # Feature edges in the apartment image are darker than background,
    # so set all dark pixels to 255
    dog_img_tmp = np.zeros(region_shape)
    dog_img_tmp[dog_img < bkg_min] = 255
    dog_img = dog_img_tmp.astype(np.uint8)

    # set all pixels outside the apt reference mask to zero
    dog_img[~utils.apt_ref_mask] = 0

    # May be noise leftover, but mostly at the apartment edges
    # and the noise regions are relatively small. Eliminate them
    # by filtering on size, rather than using morphological opening,
    # as it is non-destructive to the filtered contours.
    dog_img = utils.filter_contours_by_size(dog_img, min_size=8)

    # At this point, we should have isolated the cell boundaries,
    # though their borders are frequently broken, with the interior
    # of cells not being filled. We will attempt to connect broken
    # borders by dilating with a 3x3 rectangle kernel, followed by
    # taking their convex hull to help fill gaps.
    contour_mask = cv2.dilate(dog_img, utils.kernel_rect_3, iterations=1)

    connected_contours, hierarchy = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    edge_based_contours = []

    for c in connected_contours:
        convex_c = cv2.convexHull(c)
        edge_based_contours.append(convex_c)

    # The result should have the majority of gaps and cell interiors
    # filled, but the result is slightly over-segmented due to the
    # previous dilation. We erode by the same kernel to return to
    # a closer match to the original cell boundaries
    final_edge_based_mask = np.zeros(region_shape, dtype=np.uint8)
    cv2.drawContours(final_edge_based_mask, edge_based_contours, -1, 255, cv2.FILLED)

    final_edge_based_mask = cv2.erode(final_edge_based_mask, utils.kernel_rect_3, 1)
    final_edge_based_mask[~utils.apt_ref_mask] = 0

    # START 2ND BLOB DETECTION METHOD - NON-EDGE BASED
    bkg_bilat_min = blur_bilateral[bkg_mask].min()

    # Feature edges in the apartment image are darker than background,
    # so set all dark pixels to 255
    bilat_mask_tmp = np.zeros(region_shape, dtype=np.uint8)
    bilat_mask_tmp[blur_bilateral < bkg_bilat_min] = 255

    bilat_mask_tmp = utils.filter_contours_by_size(bilat_mask_tmp, min_size=9)

    bilat_mask_tmp = cv2.dilate(bilat_mask_tmp, utils.kernel_cross_3, iterations=3)
    bilat_mask_tmp = cv2.erode(bilat_mask_tmp, utils.kernel_cross_3, iterations=3)
    bilat_mask_tmp = ~bilat_mask_tmp
    bilat_mask_tmp[~utils.apt_ref_mask] = 0

    contours, hierarchy = cv2.findContours(
        bilat_mask_tmp,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_non_edge_contours = []

    for c in contours:
        c_mask = np.zeros(region_shape)
        cv2.drawContours(c_mask, [c], -1, 1, cv2.FILLED)

        c_area = (c_mask > 0).sum()
        union_area = np.logical_and(final_edge_based_mask, c_mask).sum()

        union_ratio = union_area / float(c_area)

        if union_ratio >= 0.5:
            final_non_edge_contours.append(c)

    final_non_edge_mask = np.zeros(region_shape)
    cv2.drawContours(final_non_edge_mask, final_non_edge_contours, -1, 255, cv2.FILLED)

    return (
        edge_based_contours, final_edge_based_mask,
        final_non_edge_contours, final_non_edge_mask
    )
