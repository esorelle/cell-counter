import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from datetime import datetime as dt
from PIL import Image
from scipy import stats
from scipy import ndimage as ndi
from skimage import filters, morphology, util
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from warnings import warn

workflow_list = [
    'denoise',
    'norm',
    'edges',
    'dilate',
    'invert',
    'maxout',
    'close',
    'extentout',
    'open',
    'minout',
    'label',
    'borderout',
    'de-rotate',
    'apartment_mask',
    'read_digits'
]
apartment_workflow = ' -> '.join(workflow_list)
cell_workflow = 'denoise -> norm -> tophat -> blur -> distance -> label -> centroids'

# get contour of a single empty apartment for masking
apt_ref_path = '../resources/apt_ref_2.tif'  # switch back to apt_ref.tif if needed
apt_ref_mask = Image.open(apt_ref_path)
apt_ref_mask = np.asarray(apt_ref_mask)
apt_ref_c, _ = cv2.findContours(apt_ref_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
apt_ref_c = apt_ref_c[0]

dig_refs = []
for i in range(10):
    dig_ref = Image.open('../resources/dig_ref_%d.tif' % i)
    dig_ref = np.asarray(dig_ref)
    dig_refs.append(dig_ref)


# defined image flip
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
def identify_digit(dig_region):
    # padding is crucial & needs to be about 1/2  of the template width/height
    # changes from median padding to avoid error...see if this works
    dig_region_pad = np.pad(dig_region, 10, mode='constant', constant_values=(128, 128))
    scores = []
    for i, dig_ref in enumerate(dig_refs):
        res = cv2.matchTemplate(dig_region_pad, dig_ref, cv2.TM_CCOEFF_NORMED)
        scores.append(res.max())
    if np.max(scores) > 0.6:
        return np.argmax(scores), np.max(scores)
    else:
        return '_', np.max(scores)


# cribbed from skimage to create remove_large_objects function
def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


def remove_large_objects(ar, max_size=64, connectivity=1, in_place=False):
    # Raising type error if not int or bool
    _check_dtype_supported(ar)
    if in_place:
        out = ar
    else:
        out = ar.copy()
    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")
    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0
    return out


# new cell count method that uses chamber blobs to do direct gating
def get_chamber_cell_counts_bf(
        input_img,
        img_name,
        gauss_blur_sigma,
        window_thresh,
        scaling_thresh,
        min_blob_area,
        max_blob_area,
        min_blob_extent,
        tophat_selem,
        min_cell_area,
        max_cell_area,
        save_process_pics
):
    # find the chambers
    img_blur = filters.gaussian(input_img, gauss_blur_sigma)
    back_map = filters.threshold_local(img_blur, window_thresh, offset=0)
    img_scaled = np.divide(img_blur, back_map) * 255
    img_8b = np.uint8(input_img / (2**8 + 1))   # new for rotational matrix calc
    edges_0 = (img_scaled < scaling_thresh)
    # 2,3 --> changed to 1,3 to reduce connectivity of dense cells
    # leading to false negative for very full apartments
    # while retaining connection of apartment entry points.
    edges_1 = morphology.dilation(edges_0, morphology.selem.rectangle(1, 4))
    blobs_0 = util.invert(edges_1)
    blobs_1 = remove_large_objects(blobs_0, max_blob_area)
    # 3,1 --> changed to 4,1 to increase blob grouping of dense cells as contiguous apartment
    blobs_2 = morphology.closing(blobs_1, morphology.selem.rectangle(5, 1))
    blobs_3 = np.zeros(np.shape(input_img))
    north, west, south, east = [], [], [], []
    true_north = []

    for blob in regionprops(label(blobs_2)):
        if blob.extent > min_blob_extent:
            tmp_north, tmp_west, tmp_south, tmp_east = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]

            # new condition based on bounding box dimensions -- remove if needed
            if np.logical_and(70 < tmp_east - tmp_west < 100, 180 < tmp_south - tmp_north < 240):
                north.append(tmp_north)
                west.append(tmp_west - 3)
                south.append(tmp_south)
                east.append(tmp_east + 3)
                true_north.append(tmp_south - 212)
                hull = blob.convex_image
                blobs_3[tmp_north:tmp_south, tmp_west:tmp_east] = hull

    blobs_4 = morphology.opening(blobs_3, morphology.selem.disk(3))
    blobs_5 = blobs_4 > 0
    blobs_6 = morphology.remove_small_objects(blobs_5, min_blob_area)
    blobs_7 = label(blobs_6, connectivity=2)
    refined_blobs = np.zeros(np.shape(input_img))
    north, west, south, east = [], [], [], []
    true_north = []
    for blob in regionprops(blobs_7):
        # edge border of image -- was 50 pixels
        if abs(blob.centroid[0] - np.shape(input_img)[0]) > 70 and blob.centroid[0] > 70:
            if abs(blob.centroid[1] - np.shape(input_img)[1]) > 70 and blob.centroid[1] > 70:
                tmp_north, tmp_west, tmp_south, tmp_east = blob.bbox[0], blob.bbox[1], blob.bbox[2], blob.bbox[3]
                north.append(tmp_north)
                west.append(tmp_west - 3)
                south.append(tmp_south)
                east.append(tmp_east + 3)
                true_north.append(tmp_south - 212)
                hull = blob.convex_image
                refined_blobs[tmp_north:tmp_south, tmp_west:tmp_east] = hull

    # make rectangles (or insert chamber polygon ndarray with reference to anchor point)
    rect_mask = np.zeros(np.shape(input_img))
    for i in range(len(true_north)):
        rect_mask[south[i]-212:south[i]-15, west[i]:east[i]] = 1

    # de-rotate the image by finding chamber row angles and correcting
    anchors_x = [np.int(np.round((east[i] + west[i]) / 2)) for i in range(len(east))]
    anchors_y = [np.int(np.round(south[i])) for i in range(len(south))]     # analog of centers_y in Scott's code
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

    r_degs = []
    for r in rows:
        gradient, intercept, r_value, p_value, std_err = stats.linregress(c_centers[r[0]:r[-1] + 1])
        if gradient < 1:              # 2020-05-13: override large angle adjustments (observed bug)
            r_deg = np.degrees(np.arctan(gradient))
            r_degs.append(r_deg)

    r_deg_mean = np.mean(r_degs)
    n_rows, n_cols = np.shape(img_8b)
    rot_mat = cv2.getRotationMatrix2D((n_cols/2., n_rows/2.), r_deg_mean, 1)
    img_rot = cv2.warpAffine(img_8b, rot_mat, (n_cols, n_rows))
    new_img = cv2.cvtColor(img_rot, cv2.COLOR_GRAY2RGB)
    row_text_regions = []
    row_numbers = []        # fill with read numbers
    row_num_avg_conf = []   # fill with mean score value from identify digits
    col_text_regions = []
    col_numbers = []        # fill with read numbers
    col_num_avg_conf = []   # fill with mean score value from identify digits
    apartment_mask = np.zeros(np.shape(input_img))
    for c_center in c_centers:
        rot_c = rotate(c_center, origin=(n_cols/2., n_rows/2.), degrees=r_deg_mean)
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
        apt_offset_x = c_int_tup[0] - apt_ref_mask.shape[1] + 44    # -10
        apt_offset_y = c_int_tup[1] - apt_ref_mask.shape[0] + 5    # +45
        apt_c = apt_ref_c + [apt_offset_x, apt_offset_y]
        cv2.circle(new_img, c_int_tup, 5, (60, 220, 60), -1)
        cv2.rectangle(new_img, row_rect_vert1, row_rect_vert2, (186, 85, 211), 2)
        cv2.rectangle(new_img, col_rect_vert1, col_rect_vert2, (190, 160, 65), 2)
        cv2.drawContours(new_img, [apt_c], 0, (65, 105, 255), 2)
        cv2.drawContours(apartment_mask, [apt_c], 0, (255, 255, 255), -1)

    single_digits = []
    for r in row_text_regions:
        if np.all(r > 0):                   # test any negative indices due to edge of image
            r_split = np.split(r, 3, axis=1)
            digits = []
            row_confs = []
            for sub_r in r_split:
                single_digits.append(sub_r)
                digits.append(str(identify_digit(sub_r)[0]))
                row_confs.append(identify_digit(sub_r)[1])
            if len(digits) < 2:
                digits = ['_', '_', '_']    # dummy numbering to prevent empty error
        else:
            digits = ['-Y', '-Y', '-Y']        # dummy numbering to prevent negative index error
        row_numbers.append(''.join(digits))
        row_num_avg_conf.append(np.mean(row_confs))
    for r in col_text_regions:
        if np.all(r > 0):
            r_split = np.split(r, 3, axis=1)
            digits = []
            col_confs = []
            for sub_r in r_split:           # test any negative indices due to edge of image
                digits.append(str(identify_digit(sub_r)[0]))
                col_confs.append(identify_digit(sub_r)[1])
            if len(digits) < 2:
                digits = ['_', '_', '_']     # dummy numbering to prevent empty error
        else:
            digits = ['-X', '-X', '-X']         # dummy numbering to prevent negative index error
        col_numbers.append(''.join(digits))
        col_num_avg_conf.append(np.mean(col_confs))

    image_save_path = img_name + '_process_steps'
    os.mkdir(image_save_path)
    os.chdir(image_save_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(new_img, cmap='gray', vmin=0, vmax=255)
    plt.title('apartment_and_address_overlay')
    plt.tight_layout()
    plt.savefig(img_name + 'apartment_and_address_overlay' + '.png')
    plt.close()
    os.chdir('..')

    # count the cells in each chamber
    img_blur = filters.gaussian(input_img, 1)
    back_map = filters.threshold_local(img_blur, window_thresh, offset=0)
    img_scaled = np.divide(img_blur, back_map) * 255
    struct_elem = morphology.disk(tophat_selem)
    img_scaled = img_scaled / 255
    b0 = morphology.white_tophat(img_scaled, struct_elem)
    b1 = b0 * (b0 > 0.035)
    b2 = filters.gaussian(b1, sigma=1.25)
    b3 = b2 > 0.05
    distance = ndi.distance_transform_edt(b3)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((11, 11)), labels=b3)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=b3)
    # optional save of process step images
    if save_process_pics == 1:
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
    # show detected centroids overlay on input image
    fig, ax = plt.subplots(figsize=(10, 6))
    # subbed new_img for input_img to show detected apartment and text regions
    ax.imshow(new_img, cmap='gray')
    x_coords, y_coords = [], []
    for region in regionprops(labels):
        if min_cell_area < region.area < max_cell_area:
            cent_x, cent_y = region.centroid
            if apartment_mask[int(np.round(cent_x)), int(np.round(cent_y))] > 0:
                y_coords.append(int(np.round(cent_x)))
                x_coords.append(int(np.round(cent_y)))
                ax.scatter(cent_y, cent_x, s=4, c='lightcoral', marker='x')
    ax.set_axis_off()
    plt.title('#_of_detected_cells_in_image = ' + str(len(x_coords)))
    plt.tight_layout()
    plt.savefig(img_name + '_detected_cells' + '.png')
    plt.close()
    # tabulate the counted cells by chamber for output
    prefix = img_name[img_name.find('ST_'):img_name.find('ST_') + 14] + '_CHAMBER_'
    address_counts = {}
    chamber_cell_count_array = np.zeros(len(north))
    for p in range(len(x_coords)):
        for c in range(len(true_north)):
            if west[c] < x_coords[p] < east[c]:
                if south[c] - 212 < y_coords[p] < south[c]-15:
                    chamber_cell_count_array[c] += 1
    for chamber in range(len(chamber_cell_count_array)):
        if chamber < 9:
            address_counts[prefix + '00' + str(chamber + 1)] = [
                str(row_numbers[chamber]),
                str(col_numbers[chamber]),
                row_num_avg_conf[chamber],
                col_num_avg_conf[chamber],
                int(chamber_cell_count_array[chamber])
            ]
        else:
            address_counts[prefix + '0' + str(chamber + 1)] = [
                str(row_numbers[chamber]),
                str(col_numbers[chamber]),
                row_num_avg_conf[chamber],
                col_num_avg_conf[chamber],
                int(chamber_cell_count_array[chamber])
            ]
    return address_counts


# directory wrapper
def process_directory_relative_id(
        flip,
        gauss_blur_sigma,
        window_thresh,
        scaling_thresh,
        min_blob_area,
        max_blob_area,
        min_blob_extent,
        tophat_selem,
        min_cell_area,
        max_cell_area,
        save_process_pics,
        count_hist,
        targetdirectory
):
    version = '_v20200513'
    cwd = targetdirectory
    os.chdir(cwd)
    save_path = 'analysis_' + dt.now().strftime('%Y_%m_%d_%H_%M')
    os.mkdir(save_path)
    images = glob.glob('./*.tif')
    images = [r[2:] for r in images]
    num_images = len(images)
    cell_counts_df = pd.DataFrame(
        columns=[
            'Folder_Name',
            'Image_Name',
            'Chamber_ID',
            'Apt_Row',
            'Apt_Col',
            'Row_Dig_ID_Conf',
            'Col_Dig_ID_Conf',
            'Detected_Cells'
        ]
    )
    num_chambers_detected = 0
    apartments_per_image = []
    for i in range(0, num_images):
        img_name = images[i]
        print(i, img_name)

        raw_img = plt.imread(img_name)
        if flip:
            raw_img = flip_horizontal(raw_img)
        os.chdir(save_path)
        address_counts = get_chamber_cell_counts_bf(
            raw_img,
            img_name,
            gauss_blur_sigma,
            window_thresh,
            scaling_thresh,
            min_blob_area,
            max_blob_area,
            min_blob_extent,
            tophat_selem,
            min_cell_area,
            max_cell_area,
            save_process_pics
        )
        os.chdir('..')
        print(img_name + ': ' + str(len(address_counts)) + ' chambers counted')
        num_chambers_detected += len(address_counts)
        apartments_per_image.append(len(address_counts))
        for chamber_key in address_counts:
            cell_counts_df = cell_counts_df.append(
                {
                    'Folder_Name': cwd,
                    'Image_Name': img_name,
                    'Chamber_ID': chamber_key,
                    'Apt_Row': address_counts[chamber_key][0],
                    'Apt_Col': address_counts[chamber_key][1],
                    'Row_Dig_ID_Conf': address_counts[chamber_key][2],
                    'Col_Dig_ID_Conf': address_counts[chamber_key][3],
                    'Detected_Cells': address_counts[chamber_key][4]
                },
                ignore_index=True
            )

    if count_hist == 1:
        os.chdir(save_path)
        plt.hist(cell_counts_df['Detected_Cells'], color='lightcoral', bins=35)
        plt.title('summary_cell_count_histogram')
        plt.xlabel('number_of_detected_cells')
        plt.ylabel('number_of_chambers_on_chip')
        plt.tight_layout()
        plt.savefig('_summary_cell_count_histogram' + '.png')
        plt.close()
        plt.hist(apartments_per_image, color='dodgerblue', bins=18)
        plt.title('apartments_per_image_in_directory')
        plt.xlabel('number_of_detected_apartments')
        plt.ylabel('number_of_images')
        plt.tight_layout()
        plt.savefig('_summary_apartment_count_histogram' + '.png')
        plt.close()
        os.chdir('..')
    os.chdir(save_path)
    cell_counts_df.to_csv('_directory_cell_counts_relative_chamber_id' + version + '.csv')
    naive_chamber_id_rate = np.around(num_chambers_detected / (num_images * 24), decimals=4)
    metadata = open('analysis_metadata.txt', 'w')
    metadata.write('data_location: ' + targetdirectory + '\n')
    metadata.write('datetime: ' + dt.now().strftime('%Y_%m_%d_%H_%M') + '\n')
    metadata.write('version: ' + version + '\n' + '\n')
    metadata.write('apartment-finding method: ' + apartment_workflow + '\n')
    metadata.write('cell-counting method: ' + cell_workflow + '\n')

    if flip:
        metadata.write('\n' + 'image_flip: True')
    else:
        metadata.write('\n' + 'image_flip: False')
    if save_process_pics == 0:
        metadata.write('\n' + 'save_processing_pics: False')
    else:
        metadata.write('\n' + 'save_processing_pics: True')
    if count_hist == 0:
        metadata.write('\n' + 'directory_chamber_cell_count_histogram: False')
    else:
        metadata.write('\n' + 'directory_chamber_cell_count_histogram: True' + '\n')
    metadata.write('\n')
    metadata.write('# of images analyzed: ' + str(num_images) + '\n')
    metadata.write('total # of chambers detected: ' + str(num_chambers_detected) + '\n')
    metadata.write('naive_chamber_id_rate: ' + str(naive_chamber_id_rate * 100) + '%' + '\n')
    metadata.write('\n')
    metadata.write('window_thresh: ' + str(window_thresh) + '\n')
    metadata.write('gauss_blur_sigma: ' + str(gauss_blur_sigma) + '\n')
    metadata.write('scaling_thresh: ' + str(scaling_thresh) + '\n')
    metadata.write('min_blob_area: ' + str(min_blob_area) + '\n')
    metadata.write('max_blob_area: ' + str(max_blob_area) + '\n')
    metadata.write('min_blob_extent: ' + str(min_blob_extent) + '\n')
    metadata.write('tophat_selem: ' + str(tophat_selem) + '\n')
    metadata.write('min_cell_area: ' + str(min_cell_area) + '\n')
    metadata.write('max_cell_area: ' + str(max_cell_area) + '\n')
    metadata.close()
    os.chdir('..')
    print("Naive chamber detection rate for chip: " + str(naive_chamber_id_rate * 100) + '%')
    return cell_counts_df
