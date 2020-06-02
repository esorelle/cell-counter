import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from datetime import datetime as dt
from scipy import stats
from scipy import ndimage as ndi
from skimage import filters, morphology, util
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from cell_counter import utils

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
    blobs_1 = utils.remove_large_objects(blobs_0, max_blob_area)
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
        rect_mask[south[i]-216:south[i], west[i]-5:east[i]+5] = 1

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

    # tabulate the counted cells by chamber for output -- original white tophat counting method
    prefix = img_name[img_name.find('ST_'):img_name.find('ST_') + 14] + '_CHAMBER_'
    address_counts = {}
    chamber_cell_count_array = np.zeros(len(north))
    for p in range(len(x_coords)):
        for c in range(len(true_north)):
            if west[c] < x_coords[p] < east[c]:
                if south[c] - 212 < y_coords[p] < south[c]-15:
                    chamber_cell_count_array[c] += 1


    ### CONTOUR COUNTING (2020-06-02) ###
    chamber_cell_count_array_contours = []
    gate_img = img_scaled * rect_mask  # changed from apartment_mask to better detect cells on apt edges
    gate_img = gate_img > 1.01  # 260 for apartment_mask, 1.01 for rect_mask
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

    chamber_cell_count_array_contours = np.zeros(len(true_north))
    apt_blobs = morphology.dilation(apartment_mask)
    apt_blobs = label(apt_blobs)  # revert to label(apartment_mask) if no dilation
    contour_points = []

    for apt in range(len(ordered_apartments)):
        temp_mask = np.zeros(np.shape(input_img))
        cv2.drawContours(temp_mask, [ordered_apartments[apt]], 0, (255, 255, 255), -1)
        temp_roi = label(temp_mask)
        # plt.imshow(temp_roi)
        # plt.show()
        for blob in regionprops(temp_roi):
            temp_roi_coords = [tuple(j) for j in blob.coords]
            for c in filtered_contours:
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                test_point = tuple([cy, cx])
                if test_point in temp_roi_coords:
                    chamber_cell_count_array_contours[apt] += 1
                    contour_points.append(test_point)  # use for scatter plot of counted cells

        chamber_cell_count_array_contours = [int(count) for count in chamber_cell_count_array_contours]


    # for i, apt in enumerate(regionprops(apt_blobs)):
    #     apt_coords = [tuple(j) for j in apt.coords]
    #     for c in filtered_contours:
    #         M = cv2.moments(c)
    #         cx = int(M['m10']/M['m00'])
    #         cy = int(M['m01']/M['m00'])
    #         test_point = tuple([cy, cx])
    #         if test_point in apt_coords:
    #             chamber_cell_count_array_contours[i] += 1
    #             contour_points.append(test_point)  # use for scatter plot of counted cells
    #
    # chamber_cell_count_array_contours = chamber_cell_count_array_contours.tolist()
    # chamber_cell_count_array_contours = [int(count) for count in chamber_cell_count_array_contours]

    fig, ax = plt.subplots(figsize=(10, 6))
    # subbed new_img for input_img to show detected apartment and text regions
    ax.imshow(new_img, cmap='gray')

    for point in contour_points:
        ax.scatter(point[1], point[0], s=4, c='lightcoral', marker='x')

    ax.set_axis_off()
    plt.title('#_of_detected_cell_contours_in_image = ' + str(len(contour_points)))
    plt.tight_layout()
    plt.savefig(img_name + '_detected_cell_contours' + '.png')
    plt.close()
    ### END CONTOUR COUNTING ###


    ### SPECTRAL CLUSTER MULTI CELL SPLITTING (SCOTT'S LUNGMAP METHOD) ###
    # split_cell_contours = utils.split_multi_cell(input_img, gate_img, max_cell_area, plot=False)
    # print('# contours from spectral clustering: ', len(split_cell_contours))
    #
    # cell_split_points = []
    #
    # for i, apt in enumerate(regionprops(apt_blobs)):
    #     apt_coords = [tuple(j) for j in apt.coords]
    #     for c in split_cell_contours:
    #         M = cv2.moments(c)
    #         cx = int(M['m10']/M['m00'])
    #         cy = int(M['m01']/M['m00'])
    #         test_point = tuple([cy, cx])
    #         if test_point in apt_coords:
    #             chamber_cell_count_array_contours[i] += 1
    #             cell_split_points.append(test_point)  # use for scatter plot of counted cells
    #
    # fig, ax = plt.subplots(figsize=(10, 6))
    # # subbed new_img for input_img to show detected apartment and text regions
    # ax.imshow(new_img, cmap='gray')
    #
    # for point in cell_split_points:
    #     ax.scatter(point[1], point[0], s=4, c='lightcoral', marker='x')
    #
    # ax.set_axis_off()
    # plt.title('#_of_detected_cell_contours_in_image = ' + str(len(cell_split_points)))
    # plt.tight_layout()
    # plt.savefig(img_name + '_detected_split_cells' + '.png')
    # plt.close()
    ### END SPECTRAL CLUSTER SPLITTING ###


    for chamber in range(len(chamber_cell_count_array)):
        if chamber < 9:
            address_counts[prefix + '00' + str(chamber + 1)] = [
                str(row_numbers[chamber]),
                str(col_numbers[chamber]),
                row_num_avg_conf[chamber],
                col_num_avg_conf[chamber],
                int(chamber_cell_count_array[chamber]),
                chamber_cell_count_array_contours[chamber]
            ]
        else:
            address_counts[prefix + '0' + str(chamber + 1)] = [
                str(row_numbers[chamber]),
                str(col_numbers[chamber]),
                row_num_avg_conf[chamber],
                col_num_avg_conf[chamber],
                int(chamber_cell_count_array[chamber]),
                chamber_cell_count_array_contours[chamber]
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
            'Tophat_Cell_Count',
            'Contour_Cell_Count'
        ]
    )
    num_chambers_detected = 0
    apartments_per_image = []

    for i in range(0, num_images):
        img_name = images[i]
        print(i, img_name)

        raw_img = plt.imread(img_name)
        if flip:
            raw_img = utils.flip_horizontal(raw_img)

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
                    'Tophat_Cell_Count': address_counts[chamber_key][4],
                    'Contour_Cell_Count': address_counts[chamber_key][5]
                },
                ignore_index=True
            )

    if count_hist == 1:
        os.chdir(save_path)

        plt.hist(cell_counts_df['Tophat_Cell_Count'], color='lightcoral', bins=35)
        plt.title('tophat_cell_count_histogram')
        plt.xlabel('number_of_detected_cells')
        plt.ylabel('number_of_chambers_on_chip')
        plt.tight_layout()
        plt.savefig('_tophat_cell_count_histogram' + '.png')
        plt.close()

        plt.hist(cell_counts_df['Contour_Cell_Count'], color='mediumseagreen', bins=35)
        plt.title('contour_cell_count_histogram')
        plt.xlabel('number_of_detected_cells')
        plt.ylabel('number_of_chambers_on_chip')
        plt.tight_layout()
        plt.savefig('_contour_cell_count_histogram' + '.png')
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
