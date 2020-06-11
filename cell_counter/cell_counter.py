import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from skimage import morphology, util
from skimage.measure import label, regionprops


from cell_counter import utils, core

workflow_list = [
    'denoise',  # 1
    'norm',
    'edges',  # 2
    'dilate',
    'invert',
    'maxout',
    'close',
    'extentout',
    'open',
    'minout',
    'label',
    'borderout',
    'de-rotate',  # 3
    'apartment_mask',  # 4
    'read_digits'
    # 5 - cell counting (multiple types)
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
    img_scaled = core.light_correction(input_img, gauss_blur_sigma, window_thresh)

    # ???: Wrap blob detection in core function (with saving of plots)
    edges_0 = (img_scaled < scaling_thresh)
    # 2,3 --> changed to 1,3 to reduce connectivity of dense cells
    # leading to false negative for very full apartments
    # while retaining connection of apartment entry points.
    edges_1 = morphology.dilation(edges_0, morphology.selem.rectangle(1, 4))

    blobs_0 = util.invert(edges_1)
    blobs_1 = utils.remove_large_objects(blobs_0, max_blob_area)
    # 3,1 --> changed to 4,1 to increase blob grouping of dense cells as contiguous apartment
    blobs_2 = morphology.closing(blobs_1, morphology.selem.rectangle(5, 1))

    blobs_3 = core.filter_blobs_by_extent(blobs_2, min_blob_extent)

    blobs_4 = morphology.opening(blobs_3, morphology.selem.disk(3))
    blobs_5 = blobs_4 > 0
    blobs_6 = morphology.remove_small_objects(blobs_5, min_blob_area)
    blobs_7 = label(blobs_6, connectivity=2)

    refined_blobs, blob_boundaries = core.filter_blobs_near_edge(blobs_7)

    # de-rotate the image by finding chamber row angles and correcting
    rows, c_centers = core.find_rows(blob_boundaries)

    # correct image rotation
    # ???: Should this take pre-processed image instead of original input image?
    new_img, img_rot, r_deg_mean, n_cols, n_rows = core.rotate_image(input_img, rows, c_centers)

    # get text and apartment regions
    new_img, apartment_mask, row_text_regions, col_text_regions, ordered_apartments = core.get_key_regions(
        new_img,
        img_rot,
        c_centers,
        r_deg_mean,
        n_cols,
        n_rows
    )

    # read row and column numbers by template matching
    row_numbers, row_num_avg_conf, col_numbers, col_num_avg_conf = core.read_digits(row_text_regions, col_text_regions)

    # save key region image
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

    # count the cells in each chamber -- tophat method
    labels = core.detect_cells_tophat(input_img, window_thresh, tophat_selem, sigma=1)

    # show detected centroids overlay on key region image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(new_img, cmap='gray')

    x_coords, y_coords = [], []
    for region in regionprops(labels):
        if min_cell_area < region.area < max_cell_area:
            cent_x, cent_y = region.centroid
            if apartment_mask[int(np.round(cent_x)), int(np.round(cent_y))][0] > 0:
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
    chamber_cell_count_array = core.count_cells_tophat(blob_boundaries, x_coords, y_coords)

    # find and count cells by chamber for output -- cell contour method
    rect_mask = core.make_rectangle_mask(refined_blobs, blob_boundaries)
    chamber_cell_count_array_contours, contour_points = core.detect_and_count_cell_contours(
        img_scaled,
        rect_mask,
        ordered_apartments,
        min_cell_area,
        max_cell_area
    )

    # show detected centroids overlay on key region image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(new_img, cmap='gray')

    for point in contour_points:
        ax.scatter(point[1], point[0], s=4, c='lightcoral', marker='x')

    ax.set_axis_off()
    plt.title('#_of_detected_cell_contours_in_image = ' + str(len(contour_points)))
    plt.tight_layout()
    plt.savefig(img_name + '_detected_cell_contours' + '.png')
    plt.close()

    # SPECTRAL CLUSTER MULTI CELL SPLITTING (SCOTT'S LUNGMAP METHOD) ###
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
    # END SPECTRAL CLUSTER SPLITTING ###

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
