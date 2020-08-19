import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

from cell_counter import utils, core


def preprocess_image(input_img, flip_horizontal=True):
    if flip_horizontal:
        input_img = utils.flip_horizontal(input_img)

    img_scaled = core.light_correction(input_img)
    img_h, img_w = img_scaled.shape

    fid_centers = core.find_fiducial_locations(img_scaled)
    rows = core.find_rows(fid_centers)
    rot_degrees = core.find_rotation_angle(fid_centers, rows)
    img_corrected = core.rotate_image(img_scaled, rot_degrees)

    # rotate fiducials around the center of the image
    fid_centers_corrected = core.rotate_points(fid_centers, (img_h / 2., img_w / 2.), rot_degrees)

    return img_corrected, fid_centers_corrected


def extract_image_apartment_data(
        img_path,
        min_cell_area,
        max_cell_area,
        digit_dir
):
    input_img = plt.imread(img_path)
    img_base_name = os.path.basename(img_path)

    img_corrected, fid_centers_corrected = preprocess_image(input_img)
    # fid_img = core.render_fiducials(img_corrected, fid_centers_corrected)
    # plt.figure(figsize=(16, 16))
    # fig = plt.imshow(fid_img)

    apt_data = core.identify_apartments(img_corrected, fid_centers_corrected, digit_dir=digit_dir)

    # count the cells in each chamber -- simple percent of apartment method
    for apt in apt_data:
        edge_contours, edge_mask, non_edge_contours, non_edge_mask = core.find_apartment_blobs(apt['apt_region'])

        edge_blob_area = (edge_mask > 0).sum()
        edge_blob_apt_ratio = edge_blob_area / utils.apt_ref_area
        edge_cell_count_min = round(edge_blob_area / max_cell_area)
        edge_cell_count_max = round(edge_blob_area / min_cell_area)

        non_edge_blob_area = (non_edge_mask > 0).sum()
        non_edge_blob_apt_ratio = non_edge_blob_area / utils.apt_ref_area
        non_edge_cell_count_min = round(non_edge_blob_area / (0.75 * max_cell_area))
        non_edge_cell_count_max = round(non_edge_blob_area / (0.75 * min_cell_area))

        apt['image_name'] = img_base_name
        apt['edge_blob_area'] = edge_blob_area
        apt['edge_blob_apt_ratio'] = edge_blob_apt_ratio
        apt['edge_cell_count_min'] = edge_cell_count_min
        apt['edge_cell_count_max'] = edge_cell_count_max
        apt['non_edge_blob_area'] = non_edge_blob_area
        apt['non_edge_blob_apt_ratio'] = non_edge_blob_apt_ratio
        apt['non_edge_cell_count_min'] = non_edge_cell_count_min
        apt['non_edge_cell_count_max'] = non_edge_cell_count_max
        apt['edge_contours'] = edge_contours
        apt['edge_mask'] = edge_mask
        apt['non_edge_contours'] = non_edge_contours
        apt['non_edge_mask'] = non_edge_mask

    return apt_data


def process_directory(
        target_directory,
        min_cell_area,
        max_cell_area,
        save_process_pics,
        save_digit_images,
        count_hist=False
):
    version = 'v20200617'
    time_stamp = dt.now().strftime('%Y_%m_%d_%H_%M_%S')

    save_path = 'analysis_%s' % time_stamp
    save_path = os.path.join(target_directory, save_path)
    os.mkdir(save_path)

    images = glob.glob(os.path.join(target_directory, '*.tif'))
    img_count = len(images)

    if img_count == 0:
        raise(FileNotFoundError("No TIFF images found in %s" % target_directory))

    if save_digit_images:
        digit_dir = os.path.join(save_path, 'digits')
        os.mkdir(digit_dir)
    else:
        digit_dir = None

    if save_process_pics:
        process_fig_dir = os.path.join(save_path, 'process_figures')
        os.mkdir(process_fig_dir)
    else:
        process_fig_dir = None

    total_apt_count = 0
    apartments_per_image = []
    apt_data_df_list = []

    for i, img_path in enumerate(images):
        img_base_name = os.path.basename(img_path)
        print(i, img_base_name)

        apt_data = extract_image_apartment_data(
            img_path,
            min_cell_area,
            max_cell_area,
            digit_dir
        )

        if process_fig_dir is not None:
            for apt in apt_data:
                fig = core.render_apartment(apt)
                fig_name = "_".join(['apt_fig', apt['image_name'], apt['row_address'], apt['col_address']])
                fig_name = '.'.join([fig_name, 'png'])
                fig.savefig(os.path.join(process_fig_dir, fig_name))
                plt.close()

        apt_count = len(apt_data)
        total_apt_count += apt_count
        apartments_per_image.append(apt_count)

        print('\t%s: %d chambers counted' % (img_base_name, apt_count))

        df_apt_data = pd.DataFrame(
            apt_data,
            columns=[
                'image_name',
                'row_address',
                'col_address',
                'fid_x',
                'fid_y',
                'edge_blob_area',
                'edge_blob_apt_ratio',
                'edge_cell_count_min',
                'edge_cell_count_max'
            ]
        )

        apt_data_df_list.append(df_apt_data)

    all_apt_data_df = pd.concat(apt_data_df_list, ignore_index=True)

    if count_hist:
        # TODO: can we show both min & max cell counts from the simple apt capacity method?
        plt.hist(all_apt_data_df['edge_cell_count_min'], color='lightcoral', bins=35)
        plt.title('Simple Cell Count - Min')
        plt.xlabel('cell count')
        plt.ylabel('# of chambers on chip')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '_simple_cell_count_min_histogram.png'))
        plt.close()

        plt.hist(apartments_per_image, color='dodgerblue', bins=18)
        plt.title('Apartments Per Image')
        plt.xlabel('# of detected apartments')
        plt.ylabel('image count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, '_summary_apartment_count_histogram.png'))
        plt.close()

    all_apt_data_df.to_csv(
        os.path.join(
            save_path,
            '_directory_cell_counts_%s.csv' % version
        )
    )

    # We expect roughly 24 apartments to be fully visible in each image
    naive_chamber_id_rate = np.around(total_apt_count / (img_count * 24.), decimals=4)

    metadata = open(os.path.join(save_path, 'analysis_metadata.txt'), 'w')

    metadata.write('data_location: %s\n' % target_directory)
    metadata.write('datetime: %s\n' % time_stamp)
    metadata.write('version: %s\n\n' % version)
    metadata.write('save_processing_pics: %s\n' % save_process_pics)
    metadata.write('count_histogram: %s\n' % count_hist)
    metadata.write('# of images analyzed: %d\n' % img_count)
    metadata.write('total # of chambers detected: %d\n' % total_apt_count)
    metadata.write('naive_chamber_id_rate: %f%%\n' % (naive_chamber_id_rate * 100))
    metadata.write('min_cell_area: %d\n' % min_cell_area)
    metadata.write('max_cell_area: %d\n' % max_cell_area)

    metadata.close()

    print("Naive chamber detection rate for chip: %.2f%%\n" % (naive_chamber_id_rate * 100))

    return all_apt_data_df
