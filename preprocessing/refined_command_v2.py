#!/usr/bin/env python3

import click
from refined_process_v2 import flip_horizontal, rotate, _check_dtype_supported, remove_large_objects, get_chamber_cell_counts_bf, process_directory_relative_id

@click.command('count_command',
    help="""very_lean_command analyzes images to count cells in each chamber
    from the command line. The bottom of each chamber is identified and used to
    define a bounding box of known height to prevent false positive counts
    arising from the chamber frits. The image data path (targetdirectory) must
    exist and be passed as an argument. See below for optional arguments
    and defaults.""")
@click.option('--flip', default=True, show_default=True,
    help="""flip images horizontally to read chamber addresses""")
@click.option('--gauss_blur_sigma', type=int, default=0.9, show_default=True,
    help="""sets extent of image blur to de-noise before blob detection""")
@click.option('--window_thresh', type=int, default=9, show_default=True,
    help="""sets size of window used for local adaptive background norm""")
@click.option('--scaling_thresh', type=int, default=247, show_default=True,
    help="""sets value threshold for initial edge creation post-background norm""")
@click.option('--min_blob_area', type=int, default=2000, show_default=True,
    help="""sets minimum area of detected blob for chamber detection""") # set back to 1500 if needed
@click.option('--max_blob_area', type=int, default=15500, show_default=True,
    help="""sets maximum area of detected blob for chamber detection""")
@click.option('--min_blob_extent', type=float, default=0.25, show_default=True,
    help="""sets min pct of bounding box blob must occupy (>25% for chambers)""")
@click.option('--tophat_selem', default=9, show_default=True,
    help="""sets size of white tophat structuring element for cell detection""")
@click.option('--min_cell_area', default=35, show_default=True,
    help="""sets minimum number of pixels to consider a cell""")
@click.option('--max_cell_area', default=1000, show_default=True,
    help="""sets maximum number of pixels to consider a cell""")
@click.option('--save_process_pics', type=int, default=0, show_default=True,
    help="""set to 1 to save images of all applied image processing steps""")
@click.option('--count_hist', type=int, default=1, show_default=True,
    help="""set to 1 to save a summary histogram of chamber cell counts for directory""")
@click.argument('targetdirectory', type = click.Path(exists=True)) # no help statements for required arguments
def cli(flip, gauss_blur_sigma, window_thresh, scaling_thresh, min_blob_area, max_blob_area, min_blob_extent, tophat_selem, min_cell_area, max_cell_area, save_process_pics, count_hist, targetdirectory):
    print('counting cells in: ' + targetdirectory + '...')
    process_directory_relative_id(flip, gauss_blur_sigma, window_thresh, scaling_thresh, min_blob_area, max_blob_area, min_blob_extent, tophat_selem, min_cell_area, max_cell_area, save_process_pics, count_hist, targetdirectory)
    print('...counting finished')
    print('refer to analysis_metadata.txt for running parameters details')

if __name__ == '__main__':
    cli()
