#!/usr/bin/env python3

import click
from cell_counter import cell_counter


@click.command(
    'count_command',
    help="""count_command analyzes images to count cells in each chamber
    from the command line. The bottom of each chamber is identified and used to
    define a bounding box of known height to prevent false positive counts
    arising from the chamber frits. The image data path (target_directory) must
    exist and be passed as an argument. See below for optional arguments
    and defaults.""")
@click.argument('target_directory', type=click.Path(exists=True))  # no help statements for required args
@click.option(
    '--min_cell_area', default=450, show_default=True,
    help="""sets minimum number of pixels to consider a cell""")
@click.option(
    '--max_cell_area', default=650, show_default=True,
    help="""sets maximum number of pixels to consider a cell""")
@click.option(
    '--flip_horizontal', default=False, show_default=True,
    help="""whether to flip the images horizontally""")
@click.option(
    '--save_process_pics', type=bool, default=True, show_default=True,
    help="""set to 1 to save images of all applied image processing steps""")
@click.option(
    '--count_hist', type=bool, default=True, show_default=True,
    help="""set to 1 to save a summary histogram of chamber cell counts for directory""")
def cli(
        target_directory,
        min_cell_area,
        max_cell_area,
        flip_horizontal,
        save_process_pics,
        count_hist,
):
    print('Processing images in target directory: %s' % target_directory)
    cell_counter.process_directory(
        target_directory,
        min_cell_area,
        max_cell_area,
        flip_horizontal=flip_horizontal,
        save_process_pics=save_process_pics,
        count_hist=count_hist
    )
    print('PROCESSING COMPLETE - refer to analysis_metadata.txt for analysis summary')


if __name__ == '__main__':
    cli()
