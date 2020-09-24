#!/usr/bin/env python3

import click
from cell_counter import cell_counter


@click.command(
    'extract_command',
    help="""Extracts apartment sub-regions from images in given target_directory""")
@click.argument('target_directory', type=click.Path(exists=True))  # no help statements for required args
@click.option(
    '--flip_horizontal', default=False, show_default=True,
    help="""whether to flip the images horizontally""")
@click.option(
    '--save_digit_images', type=bool, default=False, show_default=True,
    help="""set to True to save extracted digit regions from apartment addresses""")
def cli(
        target_directory,
        flip_horizontal,
        save_digit_images,
):
    print('Extracting apartment data in target directory: %s' % target_directory)
    cell_counter.process_directory_extract_apartments(
        target_directory,
        flip_horizontal=flip_horizontal,
        save_digit_images=save_digit_images
    )
    print('DONE')


if __name__ == '__main__':
    cli()
