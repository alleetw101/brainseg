import argparse
import os

from .brainsegpicture import BrainSegPicture
from .brainsegnifti import BrainSegNifti


def parsercli():
    parser = argparse.ArgumentParser(prog='brainseg')

    parser.add_argument('filepath', type=str, help='Path of file to segment')

    parser.add_argument('-s', '--segment', type=str, help='Segments using provided SavedModel (Filepath required)')
    parser.add_argument('--segment-precision', type=int, choices=[32, 64], help='Float32 or float64')

    parser.add_argument('--info', '-i', action='store_true')

    parser.add_argument('--output', '-o', type=str, default='segmentednifti.nii.gz', help='Output filepath')
    parser.add_argument('--output-channels', type=int, choices=[1, 3], help='Number of channel in output image',
                        default=1)

    args = parser.parse_args()

    # Format parsing
    supported_image_ext = ['.png', '.jpg', '.jpeg']
    supported_nifti_ext = ['.nii', '.nii.gz', '.gz']
    _, extension = os.path.splitext(args.filepath)
    if extension.lower() in supported_image_ext:
        data = BrainSegPicture(args.filepath, extension)
        brainsegpicture_parser(args, data)
    elif extension.lower() in supported_nifti_ext:
        data = BrainSegNifti(args.filepath, args.segment_precision)
        nifti_parser(args, data)
    else:
        print(f'Input file format not supported. \nPlease use the following formats: {supported_image_ext}')


def brainsegpicture_parser(args, data):
    if args.info:
        data.information()
    if args.output:
        data.save_image(args.output, args.output_channels)


def nifti_parser(args, data):
    if args.segment:
        data.segment_nifti(args.segment)
    if args.output:
        data.write_nifti(args.output)


def main():
    parsercli()
