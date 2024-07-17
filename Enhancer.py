import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer


def main():
    """Inference demo for choosing between Real-ESRGAN and DMSP-RealESRGAN."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-m', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model to use: RealESRGAN_x4plus or DMSP-RealESRGAN_x4plus')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU device to use (default=None), can be 0,1,2 for multi-gpu')
    args = parser.parse_args()

    # Model setup
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    # Selecting the model based on user input
    if args.model_name == 'RealESRGAN_x4plus':
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif args.model_name == 'DMSP-RealESRGAN_x4plus':
        file_url = '<URL_to_your_fine_tuned_model>'  # Replace <URL_to_your_fine_tuned_model> with the actual URL
    else:
        raise ValueError("Unsupported model name. Use 'RealESRGAN_x4plus' or 'DMSP-RealESRGAN_x4plus'.")

    # Model path
    model_path = os.path.join('weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = load_file_from_url(url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Restorer
    upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, half=not args.fp32, gpu_id=args.gpu_id)

    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        try:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error:', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            save_path = os.path.join(args.output, f'{imgname}_out.{extension}')
            cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
