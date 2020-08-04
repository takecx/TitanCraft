import sys
import os
import subprocess
import glob
import cv2
import numpy as np


class TitanGenerator(object):
    def __init__(self, args):
        self.input_image = args.input_image
        input_basename = os.path.splitext(
            os.path.basename(self.input_image))[0]
        self.temp_image = os.path.join(
            './temp/', os.path.basename(self.input_image))
        self.pifu_obj = os.path.join('./PIFu/results/pifu_demo/',
                                     'result_' + input_basename + '.obj')
        self.obj2chem_input = os.path.join('./PIFu/results/pifu_demo/',
                                           input_basename + '.obj')

        self.resize = args.resize
        # remove_bg
        self.kernel_size = args.kernel_size
        self.iteration = args.iteration
        # obj2schematic
        if args.output_dir == None:
            self.output_dir = os.path.join(
                os.path.dirname(__file__), 'output')
        else:
            self.output_dir = args.output_dir
        self.w_max = args.w_max
        self.h_max = args.h_max

        self._add_path()

    def _add_path(self):
        remove_bg_path = os.path.join(
            os.path.abspath(''), 'remove_bg/FBA_Matting')
        if not remove_bg_path in sys.path:
            sys.path.append(remove_bg_path)

    def generate(self):
        print('***************************** preprocess *****************************')
        self._preprocess_image()
        print('***************************** remove_bg *****************************')
        self._remove_background()
        print('******************************* PIFu *******************************')
        self._apply_PIFu()
        print('*************************** obj2schematic ***************************')
        self._convert_obj2schematic()
        print('***************************** postprocess *****************************')
        self._postprocess()

    def _preprocess_image(self):
        # remove_bg
        if os.path.exists('./remove_bg/output/'):
            for p in glob.glob('./remove_bg/output/*.png'):
                if os.path.isfile(p):
                    os.remove(p)

        print('start image preprocess....')
        target_formats = ['.png', '.jpg', '.jpeg']
        if not os.path.splitext(self.input_image)[1].lower() in target_formats:
            raise Exception('input_image format should be ' +
                            str(target_formats))
        # read image
        org_img = cv2.imread(self.input_image)
        # calc resize factor
        h, w, c = org_img.shape
        if w > h:
            factor = self.resize / w
            new_size = int(h * factor)
            new_shape = (self.resize, new_size)
        else:
            factor = self.resize / h
            new_size = int(w * factor)
            new_shape = (new_size, self.resize)
        new_img = cv2.resize(org_img, new_shape,
                             interpolation=cv2.INTER_LANCZOS4)
        # add margin
        half_margin = int((self.resize - new_size) / 2)
        if w > h:
            margin_val = np.full((half_margin, self.resize, 3), 255)
            new_img = np.concatenate([margin_val, new_img, margin_val], axis=0)
        else:
            margin_val = np.full((self.resize, half_margin, 3), 255)
            new_img = np.concatenate([margin_val, new_img, margin_val], axis=1)
        # save preprocessed image
        os.makedirs('./temp/', exist_ok=True)
        cv2.imwrite(self.temp_image, new_img)
        print('finish image preprocess....')

    def _postprocess(self):
        print('cleaning....')
        # temp image
        for p in glob.glob('./temp/*'):
            if os.path.isfile(p):
                os.remove(p)

        os.rmdir('./temp/')
        print('finish post process....')

    def _remove_background(self):
        commands = ['python', 'remove_bg/BackgroundRemover.py',
                    self.temp_image, '--kernel_size', str(self.kernel_size), '--iteration', str(self.iteration)]
        subprocess.run(commands)

    def _apply_PIFu(self):

        # Training
        GPU_ID = 0
        NAME = 'pifu_demo'
        # Network configuration
        BATCH_SIZE = 1
        MLP_DIM = '257 1024 512 256 128 1'
        MLP_DIM_COLOR = '513 1024 512 256 128 3'
        # Reconstruction resolution
        # NOTE: one can change here to reconstruct mesh in a different resolution.
        VOL_RES = 256
        CHECKPOINTS_NETG_PATH = './PIFu/checkpoints/net_G'
        CHECKPOINTS_NETC_PATH = './PIFu/checkpoints/net_C'
        TEST_FOLDER_PATH = './remove_bg/output'
        RESULTS_PATH = './PIFu/results'

        commands = ['python', './PIFu/apps/eval.py',
                    '--name', NAME, '--batch_size', str(BATCH_SIZE),
                    '--num_stack', str(4), '--num_hourglass', str(2),
                    '--resolution', str(VOL_RES), '--hg_down', 'ave_pool', '--norm', 'group',
                    '--norm_color', 'group', '--test_folder_path', TEST_FOLDER_PATH, '--load_netG_checkpoint_path',
                    CHECKPOINTS_NETG_PATH, '--load_netC_checkpoint_path', CHECKPOINTS_NETC_PATH,
                    '--results_path', RESULTS_PATH]
        mlp_dims = [v for v in MLP_DIM.split(' ')]
        commands.extend(['--mlp_dim'])
        commands.extend(mlp_dims)
        mlp_colors = [v for v in MLP_DIM_COLOR.split(' ')]
        commands.extend(['--mlp_dim_color'])
        commands.extend(mlp_colors)
        subprocess.run(commands)

    def _convert_obj2schematic(self):
        # PIFuの出力ファイル名に'result_'が入っているのでファイル名を変更
        os.rename(self.pifu_obj, self.obj2chem_input)

        commands = ['python', 'obj2schematic/Obj2SchemticConverter.py',
                    self.obj2chem_input, '--output_dir', self.output_dir, '--h_max', str(self.h_max), '--w_max', str(self.w_max)]
        subprocess.run(commands)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image",
                        help="input .png image path")
    parser.add_argument('--resize', type=int, default=512,
                        help='preprocessed image size')
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=3)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir of generated .schematic file.')
    parser.add_argument('--h_max', type=int, default=100,
                        help='Max height of converted schematic object.')
    parser.add_argument('--w_max', type=int, default=100,
                        help='Max width(length) of converted schematic object.')
    args = parser.parse_args()
    print(args.input_image, args.kernel_size, args.iteration)
    return args


if __name__ == '__main__':
    try:
        print('checking arguments...')
        args = get_args()
        print('start titan generating...')
        generator = TitanGenerator(args)
        generator.generate()
    except Exception as e:
        import traceback
        print('Titan generation failed: ', e)
        print(traceback.format_exc())
