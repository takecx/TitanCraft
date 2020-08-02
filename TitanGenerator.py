import sys
import os
import subprocess
import glob


class TitanGenerator(object):
    def __init__(self, input_image, kernel_size, iteration, w_max, h_max):
        self.input_image = input_image
        input_basename = os.path.splitext(
            os.path.basename(input_image))[0]
        self.pifu_obj = os.path.join('./results/pifu_demo/',
                                     'result_' + input_basename + '_fg.obj')
        # remove_bg
        self.kernel_size = kernel_size
        self.iteration = iteration
        # obj2schematic
        self.w_max = w_max
        self.h_max = h_max

        self._add_path()

    def _add_path(self):
        remove_bg_path = os.path.join(
            os.path.abspath(''), 'remove_bg/FBA_Matting')
        if not remove_bg_path in sys.path:
            sys.path.append(remove_bg_path)

    def generate(self):
        print('***************************** remove_bg *****************************')
        self._remove_background()
        print('******************************* PIFu *******************************')
        self._apply_PIFu()
        print('*************************** obj2schematic ***************************')
        self._convert_obj2schematic()

    def _remove_background(self):
        commands = ['python', 'remove_bg/BackgroundRemover.py',
                    self.input_image, '--kernel_size', str(self.kernel_size), '--iteration', str(self.iteration)]
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
        TEST_FOLDER_PATH = './output'

        commands = ['python', './PIFu/apps/eval.py',
                    '--name', NAME, '--batch_size', str(BATCH_SIZE),
                    '--num_stack', str(4), '--num_hourglass', str(2),
                    '--resolution', str(VOL_RES), '--hg_down', 'ave_pool', '--norm', 'group',
                    '--norm_color', 'group', '--test_folder_path', TEST_FOLDER_PATH, '--load_netG_checkpoint_path',
                    CHECKPOINTS_NETG_PATH, '--load_netC_checkpoint_path', CHECKPOINTS_NETC_PATH]
        mlp_dims = [v for v in MLP_DIM.split(' ')]
        commands.extend(['--mlp_dim'])
        commands.extend(mlp_dims)
        mlp_colors = [v for v in MLP_DIM_COLOR.split(' ')]
        commands.extend(['--mlp_dim_color'])
        commands.extend(mlp_colors)
        subprocess.run(commands)

        # postprocess
        for p in glob.glob('./output/*.png'):
            if os.path.isfile(p):
                os.remove(p)

    def _convert_obj2schematic(self):
        commands = ['python', 'obj2schematic/Obj2SchemticConverter.py',
                    self.pifu_obj, '--h_max', str(self.h_max), '--w_max', str(self.w_max)]
        subprocess.run(commands)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image",
                        help="input .png image path")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=3)
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
        generator = TitanGenerator(
            args.input_image, args.kernel_size, args.iteration, args.w_max, args.h_max)
        generator.generate()
    except Exception as e:
        import traceback
        print('Titan generation failed: ', e)
        print(traceback.format_exc())
