# coding: utf-8
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
import argparse
import os
import glob
import time
import matplotlib.pyplot as plt
import sys

sys.path.append('./data_loader/')
from data_augmentation import *
from networks.faceswap_gan_model import FaceswapGANModel
from keras_vggface.vggface import VGGFace
from keras.layers import *
from pathlib import Path
from tqdm import tqdm
from utils import showG, showG_mask
from mxnet.gluon.data import Dataset, DataLoader
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser()

    # network architecture configs
    parser.add_argument('--resolution', '-res', type=int, default=64)
    parser.add_argument('--self-attention', '-attn', type=bool, default=True)
    parser.add_argument('--norm', type=str, default='instancenorm',
                        choices=['instancenorm', 'batchnorm', 'layernorm', 'groupnorm', 'none'])
    parser.add_argument('--model-capacity', '-mc', type=str, default='standard',
                        choices=['standard', 'lite'])

    # data augmentation configs
    parser.add_argument('--img-dir-A', '-iA', required=True)
    parser.add_argument('--img-dir-B', '-iB', required=True)
    parser.add_argument('--eyes-bm-dir-A', '-eA', required=True)
    parser.add_argument('--eyes-bm-dir-B', '-eB', required=True)
    parser.add_argument('--motion-blur', '-mb', type=bool, default=False,
                        help='set True if training data contains images extracted from videos')
    parser.add_argument('--bm-eyes', type=bool, default=True,
                        help='set True to use eye-aware training')
    parser.add_argument('--proba-random-color-match', type=float, default=0.5,
                        help='probability of random color matching (data augmentation)')
    parser.add_argument('--multi-processing', '-mp', type=int, default=8,
                        help='used for image pre-processing')

    # loss configuration
    parser.add_argument('--weight-discriminator', '-w-d', type=float, default=0.1)
    parser.add_argument('--weight-reconstruction', '-w-recon', type=float, default=1.)
    parser.add_argument('--weight-edge', '-w-edge', type=float, default=0.1)
    parser.add_argument('--weight-eyes', '-w-eyes', type=float, default=30.,
                        help='reconstruction and edge loss on eyes area')
    parser.add_argument('--weight-perceptual', '-w-pl', type=float, nargs=4,
                        default=[0.01, 0.1, 0.3, 0.1])

    parser.add_argument('--gan-training', '-gan', type=str, default='mixup_LSGAN',
                        choices=['mixup_LSGAN', 'relativistic_avg_LSGAN'])
    parser.add_argument('--use-pl', type=bool, default=True)
    parser.add_argument('--use-mask-hinge-loss', type=bool, default=False)
    parser.add_argument('--m-mask', type=float, default=0.)
    parser.add_argument('--lr-factor', type=float, default=1.)
    parser.add_argument('--use-cyclic-loss', type=bool, default=False)

    # training configs
    parser.add_argument('--gpu-device', '-gpu', type=int, default=0)
    parser.add_argument('--batch-size', '-bs', type=int, default=8)
    parser.add_argument('--models-dir', '-md', type=str, default='./results/models')
    parser.add_argument('--samples-dir', '-sd', type=str, default='./results/samples')
    parser.add_argument('--total-iter', type=int, default=40000)
    parser.add_argument('--display-iter', type=int, default=1000)
    parser.add_argument('--backup-iter', type=int, default=5000)

    return parser.parse_args()


class FaceSwapDataset(Dataset):
    def __init__(self, filenames, all_filenames, dir_bm_eyes,
                 resolution, **da_config):
        self.filenames = filenames
        self.all_filenames = all_filenames
        self.dir_bm_eyes = dir_bm_eyes
        self.resolution = resolution

        self.set_data_augm_config(
            da_config["prob_random_color_match"],
            da_config["use_da_motion_blur"],
            da_config["use_bm_eyes"])

    def set_data_augm_config(self, prob_random_color_match=0.5,
                             use_da_motion_blur=True, use_bm_eyes=True):
        self.prob_random_color_match = prob_random_color_match
        self.use_da_motion_blur = use_da_motion_blur
        self.use_bm_eyes = use_bm_eyes

    def __getitem__(self, idx):
        img = read_image(self.filenames[idx],
                         self.all_filenames,
                         self.dir_bm_eyes,
                         self.resolution,
                         self.prob_random_color_match,
                         self.use_da_motion_blur,
                         self.use_bm_eyes)
        return img

    def __len__(self):
        return len(self.filenames)


class DataLoaderWrapper(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_loader._batchify_fn = self._mp_batchify_fn
        self._gen = self.__gen__()

    def get_next_batch(self):
        return next(self._gen)

    def __gen__(self):
        while True:
            for i in self.data_loader:
                yield i

    def _mp_batchify_fn(self, data):
        if isinstance(data[0], tuple):
            data = zip(*data)
            return [self._mp_batchify_fn(i) for i in data]
        else:
            return np.asarray(data)


if __name__ == '__main__':
    '''parse arguments'''
    args = parse_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    K.set_learning_phase(1)

    RESOLUTION = args.resolution
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."
    arch_config = {'IMAGE_SHAPE': (RESOLUTION, RESOLUTION, 3),
                   'use_self_attn': args.self_attention,
                   'norm': args.norm,
                   'model_capacity': args.model_capacity}

    loss_weights = {'w_D': args.weight_discriminator,
                    'w_recon': args.weight_reconstruction,
                    'w_edge': args.weight_edge,
                    'w_eyes': args.weight_eyes,
                    'w_pl': args.weight_perceptual}
    loss_config = {"gan_training": args.gan_training,
                   'use_PL': args.use_pl,
                   'use_mask_hinge_loss': args.use_mask_hinge_loss,
                   'm_mask': args.m_mask,
                   'lr_factor': args.lr_factor,
                   'use_cyclic_loss': args.use_cyclic_loss}

    num_cpus = args.multi_processing
    da_config = {
        "prob_random_color_match": args.proba_random_color_match,
        "use_da_motion_blur": args.motion_blur,
        "use_bm_eyes": args.bm_eyes
    }
    img_dirA = args.img_dir_A
    img_dirB = args.img_dir_B
    img_dirA_bm_eyes = args.eyes_bm_dir_A
    img_dirB_bm_eyes = args.eyes_bm_dir_B

    # Path to saved model weights
    models_dir = args.models_dir
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    samples_dir = args.samples_dir
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    batchSize = args.batch_size
    display_iters = args.display_iter
    backup_iters = args.backup_iter
    TOTAL_ITERS = args.total_iter

    '''define model'''
    model = FaceswapGANModel(**arch_config)
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    model.build_pl_model(vggface_model=vggface)
    model.build_train_functions(loss_weights=loss_weights, **loss_config)

    '''define data_geneator and load images'''
    # Get filenames
    train_A = glob.glob(img_dirA + "/*.*g")
    train_B = glob.glob(img_dirB + "/*.*g")
    train_AnB = train_A + train_B

    assert len(train_A), "No image found in " + str(img_dirA)
    assert len(train_B), "No image found in " + str(img_dirB)
    print("Number of images in folder A: " + str(len(train_A)))
    print("Number of images in folder B: " + str(len(train_B)))
    if da_config['use_bm_eyes']:
        assert len(glob.glob(img_dirA_bm_eyes + "/*.*g")), "No binary mask found in " + str(img_dirA_bm_eyes)
        assert len(glob.glob(img_dirB_bm_eyes + "/*.*g")), "No binary mask found in " + str(img_dirB_bm_eyes)
        assert len(glob.glob(img_dirA_bm_eyes + "/*.*g")) == len(train_A), \
            "Number of faceA images does not match number of their binary masks. " \
            "Can be caused by any none image file in the folder."
        assert len(glob.glob(img_dirB_bm_eyes + "/*.*g")) == len(train_B), \
            "Number of faceB images does not match number of their binary masks. " \
            "Can be caused by any none image file in the folder."

    train_setA = FaceSwapDataset(train_A, train_AnB, img_dirA_bm_eyes,
                                 RESOLUTION, **da_config)
    train_setB = FaceSwapDataset(train_B, train_AnB, img_dirB_bm_eyes,
                                 RESOLUTION, **da_config)
    _DataLoader = partial(DataLoader, batch_size=batchSize, shuffle=True,
                          last_batch='rollover', num_workers=num_cpus)
    train_batchA = DataLoaderWrapper(_DataLoader(train_setA))
    train_batchB = DataLoaderWrapper(_DataLoader(train_setB))

    # # Display random binary masks of eyes
    # tA, _, bmA = train_batchA.get_next_batch()
    # tB, _, bmB = train_batchB.get_next_batch()
    # img=showG_eyes(tA, tB, bmA, bmB, batchSize)
    # print(img.shape)
    # plt.figure(figsize=(16,8))
    # plt.imshow(img)
    # plt.show()

    '''Start Training'''
    t0 = time.time()
    gen_iterations = 0
    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    errGAs = {}
    errGBs = {}
    # Dictionaries are ordered in Python 3.6
    for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
        errGAs[k] = 0
        errGBs[k] = 0

    with tqdm(total=TOTAL_ITERS, desc='Training') as pbar:
        while gen_iterations <= TOTAL_ITERS:
            # Train dicriminators for one batch
            data_A = train_batchA.get_next_batch()
            data_B = train_batchB.get_next_batch()
            errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
            errDA_sum += errDA[0]
            errDB_sum += errDB[0]

            # Train generators for one batch
            data_A = train_batchA.get_next_batch()
            data_B = train_batchB.get_next_batch()
            errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
            errGA_sum += errGA[0]
            errGB_sum += errGB[0]
            for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
                errGAs[k] += errGA[i]
                errGBs[k] += errGB[i]
            gen_iterations += 1

            # Visualization
            if gen_iterations % display_iters == 0:
                # Display loss information
                # show_loss_config(loss_config)
                print("----------")
                print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
                      % (gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters,
                         errGA_sum / display_iters, errGB_sum / display_iters, time.time() - t0))
                print("----------")
                print("Generator loss details:")
                print(f'[Adversarial loss]')
                print(f'GA: {errGAs["adv"]/display_iters:.4f} GB: {errGBs["adv"]/display_iters:.4f}')
                print(f'[Reconstruction loss]')
                print(f'GA: {errGAs["recon"]/display_iters:.4f} GB: {errGBs["recon"]/display_iters:.4f}')
                print(f'[Edge loss]')
                print(f'GA: {errGAs["edge"]/display_iters:.4f} GB: {errGBs["edge"]/display_iters:.4f}')
                if loss_config['use_PL']:
                    print(f'[Perceptual loss]')
                    try:
                        print(f'GA: {errGAs["pl"][0]/display_iters:.4f} GB: {errGBs["pl"][0]/display_iters:.4f}')
                    except:
                        print(f'GA: {errGAs["pl"]/display_iters:.4f} GB: {errGBs["pl"]/display_iters:.4f}')
                print("----------")
                
                # Display and save images
                n_batch=4
                dataA=[train_batchA.get_next_batch()[:2] for i in range(n_batch)]
                dataB=[train_batchB.get_next_batch()[:2] for i in range(n_batch)]
                wA=np.concatenate([i[0] for i in dataA])
                tA=np.concatenate([i[1] for i in dataA])
                wB=np.concatenate([i[0] for i in dataB])
                tB=np.concatenate([i[1] for i in dataB])
                # print("Transformed (masked) results:")
                img = showG(tA, tB, model.path_A, model.path_B, batchSize*n_batch)
                plt.imsave(os.path.join(samples_dir, 'result_%05d.jpg' % gen_iterations), img)
                # print("Masks:")
                img = showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize*n_batch)
                plt.imsave(os.path.join(samples_dir, 'mask_%05d.jpg' % gen_iterations), img)
                # print("Reconstruction results:")
                img = showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize*n_batch)
                plt.imsave(os.path.join(samples_dir, 'reconstruction_%05d.jpg' % gen_iterations), img)

                # Reset statistic
                errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
                for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
                    errGAs[k] = 0
                    errGBs[k] = 0

                # Save models
                save_dir = os.path.join(models_dir, '%05d' % gen_iterations)
                os.makedirs(save_dir, exist_ok=True)
                model.save_weights(path=save_dir)

            # Backup models
            if gen_iterations % backup_iters == 0:
                bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
                Path(bkup_dir).mkdir(parents=True, exist_ok=True)
                model.save_weights(path=bkup_dir)

            pbar.update()
