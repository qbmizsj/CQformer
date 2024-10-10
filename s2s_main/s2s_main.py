import os
import argparse
import logging
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from Data import uniDataset_2D, Abdomen1

from networks.model import S2S_TransNet, dice_coefficient, CONFIGS
from monai.inferers import sliding_window_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_crop_2d(volume, label, crop_size=(64, 64)):
    volume_shape = volume.shape 
    crop_height, crop_width = crop_size

    # Calculate the valid ranges for cropping along each axis
    max_crop_height = volume_shape[2] - crop_height
    max_crop_width = volume_shape[3] - crop_width
    
    # Generate random starting points along each axis
    start_height = torch.randint(0, max_crop_height + 1, (1,)).item()
    start_width = torch.randint(0, max_crop_width + 1, (1,)).item()
    
    # Crop the sub-volume using the starting points and crop size
    cropped_sub_volume = volume[:, :, start_height:start_height + crop_height,
                                start_width:start_width + crop_width, :]
    cropped_sub_label = label[:, :, start_height:start_height + crop_height,
                                start_width:start_width + crop_width, :]

    return cropped_sub_volume, cropped_sub_label


def run(args):
    path = args.result_path 
    if not os.path.exists(path):
        os.mkdir(path)
        print("make the dir")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',  filename=path + '/logging.log', filemode='a')

    logging.info(args)
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    setup_seed(args.seed)

    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.res_channel = args.in_channel
    config_vit.channel = args.hidden_size
    config_vit.original = True
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size))

    if args.name == 'Abdomen1':
        train_dataset = Abdomen1(args.root, args.seed, 'train')
        val_dataset = Abdomen1(args.root, args.seed, 'test')
    else:
        train_dataset = uniDataset_2D(args, args.name, 'train', args.seed)
        val_dataset = uniDataset_2D(args, args.name, 'test', args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle = False)

    model = S2S_TransNet(config_vit, img_size=args.img_size, num_classes=args.num_classes).to(device)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.model_lr, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)

    max_tr_dice = 0
    max_val_dice = 0

    with torch.autograd.set_detect_anomaly(True):

        for epoch in range(args.epochs):
            model.train()
            seg_losses = AverageMeter()
            tr_dice = 0
            each_tr_dice = np.zeros((1, args.num_classes))

            training_process = tqdm(train_loader, desc='training')
            for _, batch in enumerate(training_process):
                imgs, labels = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                imgs, labels = random_crop_2d(imgs, labels, args.img_size)

                output = model(imgs, labels, 'train')
                outputs = output['outputs']
                losses = output['seg_loss']
                seg_losses.update(output['seg_loss'].item(), imgs.size(0))

                tr_dice += dice_coefficient(outputs, labels)
                    
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            fin_tr_dice = tr_dice / len(train_loader)
            if args.num_classes > 1:
                each_tr_dice_avg = each_tr_dice / len(train_loader)

            if fin_tr_dice > max_tr_dice:
                best_train_epoch = epoch + 1
                max_tr_dice = fin_tr_dice
                states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
                torch.save(states, path + '/batch_train_states.pth')
   
            val_dice = 0
            each_val_dice = np.zeros((1, args.num_classes))
            
            model.eval()
            validating_process = tqdm(val_loader, desc='validating')
            with torch.no_grad():
                val_images = None
                for _, batch in enumerate(validating_process):
                    imgs, labels = batch
                    val_images = imgs.cuda()
                    labels = labels.cuda()
                    outputs = sliding_window_inference(val_images, args.size, args.batch_size, model)

                    val_dice += dice_coefficient(outputs, labels)
                fin_val_dice = val_dice / len(val_loader)
                if args.num_classes > 1:
                    each_val_dice_avg = each_val_dice / len(val_loader)

                if fin_val_dice > max_val_dice:
                    best_val_epoch = epoch + 1
                    max_val_dice = fin_val_dice
                    states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, path + '/batch_val_states.pth')

            if epoch % 20 == 0:
                states = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }
                torch.save(states, path + '/e{}.pth'.format(epoch))
                
            logging.info(
                'Epoch {}, Model seg Loss: {}'.format(epoch + 1, seg_losses.avg))

            logging.info(
                'Model  dice: {}'.format(fin_tr_dice))

            logging.info(
                'val  dice: {}'.format(fin_val_dice))

    logging.info('FinishedTraining!')
    logging.info('BestTrainEpoch: {}, BestValEpoch: {}'.format(best_train_epoch, best_val_epoch))


def arg_parse():
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument('--name', type=str, default='Abdomen1', choices=['Heart', 'Prostate', 'Hippocampus', 'Abdomen1', 'COVID_2019', 'Barts_2018', 'iSeg-2017-Training'],
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='',
                        help='source path')
    parser.add_argument('--model_lr', type=float, default=0.01,
                        help='model Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--group', type=int, default=8,
                        help='depth')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Train Epochs') 
    parser.add_argument('--in_channel', type=int, default=1,
                        help='number of img modality')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='number of label class = 1')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='dim of hidden state')
    parser.add_argument('--img_size', type=tuple,
                        default=(128, 128), help='input patch size of network input')
    parser.add_argument('--size', type=tuple,
                        default=(128, 128, 8), help='input image size of network input')
    parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='type of optimizer')
    parser.add_argument('--result_path', type=str, default='',
                        help='path to save')
    parser.add_argument('--seed', type=int, default=44)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)