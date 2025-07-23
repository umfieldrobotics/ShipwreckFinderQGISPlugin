import argparse
import copy
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torchmetrics
import torchvision.transforms.functional as TF
import torchvision
import wandb

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as T
from tqdm import tqdm

from config import config
from config import update_config
import models

###############################################################################################################
# HRNET BASELINE FOR NOAA OCEAN EXPLORATION THUNDER BAY NATIONAL MARINE SANCTUARY SHIPWRECK DETECTION PROJECT #
# Anja Sheppard, 2023                                                                                         #
###############################################################################################################

CUDA = 'cuda:0'

# train sites
train_sites = ['NearShore',
               'EBAllen',
               'Egyptian',
               'Flint',
               'Grecian',
               'Hanna',
               'Heart_Failure',
               'Montana',
               'Pewabic',
               'Rend',
               'Scott',
               'Wilson']

# test sites
per_site_metrics = {'iou_ship': 0,
                      'iou_nonship': 0,
                      'miou': 0,
                      'f1': 0,
                      'tpr': 0,
                      'tnr': 0,
                      'fpr': 0,
                      'fnr': 0}

site_confusions = {'Reef': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Chicken': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Corsair': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Corsican': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Gilbert': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Haltiner': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Lucy': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Monohansett': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Monrovia': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Shamrock': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Thew': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0},
                   'Viator': {'cf_matrix': np.zeros((2, 2)), 'metrics': per_site_metrics.copy(), 'num_images': 0}}


# wandb init
wandb.init(
    # set the wandb project where this run will be logged
    entity='anja-sheppard',
    project='RAM_experiments',
    group='hrnet',
    # track hyperparameters and run metadata
    config={
    'resolution': 512,
    'epochs': 25,
    'batch_size': 4,
    'learning_rate': 1e-5,
    'is_frozen': False,
    'model_filename': ''
    }
)

class TBNMS_Dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, target_transform=None, mode=None, train_files=None, val_files=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.ext = mode
        if mode == 'train':
          self.img_labels = [os.path.join(self.img_dir + '/train', train_file) for train_file in train_files]
          self.img_labels.sort()
          self.labels = [os.path.join(self.lbl_dir + '/train', train_file) for train_file in train_files]
          self.labels.sort()
        elif mode == 'val':
          self.img_labels = [os.path.join(self.img_dir + '/train', train_file) for train_file in val_files]
          self.img_labels.sort()
          self.labels = [os.path.join(self.lbl_dir + '/train', train_file) for train_file in val_files]
          self.labels.sort()
        elif mode == 'test':
          self.img_labels = list(os.listdir(os.path.join(self.img_dir, self.ext)))
          self.img_labels.sort()
          self.labels = list(os.listdir(os.path.join(self.lbl_dir, self.ext)))
          self.labels.sort()
        else:
          print('mode not valid. exiting.')
          exit()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.ext, self.img_labels[idx])
        name = os.path.splitext(self.img_labels[idx])[0]
        lbl_path = os.path.join(self.lbl_dir,self.ext, self.labels[idx])
        image = read_image(img_path)
        lbl = read_image(lbl_path)[0] # we only want the first channel
        lbl = lbl.type(torch.LongTensor)
        
        if self.transform:
            image = self.transform(image/255.)
        if self.target_transform:
            lbl = self.target_transform(lbl)

        sample = {'image': image, 'gt': lbl, 'name': name}
       
        
        # image = transforms.ColorJitter()(image)
        if self.ext == 'train': 
            if random.random() > 0.5:
                image = TF.hflip(image)
                lbl = TF.hflip(lbl)
            if random.random() > 0.5:
                image = TF.vflip(image)
                lbl = TF.vflip(lbl)
            # param = T.RandomRotation.get_params([-180, 180])
            # image = TF.rotate(image, param, interpolation=InterpolationMode.NEAREST)
        return sample

###############
# TRAIN MODEL #
###############

def train_model(model, criterion, dataloaders, optimizer, metrics, num_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = -1
    # Use gpu if available
    device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data
            epoch_loss = 0.0
            epoch_iou = 0

            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device).float()
                gt = sample['gt'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(dataloaders[phase].dataset.ext=='train'):
                    out = model(inputs)
                    outputs = torch.nn.functional.interpolate(out[0], size=sample['image'].shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(outputs, gt)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    elif phase == 'val':
                        # if during val phase, calculate the iou per image and then average later
                        softmax = torch.nn.Softmax(dim=1)
                        prediction = softmax(outputs)
                        prediction = torch.argmax(prediction, dim=1)

                        cf_matrix = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)(prediction.cpu(), gt.cpu()).numpy()
                        tp = cf_matrix[1,1]
                        fp = cf_matrix[0,1]
                        fn = cf_matrix[1,0]
                        if tp + fp + fn != 0:
                          epoch_iou += (tp / (tp + fp + fn)) * inputs.size(0)


                    epoch_loss += loss.item() * inputs.size(0)

            epoch_loss /= len(dataloaders[phase].dataset)
            epoch_iou /= len(dataloaders[phase].dataset) # avg ious per image in epoch
                    
            # wandb logging
            if phase == 'train':
                wandb.log({'train_loss': epoch_loss}) # wandb logging
            else:
                wandb.log({'val_loss': epoch_loss, \
                           'val_iou': epoch_iou}) # wandb logging

        if phase == 'val' and epoch_iou > best_iou:
            best_iou = epoch_iou
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Highest IoU: {:4f}'.format(best_iou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

###############################
# TEST MODEL ON SAMPLE IMAGES #
###############################

def test_model(model_dir, model_file_name, test_img_path, test_gt_path, transform_norm, transform_unnorm, metrics):
  eval_image_indices = [10, 53, 129, 1718, 1720]

  model_path = model_dir + model_file_name

  if not os.path.exists(model_path):
     file_list = glob.glob(model_dir + '/*')
     model_path = max(file_list, key=os.path.getctime) # if no model file is found with the timestamp of the run, find the most recent model weights file

  # Load the trained model
  if torch.cuda.is_available():
    model = torch.load(model_path)
  else:
    model = torch.load(model_path, map_location=torch.device('cpu'))
      
  device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

  # Set the model to evaluate mode
  model.eval()

  # Test model on each of the 5 images in test dir
  test_img_list = os.listdir(test_img_path)
  for index in eval_image_indices:
    # Read a sample image and ground truth
    test_image = read_image(test_img_path + test_img_list[index])
    test_gt = read_image(test_gt_path + test_img_list[index]) # same name for image and gt, diff dirs
    test_gt = test_gt.type(torch.LongTensor)
    
    test_image = transform_norm(test_image / 255.)

    # Evaluate model
    start_time = time.time()
    with torch.no_grad():
      if torch.cuda.is_available():
        a = model(test_image.unsqueeze(0).to(device))
      else:
        a = model(test_gt.to(device))
    
    print("--- %s seconds ---" % (time.time() - start_time))

    # Resize because HRNet outputs image 4 times smaller
    prediction = torch.nn.functional.interpolate(a[0].cpu().detach(), size=test_image.shape[-2:], mode='bilinear', align_corners=False)
    
    # Apply softmax to get probability output. Each slice of the 2 x H x W tensor has the probabilities for each pixel of being that class.
    softmax = torch.nn.Softmax(dim=1)
    prediction = softmax(prediction)
    prediction = torch.argmax(prediction, dim=1)

    # un normalize
    test_image = transform_unnorm(test_image) * 255

    # Overlay ground truth on test image
    out_true = torchvision.utils.draw_segmentation_masks(test_image.type(torch.uint8), test_gt.type(torch.BoolTensor), alpha=0.5)

    # Overlay predicted label on test image
    out_pred = torchvision.utils.draw_segmentation_masks(test_image.type(torch.uint8), prediction.repeat(3, 1, 1).type(torch.BoolTensor), alpha=0.5)

    numpy_out_true = out_true.numpy().transpose(1, 2, 0)
    numpy_out_pred = out_pred.numpy().transpose(1, 2, 0)

    # Calculate metrics
    metric_calc = {}
    for metric in metrics:
        metric_calc[metric] = metrics[metric](prediction[0], test_gt[0])

    # Log to wandb
    wandb_out_true = wandb.Image(numpy_out_true, caption="Ground Truth")
    wandb_out_pred = wandb.Image(numpy_out_pred, caption="Prediction")
    wandb.log({test_img_list[index][:-4]: [wandb_out_true, wandb_out_pred], 'iou': metric_calc['iou'], 'f1_score': metric_calc['f1_score']})

###########################
# CALCULATE MODEL METRICS #
###########################

def calc_confusion_matrix(dataloader, model_dir, model_file_name):

  model_path = model_dir + model_file_name

  if not os.path.exists(model_path):
     file_list = glob.glob(model_dir + '/*')
     model_path = max(file_list, key=os.path.getctime) # if no model file is found with the timestamp of the run, find the most recent model weights file

  # Load the trained model
  if torch.cuda.is_available():
    model = torch.load(model_path)
  else:
    model = torch.load(model_path, map_location=torch.device('cpu'))
      
  device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

  # Set the model to evaluate mode
  model.eval()

  # loop through all images in test set
  for test_sample in tqdm(dataloader):
    # get site cumulative cf matrix
    cf_matrix = None
    for site in site_confusions:
      if site in test_sample['name'][0] or (site == 'Monohansett' and 'Davidson' in test_sample['name'][0]):
        cf_matrix = site_confusions[site]['cf_matrix']
        site_confusions[site]['num_images'] += 1

    if cf_matrix is None:
      print('something is wrong and your test sample doesnt have a corresponding site in the dict. exiting.')
      print(test_sample['name'])
      exit()    

    # Evaluate model
    input = test_sample['image'].to(device).float()

    with torch.no_grad():
      a = model(input)

    # Resize because HRNet outputs image 4 times smaller
    prediction = torch.nn.functional.interpolate(a[0].cpu().detach(), size=test_sample['image'].shape[-2:], mode='bilinear', align_corners=False)
        
    # Apply softmax to get probability output. Each slice of the 2 x H x W tensor has the probabilities for each pixel of being that class.
    softmax = torch.nn.Softmax(dim=1)
    prediction = softmax(prediction)
    prediction = torch.argmax(prediction, dim=1)

    gt = test_sample['gt']

    prediction = prediction.flatten()
    gt = gt.flatten()

    cf_matrix += torchmetrics.ConfusionMatrix(task='binary', num_classes=2)(prediction, gt).numpy()

  # Build confusion matrix
  avg_metrics = {'avg_iou_ship': 0,
                     'avg_iou_nonship': 0,
                     'avg_miou': 0,
                     'avg_f1': 0,
                     'avg_tpr': 0,
                     'avg_tnr': 0,
                     'avg_fpr': 0,
                     'avg_fnr': 0}

  for site in site_confusions:
    site_confusions[site]['cf_matrix'] /= site_confusions[site]['num_images']
    tp = site_confusions[site]['cf_matrix'][1, 1]
    tn = site_confusions[site]['cf_matrix'][0, 0]
    fp = site_confusions[site]['cf_matrix'][0, 1]
    fn = site_confusions[site]['cf_matrix'][1, 0]
    site_confusions[site]['metrics']['iou_ship'] = tp / (tp + fp + fn)
    site_confusions[site]['metrics']['iou_nonship'] = tn / (tn + fn + fp)
    site_confusions[site]['metrics']['miou'] = (site_confusions[site]['metrics']['iou_ship'] + site_confusions[site]['metrics']['iou_nonship']) / 2
    site_confusions[site]['metrics']['tpr'] = tp / (tp + fn)
    site_confusions[site]['metrics']['tnr'] = tn / (tn + fp)
    site_confusions[site]['metrics']['fpr'] = fp / (fp + tn)
    site_confusions[site]['metrics']['fnr'] = fn / (fn + tp)
    if tp != 0 and fp != 0:
      site_precision = tp / (tp + fp)
      site_recall = tp / (tp + fn)
      site_confusions[site]['metrics']['f1'] = 2 * (site_precision * site_recall) / (site_precision + site_recall)
    else:
      site_confusions[site]['metrics']['f1'] = 0

    avg_metrics['avg_iou_ship'] += site_confusions[site]['metrics']['iou_ship']
    avg_metrics['avg_iou_nonship'] += site_confusions[site]['metrics']['iou_nonship']
    avg_metrics['avg_miou'] += site_confusions[site]['metrics']['miou']
    avg_metrics['avg_f1'] += site_confusions[site]['metrics']['f1']
    avg_metrics['avg_tpr'] += site_confusions[site]['metrics']['tpr']
    avg_metrics['avg_tnr'] += site_confusions[site]['metrics']['tnr']
    avg_metrics['avg_fpr'] += site_confusions[site]['metrics']['fpr']
    avg_metrics['avg_fnr'] += site_confusions[site]['metrics']['fnr']

    wandb.log({site + '_metrics': site_confusions[site]['metrics']})

  for metric in avg_metrics:
     avg_metrics[metric] = avg_metrics[metric] / len(site_confusions.keys())

  wandb.log(avg_metrics)

def freeze_encoder(model):
    # Freeze the backbone of the model, which is instantiated as model.backbone
    #for param in model.backbone.parameters():
    #    param.requires_grad = False

    # Freeze the encoder of the model, which is instantiated as model.context(1,2,3,4,5)
    context_layers = [
        model.conv1
    ]

    for context_layer in context_layers:
        for param in context_layer.parameters():
            param.requires_grad = False

    return model

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config.update_config(config, args)

    return args

if __name__=="__main__":
    torch.cuda.empty_cache()

    # parameters
    RESOLUTION = 512
    BATCH_SIZE = 16 # must be > 1 for HRnet
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    FROZEN = False
    IMAGE_FOLDER = '/mnt/ws-frb/users/obagoren/datasets/square_images_' + str(RESOLUTION) + '/'
    GT_FOLDER = '/mnt/ws-frb/users/obagoren/datasets/square_labels_' + str(RESOLUTION) + '/'

    # hrnet parameters
    a = argparse.Namespace(cfg='../configs/hrnet_config.yaml',
                                   local_rank=-1,
                                   opts=[],
                                   seed=304)

    update_config(config, a)
    
    assert BATCH_SIZE > 1

    # wandb setup
    wandb.config.update({"resolution": RESOLUTION}, allow_val_change=True)
    wandb.config.update({"batch_size": BATCH_SIZE}, allow_val_change=True)
    wandb.config.update({"epochs": EPOCHS}, allow_val_change=True)
    wandb.config.update({"learning_rate": LEARNING_RATE}, allow_val_change=True)
    wandb.config.update({"is_frozen": FROZEN}, allow_val_change=True)

    # Create image transforms
    train_mean = np.array([0.2746905462571075, 0.2746905462571075, 0.2746905462571075])
    train_std = np.array([0.2318517795744138, 0.2318517795744138, 0.2318517795744138]) # from mean_and_std_of_dataset.py
    transform_norm = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(train_mean, train_std)
    ])

    transform_unnorm = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0., 0., 0.],  1 / train_std),
        torchvision.transforms.Normalize(-train_mean, [1., 1., 1.])
    ])

    # Create train/val split
    # first, randomly split
    random.shuffle(train_sites)
    train_set_sites = train_sites[:9]
    val_set_sites = train_sites[9:]
    train_set_files = []
    val_set_files = []
    # then loop through each file in the train folder and split into train or val based off file name
    train_set_files = os.listdir(os.path.join(IMAGE_FOLDER, "train"))
    for file in os.listdir(os.path.join(IMAGE_FOLDER, "train")):
        for val_site in val_set_sites:
            if val_site in file:
                val_set_files.append(file)
                train_set_files.remove(file)

    # Create the dataloaders
    train_dataset = TBNMS_Dataset(img_dir=IMAGE_FOLDER, lbl_dir=GT_FOLDER, transform=transform_norm, mode='train', train_files=train_set_files)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataset = TBNMS_Dataset(img_dir=IMAGE_FOLDER, lbl_dir=GT_FOLDER, transform=transform_norm, mode='val', val_files=val_set_files)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TBNMS_Dataset(img_dir=IMAGE_FOLDER, lbl_dir=GT_FOLDER, transform=transform_norm, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')

    # Create the model
    model = models.seg_hrnet_ocr.get_seg_model(config).to(device)

    # Freeze encoder
    if FROZEN:
      freeze_encoder(model)

    # model weights save path
    bpath = os.path.expanduser('~') + '/RAM_experiments/hrnet/out/'
    model_filename = 'weights_150_16_1688954443.1075728.pt' #'weights_' + str(EPOCHS) + '_' + str(BATCH_SIZE) + '_' + str(time.time()) + '.pt'

    wandb.config.update({'model_filename': model_filename}, allow_val_change=True)

    # Specify the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Specify the evaluation metrics 
    metrics = {'iou': torchmetrics.JaccardIndex(task='binary'), 'f1_score': torchmetrics.F1Score(task='binary')}

    # # train model
    # print('Training model...')
    # trained_model = train_model(model, criterion, dataloaders, optimizer, metrics=metrics, num_epochs=EPOCHS)

    # torch.save(trained_model, os.path.join(bpath, model_filename))

    # print('Model saved to ' + os.path.join(bpath, model_filename))

    # # test
    # test_model(bpath, model_filename, IMAGE_FOLDER + 'test/', GT_FOLDER + 'test/', transform_norm, transform_unnorm, metrics)

    # calculate confusion matrix and iou per site
    print('Evaluating model...')
    eval_dataloader = DataLoader(test_dataset)
    calc_confusion_matrix(eval_dataloader, bpath, model_filename)