import glob

import torch
import numpy as np
import scipy.ndimage
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tqdm.auto import tqdm
from pydicom import dcmread
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')


def load_scan(path):
    files = glob.glob(path + '/*/*/*.dcm')

    if len(files) < 1:
        return None

    slices = [dcmread(x) for x in files]
    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans, HU=True):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    if HU:
        # Convert to Hounsfield units (HU)

        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, size=128):
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))
    factor = spacing[1] / (size / image.shape[1])
    return scipy.ndimage.interpolation.zoom(image, spacing / factor)


def pad(image, size=128):
    if image.shape[0] < size:
        margin = size - image.shape[0]
        left = margin // 2
        image = np.pad(image, ((left, margin - left), (0,0), (0,0)), constant_values=-1024)

    if image.shape[0] > size:
        image = scipy.ndimage.interpolation.zoom(image, size / image.shape[0])
        margin = size - image.shape[1]
        left = margin // 2
        image = np.pad(image, ((0,0), (left, margin - left), (left, margin - left)), constant_values=-1024)

    return image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image):
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    background_label = labels[0,0,0]
    binary_image[background_label == labels] = 2

    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:
            binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1-binary_image

    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image


def plot_3d(image, threshold=-300, color_mask=None):
    p = image.transpose(2,1,0)[:,:,::-1]

    if color_mask is not None:
        color_mask = color_mask.transpose(2,1,0)[:,:,::-1]

    verts, faces, norm, val = measure.marching_cubes(p, threshold, allow_degenerate=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    if color_mask is None:
        color_mask = [0.5, 0.5, 1]
    else:
        norm = colors.Normalize(color_mask.min(), color_mask.max())
        c = cm.jet(norm(color_mask))
        temp = []
        for x in verts[faces]:
            co = x.mean(axis=0).astype(int)
            temp.append(c[co[0], co[1], co[2]])
        color_mask = np.stack(temp)

    mesh.set_facecolor(color_mask)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    return fig


def get_predictions(outputs):
    return (torch.nn.Softmax(dim=1))(outputs).argmax(axis=1)


def get_accuracy(labels, outputs):
    return accuracy_score(labels.cpu().detach().numpy(), get_predictions(outputs.cpu().detach()).numpy())


def train(epochs, device, model, criterion, optimizer, train_set, validation_set, data_fetcher, callback=None):

    for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
        hist_loss, hist_accuracy  = [], []

        model.train()

        for data in tqdm(train_set, desc='Batches', leave=False):
            inputs, labels = data_fetcher(data)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())
            hist_accuracy.append(get_accuracy(labels, outputs))

        train_loss, train_accuracy = np.mean(hist_loss), np.mean(hist_accuracy)
        hist_loss, hist_accuracy  = [], []

        with torch.no_grad():
            model.eval()

            for data in tqdm(validation_set, desc='Batches', leave=False):
                inputs, labels = data_fetcher(data)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)

                hist_loss.append(loss.item())
                hist_accuracy.append(get_accuracy(labels, outputs))

        if callback is not None:
            callback(epoch, model, [train_loss, train_accuracy], [np.mean(hist_loss), np.mean(hist_accuracy)])

    return model

def return_CAM(feature_conv, weight, idx):
    bz, nc, d, h, w = feature_conv.shape
    beforeDot =  feature_conv.reshape((nc, d*h*w))
    cam = weight[idx].dot(beforeDot)
    cam = cam.reshape(d, h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return scipy.ndimage.interpolation.zoom(cam_img, 128 / np.array(cam_img.shape))

def cam_visualize(device, dataset, idx, model):
    params = list(model.parameters())
    weight = np.squeeze(params[-2].data.cpu().detach().numpy())

    img_variable = torch.autograd.Variable(dataset[idx][0].unsqueeze(0)).to(device)
    logit = model(img_variable)

    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()

    cls = h_x.argmax().item()
    print("predicted: ", cls)

    features_blobs = model.res_blocks(img_variable)
    features_blobs1 = features_blobs.cpu().detach().numpy()
    CAM = return_CAM(features_blobs1, weight, cls)

    plot_3d(dataset[idx][1], None, CAM).show()

    return CAM