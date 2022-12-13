import images_extracting
from extract import extract
from pathlib import Path
from extract import extract_utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import time
from accelerate import Accelerator
import os
from torch.utils.data import DataLoader
import cv2
import numpy as np
from scipy import optimize
import lmfit

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def main():
    extract_video = False
    plot = False
    on_GPU = False


    '''
    video_path = "/Frames.m4v"
    images_folder = "/color_full"
    imlist_folder = "/lists"
    file_txt = "/images.txt"
    model_name = "dino_vits16"
    #default_data_path = "/Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/ZeroShotPoseEstimation-main/tiger"
    default_data_path = "/Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/Pre-trained/test_coffee/test_coffee-test"

    images_root = default_data_path + images_folder  #Used
    imlist_root = default_data_path + imlist_folder  #Used
    images_list = imlist_root + file_txt
    video_path = default_data_path + video_path
    '''

    video_path = "/Frames.m4v"
    images_folder = "/color_full"
    imlist_folder = "/lists"
    file_txt = "/images.txt"
    model_name = "dino_vits16"
    detection = "/detection"
    #default_data_path = "/Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/OnePose/data/onepose_datasets/sample_data/0501-matchafranzzi-box/matchafranzzi-3"
    default_data_path = '/Users/diego/Desktop/Escritorio_MacBook_Pro_de_Diego/ETH/Third_Semester/Mixed_Reality/OnePose/data/onepose_datasets/val_data/0616-hmbb-others/hmbb-1'
    images_root = default_data_path + images_folder  #Used
    imlist_root = default_data_path + imlist_folder  #Used
    images_list = imlist_root + file_txt
    video_path = default_data_path + video_path
    detection_path = default_data_path + detection

    # Try with first example
    # Before running this example, you should already have the list of images and the images per frame from the video
    # You can get this by running the normal DSM applied to the video recorded by Roberto

    if extract_video:
        print("Extracting images")
        images_extracting.extract_images(video_path, images_root)
        images_extracting.write_images_txt(imlist_root, file_txt, images_root)

    # list files in img directory, this is the txt file containing the name of all images
    filenames = Path(images_list).read_text().splitlines()
    #filenames = os.listdir(images_root)
    #filenames = filenames[207:]

    # Load the model
    model, val_transform, patch_size, num_heads = utils.get_model(model_name)

    dataset = utils.ImagesDataset(
        filenames=filenames, images_root=images_root, transform=val_transform
    )

    dataloader = DataLoader(dataset, batch_size=1)
    if on_GPU:
        accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    else:
        accelerator = Accelerator(mixed_precision="no", cpu=True)
    model, dataloader = accelerator.prepare(model, dataloader)
    model = model.to(accelerator.device)

    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
        hook_fn_forward_qkv
    )


    fitting_model = lmfit.models.Gaussian2dModel()
    params = 0

    for k, (images, l, m) in enumerate(tqdm(dataloader)):
        bbox = extract_bbox(
            fitting_model=fitting_model,
            params=params,
            model=model,
            patch_size=patch_size,
            num_heads=num_heads,
            accelerator=accelerator,
            feat_out=feat_out,
            images=images,
            on_GPU=on_GPU,
        )
        

        if plot:
            # Bounding boxes
            limits = bbox["bboxes_original_resolution"][0]
            limits = [lim for lim in limits]
            print(limits)
            image_PIL = Image.open(images_root + "/" + filenames[k])
            plt.figure(1)
            plt.imshow(image_PIL, alpha=0.9)
            plt.gca().add_patch(
                Rectangle(
                    (limits[0], limits[1]),
                    limits[2] - limits[0],
                    limits[3] - limits[1],
                    edgecolor="red",
                    facecolor="none",
                    lw=4,
                )
            )
            plt.show(block=False)
            plt.pause(0.0001)
            plt.clf()


def extract_bbox(fitting_model, params, model, patch_size, num_heads, accelerator, feat_out, images, on_GPU):
    feature_dict = extract.extract_features(
        model=model,
        patch_size=patch_size,
        num_heads=num_heads,
        accelerator=accelerator,
        feat_out=feat_out,
        images=images,
    )
    
    eigs_dict = extract._extract_eig(K=4, data_dict=feature_dict, on_gpu=on_GPU)
    

    segmap= extract.gaussian_fitting(
        feature_dict=feature_dict,
        eigs_dict=eigs_dict,
        fitting_model=fitting_model
    )

    bbox = extract.extract_bboxes(
        feature_dict=feature_dict,
        segmap=segmap,
        )
    if not bbox['bboxes']:
        # Segmentation
        segmap = extract.extract_single_region_segmentations(
        feature_dict=feature_dict,
        eigs_dict=eigs_dict,
        )
        bbox = extract.extract_bboxes(
        feature_dict=feature_dict,
        segmap=segmap,
        )

    return bbox


if __name__ == "__main__":
    main()
