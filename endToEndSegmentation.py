import images_extracting
from extract import extract
from pathlib import Path
import os
from extract import extract_utils as utils
import torch
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
import time

# TODO: make parameters globally setted by user
# TODO: use pathlib.Path for path setting

video_path = "./IMG_6943.MOV"
images_folder = "/images"
imlist_folder = "/lists"
file_txt = "/images.txt"
model_name = "dino_vits16"
default_data_path = "data/object-segmentation/custom_dataset"
feature_relative_path = "/features/dino_vits16"
eigs_relative_path = "/eigs/laplacian_dino_vits16"
#sr_segmentations = "/single_region_segmentation/patches/laplacian_dino_vits16"
sr_segmentations = "/single_region_segmentation/patches/filtered"
full_segmentations = "/single_region_segmentation/crf/laplacian_dino_vits16"

images_root = default_data_path + images_folder
imlist_root = default_data_path + imlist_folder
images_list = imlist_root + file_txt
feature_dir = default_data_path + feature_relative_path
eigs_dir = default_data_path + eigs_relative_path
sr_dir = default_data_path + sr_segmentations
full_seg_dir = default_data_path + full_segmentations

#Try with first example
#Before running this example, you should already have the list of images and the images per frame from the video
#You can get this by running the normal DSM applied to the video recorded by Roberto

images_extracting.extract_images(video_path, images_root)
images_extracting.write_images_txt(imlist_root, file_txt, images_root)

# list files in img directory, this is the txt file containing the name of all images
filenames = Path(images_list).read_text().splitlines()

#Second step, all the functions without creating folders
#Load the model 
model_name = model_name.lower()
model, val_transform, patch_size, num_heads = utils.get_model(model_name)

# here we are creating sub plots
for k in range(len(filenames)):
    
    dataset = utils.ImagesDataset(
        filenames=[filenames[k]], images_root=images_root, transform=val_transform
    )

    #Load image with PIL, maybe we can optimize this part
    image_PIL = Image.open(images_root +'/'+filenames[k])
    
    start = time.time()
    feature_dict = extract.extract_features(
        output_dir=feature_dir,
        batch_size=1,
        num_workers=0,
        model_name = model_name,
        model= model,
        patch_size = patch_size,
        num_heads = num_heads,
        dataset = dataset,
    )

    eigs_dict = extract.extract_eigs(
        which_matrix="laplacian",
        K=2,
        data_dict = feature_dict,
        image_file = dataset,
    )

    #Segmentation
    segmap = extract.extract_single_region_segmentations(
        feature_dict = feature_dict,
        eigs_dict = eigs_dict,
    )

    segmented_im = extract.extract_crf_segmentations(
        output_dir=full_seg_dir,
        num_classes=2,
        downsample_factor=20,
        segmap = segmap,
        image_file=image_PIL,
        id=k,
    )
    #Bounding boxes 
    bbox = extract.extract_bboxes(
        feature_dict = feature_dict,
        segmap = segmap,
    )

    limits = bbox['bboxes_original_resolution'][0]


    #plotting just to see the results
    plt.imshow(image_PIL, alpha=0.9)
    plt.gca().add_patch(Rectangle((limits[0],limits[1]),limits[2]-limits[0],limits[3]-limits[1],
                    edgecolor='red',
                    facecolor='none',
                    lw=4))
    plt.pause(0.0001)
    plt.clf()
    print(f"Finished in {time.time() - start:.1f}s")

plt.show()


