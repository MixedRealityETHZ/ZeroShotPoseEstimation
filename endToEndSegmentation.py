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

from torch.utils.data import DataLoader


def main():
    extract_video = False
    plot = True
    on_GPU = True
    video_path = "./tiger.mp4"
    images_folder = "/images"
    imlist_folder = "/lists"
    file_txt = "/images.txt"
    model_name = "dino_vits16"
    default_data_path = "data/object-segmentation/custom_dataset"
    feature_relative_path = "/features/dino_vits16"
    eigs_relative_path = "/eigs/laplacian_dino_vits16"
    sr_segmentations = "/single_region_segmentation/patches/filtered"
    full_segmentations = "/single_region_segmentation/crf/laplacian_dino_vits16"

    images_root = default_data_path + images_folder
    imlist_root = default_data_path + imlist_folder
    images_list = imlist_root + file_txt
    feature_dir = default_data_path + feature_relative_path
    eigs_dir = default_data_path + eigs_relative_path
    sr_dir = default_data_path + sr_segmentations
    full_seg_dir = default_data_path + full_segmentations

    # Try with first example
    # Before running this example, you should already have the list of images and the images per frame from the video
    # You can get this by running the normal DSM applied to the video recorded by Roberto

    if extract_video:
        images_extracting.extract_images(video_path, images_root)
        images_extracting.write_images_txt(imlist_root, file_txt, images_root)

    # list files in img directory, this is the txt file containing the name of all images
    filenames = Path(images_list).read_text().splitlines()

    # Second step, all the functions without creating folders
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

    # here we are creating sub plots
    for k, (images, files, indices) in enumerate(tqdm(dataloader)):

        feature_dict = extract.extract_features(
            model=model,
            patch_size=patch_size,
            num_heads=num_heads,
            accelerator=accelerator,
            feat_out=feat_out,
            images=images,
            files=files,
            indices=indices,
        )

        eigs_dict = extract._extract_eig(K=4, data_dict=feature_dict, on_gpu=on_GPU)

        # Segmentation
        segmap = extract.extract_single_region_segmentations(
            feature_dict=feature_dict,
            eigs_dict=eigs_dict,
        )

        bbox = extract.extract_bboxes(
            feature_dict=feature_dict,
            segmap=segmap,
        )

        if plot:
            # Bounding boxes
            limits = bbox["bboxes_original_resolution"][0]
            image_PIL = Image.open(images_root + "/" + filenames[k])
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


if __name__ == "__main__":
    main()
