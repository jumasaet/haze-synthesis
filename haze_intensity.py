from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
from tqdm import tqdm

import torch
from torchvision import transforms, datasets
import networks

import warnings
# Suprimir warnings específicos de torchvision
warnings.filterwarnings("ignore", 
                        message="Using 'weights' as positional parameter*",
                        category=UserWarning)
warnings.filterwarnings("ignore", 
                        message="Arguments other than a weight enum*",
                        category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', default="images")
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],
                        default="mono+stereo_1024x320")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument('--output_image_path', type=str,
                        help='path to folder of output images', default="outputs")
    parser.add_argument('--beta_values', type=float, nargs='+',
                        help='list of beta values for haze generation', default=[0.5, 0.9, 2.0])
    parser.add_argument('--airlight', type=float,
                        help='atmospheric light', default=200.0)
    parser.add_argument("--device", 
                        type=int,
                        default=0,
                        help='ID of the GPU to use (default: 0)')
    parser.add_argument('--batch', type=int,
                        help='batch size for processing images', default=4)

    return parser.parse_args()


def gen_haze(clean_img, depth_img, beta=1.0, A=150):
    depth_img_3c = np.zeros_like(clean_img)
    depth_img_3c[:,:,0] = depth_img
    depth_img_3c[:,:,1] = depth_img
    depth_img_3c[:,:,2] = depth_img

    norm_depth_img = depth_img_3c/255
    trans = np.exp(-norm_depth_img*beta)

    hazy = clean_img*trans + A*(1-trans)
    hazy = np.array(hazy, dtype=np.uint8)

    return hazy


def process_batch(images, encoder, depth_decoder, device, feed_width, feed_height, args):
    """Process a batch of images"""
    batch_tensors = []
    original_sizes = []
    clean_images = []
    
    # Preprocess batch
    for img_path in images:
        input_image = pil.open(img_path).convert('RGB')
        clean_images.append(np.array(input_image.copy()))
        original_width, original_height = input_image.size
        original_sizes.append((original_height, original_width))
        
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image)
        batch_tensors.append(input_image)
    
    # Create batch tensor
    batch_tensor = torch.stack(batch_tensors).to(device)
    
    # Process batch through model
    with torch.no_grad():
        features = encoder(batch_tensor)
        outputs = depth_decoder(features)
    
    # Process each image in batch
    results = []
    for i, (img_path, clean_img, original_size) in enumerate(zip(images, clean_images, original_sizes)):
        disp = outputs[("disp", 0)][i:i+1]
        disp_resized = torch.nn.functional.interpolate(
            disp, original_size, mode="bilinear", align_corners=False)
        
        # EXTRACT DEPTH IMAGE
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        gray_colormapped_im = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2GRAY)
        inv_gray_colormapped_im = 255 - gray_colormapped_im
        
        results.append({
            'path': img_path,
            'clean_img': clean_img,
            'depth_map': inv_gray_colormapped_im
        })
    
    return results


def test_simple(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if args.device is not None:
        use_cuda = torch.cuda.is_available() and not args.no_cuda
        device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    else:
        use_cuda = False

    # Verificar si la GPU específica está disponible
    if use_cuda:
        if args.device < torch.cuda.device_count():
            print(f"\nUsing GPU: {torch.cuda.get_device_name(args.device)} (ID: {args.device})")
        else:
            print(f"Warning: GPU ID {args.device} not available. Using CPU instead.")
            device = torch.device("cpu")
            use_cuda = False
    else:
        print("Using device: CPU")
    
    print(f"Batch size: {args.batch}")
    print(f"Beta values: {args.beta_values}")
    print(f"Airlight: {args.airlight}\n")
        
    # download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # EXTRACT THE HEIGHT AND WIDTH OF IMAGE THAT THIS MODEL WAS TRAINED WITH
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Searching folder for images with any extension
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        paths = []
        for ext in image_extensions:
            paths.extend(glob.glob(os.path.join(args.image_path, ext)))
            paths.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
        # Filter out disparity images
        paths = [p for p in paths if not any(p.endswith(ext) for ext in ['_disp.jpg', '_disp.png'])]
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Found {:d} images to process\n".format(len(paths)))

    # CHECK IF OUTPUT FOLDER EXISTS
    if not os.path.isdir(args.output_image_path):
        os.makedirs(args.output_image_path)

    output_dir = args.output_image_path

    # PROCESS IMAGES IN BATCHES
    total_images = len(paths)
    processed_count = 0
    
    # Create progress bar for batch processing
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for batch_start in range(0, total_images, args.batch):
            batch_end = min(batch_start + args.batch, total_images)
            batch_paths = paths[batch_start:batch_end]
            
            try:
                # Process the batch
                batch_results = process_batch(batch_paths, encoder, depth_decoder, device, 
                                            feed_width, feed_height, args)
                
                # Generate and save hazy images for each result
                for result in batch_results:
                    output_name = os.path.splitext(os.path.basename(result['path']))[0]
                    
                    for beta in args.beta_values:
                        hazy = gen_haze(result['clean_img'], result['depth_map'], 
                                      beta=beta, A=args.airlight)
                        
                        # Save with beta value in filename
                        beta_str = str(beta).replace('.', '_')
                        output_filename = f"{output_dir}/{output_name}_beta_{beta_str}_synt.jpg"
                        cv2.imwrite(output_filename, cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
                
                processed_count += len(batch_results)
                pbar.update(len(batch_results))
                
                # Update progress bar description
                pbar.set_postfix({
                    'Batch': f"{batch_start//args.batch + 1}/{(total_images-1)//args.batch + 1}",
                    'Beta': f"{args.beta_values}"
                })
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\nOut of memory error with batch size {args.batch}. Reducing batch size...")
                    if args.batch > 1:
                        args.batch //= 2
                        print(f"New batch size: {args.batch}")
                        # Retry current batch with smaller size
                        batch_start -= args.batch * 2  # Go back to previous batch
                    else:
                        print("Batch size already at 1, cannot reduce further.")
                        raise e
                else:
                    raise e

    print(f'\n-> Done! Processed {processed_count} images')
    print(f'-> Generated {processed_count * len(args.beta_values)} hazy images')
    print(f'-> Outputs saved in: {output_dir}\n')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)