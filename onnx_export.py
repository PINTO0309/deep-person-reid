import os
import onnx
import requests
import torch
from torchreid.utils import FeatureExtractor, SimilarityCalculator

MODELS = [
    # ['hacnn', [160,64], 'hacnn_duke_xent.pth.tar'], # invalid
    # ['hacnn', [160,64], 'hacnn_market_xent.pth.tar'], # invalid
    # ['hacnn', [160,64], 'hacnn_msmt_xent.pth.tar'], # invalid
    # ['mlfn', [256,128], 'mlfn_duke_xent.pth.tar'], # invalid
    # ['mlfn', [256,128], 'mlfn_market_xent.pth.tar'], # invalid
    # ['mlfn', [256,128], 'mlfn_msmt_xent.pth.tar'], # invalid
    # ['resnet50', [256,128], 'resnet50_duke_xent.pth.tar'], # invalid
    # ['resnet50', [256,128], 'resnet50_fc512_duke_xent.pth.tar'], # invalid
    # ['resnet50', [256,128], 'resnet50_fc512_market_xent.pth.tar'], # invalid
    # ['resnet50', [256,128], 'resnet50_market_xent.pth.tar'], # invalid
    # ['resnet50', [256,128], 'resnet50_msmt_xent.pth.tar'], # invalid

    ['mlfn', [256,128], 'mlfn-9cb5a267.pth.tar', 'cosine'],
    ['mobilenetv2_x1_0', [256,128], 'mobilenetv2_1.0-0f96a698.pth.tar', 'cosine'],
    # ['mobilenetv2_x1_4', [256,128], 'mobilenetv2_1.4-bc1cc36b.pth.tar', 'cosine'],
    ['mobilenetv2_x1_0', [256,128], 'mobilenetv2_1dot0_duke.pth.tar', 'cosine'],
    ['mobilenetv2_x1_0', [256,128], 'mobilenetv2_1dot0_market.pth.tar', 'cosine'],
    ['mobilenetv2_x1_0', [256,128], 'mobilenetv2_1dot0_msmt.pth.tar', 'cosine'],
    ['mobilenetv2_x1_4', [256,128], 'mobilenetv2_1dot4_duke.pth.tar', 'cosine'],
    ['mobilenetv2_x1_4', [256,128], 'mobilenetv2_1dot4_market.pth.tar', 'cosine'],
    ['mobilenetv2_x1_4', [256,128], 'mobilenetv2_1dot4_msmt.pth.tar', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_d_m_c.pth.tar', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_ms_d_c.pth.tar', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_ms_d_m.pth.tar', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_ms_m_c.pth.tar', 'cosine'],
    ['osnet_ain_x0_25', [256,128], 'osnet_ain_x0_25_imagenet.pyth', 'cosine'],
    ['osnet_ain_x0_5', [256,128], 'osnet_ain_x0_5_imagenet.pyth', 'cosine'],
    ['osnet_ain_x0_75', [256,128], 'osnet_ain_x0_75_imagenet.pyth', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_x1_0_imagenet.pth', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth', 'cosine'],
    ['osnet_ain_x1_0', [256,128], 'osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_d_m_c.pth.tar', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_d_m_c.pth.tar', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_ms_d_c.pth.tar', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_ms_d_m.pth.tar', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_ms_m_c.pth.tar', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_x1_0_imagenet.pth', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['osnet_ibn_x1_0', [256,128], 'osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_ms_d_c.pth.tar', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_ms_d_m.pth.tar', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_ms_m_c.pth.tar', 'cosine'],
    ['osnet_x0_25', [256,128], 'osnet_x0_25_duke_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_25', [256,128], 'osnet_x0_25_imagenet.pth', 'cosine'],
    ['osnet_x0_25', [256,128], 'osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_25', [256,128], 'osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_25', [256,128], 'osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['osnet_x0_5', [256,128], 'osnet_x0_5_duke_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_5', [256,128], 'osnet_x0_5_imagenet.pth', 'cosine'],
    ['osnet_x0_5', [256,128], 'osnet_x0_5_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_5', [256,128], 'osnet_x0_5_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_5', [256,128], 'osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['osnet_x0_75', [256,128], 'osnet_x0_75_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_75', [256,128], 'osnet_x0_75_imagenet.pth', 'cosine'],
    ['osnet_x0_75', [256,128], 'osnet_x0_75_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_75', [256,128], 'osnet_x0_75_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x0_75', [256,128], 'osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_x1_0_imagenet.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'cosine'],
    ['osnet_x1_0', [256,128], 'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['resnet50', [256,128], 'resnet50_fc512_msmt_xent.pth.tar', 'cosine'],
    ['resnet50', [256,128], 'resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth', 'cosine'],
    ['shufflenet', [256,128], 'shufflenet-bee1b265.pth.tar', 'cosine'],
]

def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.
    If the folder does not exist, it is created.

    :param url: URL of the file to download.
    :param folder: Folder where the file will be saved.
    :param filename: Filename to save the file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Full path for the file
    file_path = os.path.join(folder, filename)
    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download completed: {file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

for model_name, [H, W], weight_file, distance in MODELS:
    url = "https://github.com/PINTO0309/deep-person-reid/releases/download/weights/" + weight_file

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ exporting')
    print(os.path.join("weights", weight_file))
    if not os.path.isfile(os.path.join("weights", weight_file)):
        download_file(url=url, folder="weights", filename=weight_file)

    model = SimilarityCalculator(
        model_name=model_name,
        model_path=f'weights/{weight_file}',
        device='cpu',
        image_size=(H, W),
        distance=distance,
        verbose=False,
    )

    MODEL = os.path.basename(weight_file).split('.', 1)[0]

    onnx_file = f"{MODEL}_11x3x{H}x{W}.onnx"
    x = torch.randn(1, 3, H, W).cpu()
    y = torch.randn(1, 3, H, W).cpu()
    torch.onnx.export(
        model,
        args=(x, y),
        f=onnx_file,
        opset_version=11,
        input_names=['base_image', 'target_image'],
        output_names=['similarity'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    if model_name != "shufflenet":
        onnx_file = f"{MODEL}_1Nx3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        y = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            model,
            args=(x, y),
            f=onnx_file,
            opset_version=11,
            input_names=['base_image', 'target_images'],
            output_names=['similarities'],
            dynamic_axes={
                'target_images' : {0: 'N'},
                'similarities' : {0: '1', 1: 'N'},
            }
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

    if model_name != "shufflenet":
        onnx_file = f"{MODEL}_NMx3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        y = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            model,
            args=(x, y),
            f=onnx_file,
            opset_version=11,
            input_names=['base_images', 'target_images'],
            output_names=['similarities'],
            dynamic_axes={
                'base_images' : {0: 'N'},
                'target_images' : {0: 'M'},
                'similarities' : {0: 'N', 1: 'M'},
            }
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

