import os
import onnx
import glob

folder_path = '1x3xHxW'

onnx_files = glob.glob(os.path.join(folder_path, '*.onnx'))
onnx_files = sorted(onnx_files)

for file in onnx_files:
    onnx_model = onnx.load(file)
    feature_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    filename = os.path.basename(file)
    print(f'["{filename}", {feature_shape}],')

"""
["mlfn-9cb5a267_1x3x256x128.onnx", [1, 1024]],
["mobilenetv2_1_1x3x256x128.onnx", [1, 1792]],
["mobilenetv2_1dot0_duke_1x3x256x128.onnx", [1, 1280]],
["mobilenetv2_1dot0_market_1x3x256x128.onnx", [1, 1280]],
["mobilenetv2_1dot0_msmt_1x3x256x128.onnx", [1, 1280]],
["mobilenetv2_1dot4_duke_1x3x256x128.onnx", [1, 1792]],
["mobilenetv2_1dot4_market_1x3x256x128.onnx", [1, 1792]],
["mobilenetv2_1dot4_msmt_1x3x256x128.onnx", [1, 1792]],
["osnet_ain_d_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_ain_ms_d_c_1x3x256x128.onnx", [1, 512]],
["osnet_ain_ms_d_m_1x3x256x128.onnx", [1, 512]],
["osnet_ain_ms_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x0_25_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x0_5_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x0_75_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x1_0_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_d_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_d_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_ms_d_c_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_ms_d_m_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_ms_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_x1_0_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_ms_d_c_1x3x256x128.onnx", [1, 512]],
["osnet_ms_d_m_1x3x256x128.onnx", [1, 512]],
["osnet_ms_m_c_1x3x256x128.onnx", [1, 512]],
["osnet_x0_25_duke_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_25_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_5_duke_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_5_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_x0_5_market_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_5_msmt17_256x128_amsgrad_ep180_stp80_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_75_duke_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_75_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_x0_75_market_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_75_msmt17_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x1_0_imagenet_1x3x256x128.onnx", [1, 512]],
["osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x1_0_msmt17_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 512]],
["resnet50_fc512_msmt_xent_1x3x256x128.onnx", [1, 2048]],
["resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_1x3x256x128.onnx", [1, 2048]],
["shufflenet-bee1b265_1x3x256x128.onnx", [1, 960]],
"""