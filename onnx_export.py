from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='a/b/c/model.pth.tar',
    device='cuda'
)

image_list = [
    'a/b/c/image001.jpg',
    'a/b/c/image002.jpg',
    'a/b/c/image003.jpg',
    'a/b/c/image004.jpg',
    'a/b/c/image005.jpg'
]

