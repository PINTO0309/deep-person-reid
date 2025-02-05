from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model

class SimilarityCalculatorFeatureOnly(nn.Module):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True,
        distance='euclidean'
    ):
        super(SimilarityCalculatorFeatureOnly, self).__init__()
        bf = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            image_size=(image_size[0], image_size[1]),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            device=device,
            verbose=verbose,
        )
        self.base_model = bf.model
        self.distance = distance # euclidean, cosine

    def _pairwise_distance_primitive(self, x, y):
        # xとyをブロードキャストできる形に調整
        x = x.unsqueeze(1)  # xの形状を[2, 1, 1280]に変更
        y = y.unsqueeze(0)  # yの形状を[1, 3, 1280]に変更

        # これにより、xとyの形状が[2, 3, 1280]にブロードキャストされ、
        # それぞれのペアの差を計算できる
        diff = x - y
        squared_diff = diff.pow(2)
        sum_squared_diff = torch.sum(squared_diff, dim=2)  # 最後の次元に沿って和を取る
        distance = torch.sqrt(sum_squared_diff)
        return distance

    def forward(self, base_input, target_features):
        with torch.no_grad():
            base_features = self.base_model(base_input)

            if self.distance == 'euclidean':
                euclidean_distance = self._pairwise_distance_primitive(base_features, target_features)
                similarity = torch.clip(1.0 / euclidean_distance, max=1/1e-6)

            elif self.distance == 'cosine':
                base_features_norm = F.normalize(base_features, dim=1)
                target_features_norm = F.normalize(target_features, dim=1)
                similarity = base_features_norm.matmul(target_features_norm.transpose(1, 0))
            else:
                raise NotImplementedError

        return base_features, similarity


class SimilarityCalculator(nn.Module):
    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True,
        distance='euclidean'
    ):
        super(SimilarityCalculator, self).__init__()
        bf = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            image_size=(image_size[0], image_size[1]),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            device=device,
            verbose=verbose,
        )
        self.base_model = bf.model

        tf = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            image_size=(image_size[0], image_size[1]),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            device=device,
            verbose=verbose,
        )
        self.target_model = tf.model
        self.distance = distance # euclidean, cosine

    def _pairwise_distance_primitive(self, x, y):
        # xとyをブロードキャストできる形に調整
        x = x.unsqueeze(1)  # xの形状を[2, 1, 1280]に変更
        y = y.unsqueeze(0)  # yの形状を[1, 3, 1280]に変更

        # これにより、xとyの形状が[2, 3, 1280]にブロードキャストされ、
        # それぞれのペアの差を計算できる
        diff = x - y
        squared_diff = diff.pow(2)
        sum_squared_diff = torch.sum(squared_diff, dim=2)  # 最後の次元に沿って和を取る
        distance = torch.sqrt(sum_squared_diff)
        return distance

    def forward(self, base_input, target_input):
        with torch.no_grad():
            base_features = self.base_model(base_input)
            target_features = self.target_model(target_input)

            if self.distance == 'euclidean':
                euclidean_distance = self._pairwise_distance_primitive(base_features, target_features)
                similarity = torch.clip(1.0 / euclidean_distance, max=1/1e-6)

            elif self.distance == 'cosine':
                base_features_norm = F.normalize(base_features, dim=1)
                target_features_norm = F.normalize(target_features, dim=1)
                similarity = base_features_norm.matmul(target_features_norm.transpose(1, 0))
            else:
                raise NotImplementedError

        return similarity

class FeatureExtractor(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

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

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):
        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (model_path and check_isfile(model_path)),
            use_gpu=device.startswith('cuda')
        )
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(
                model, (1, 3, image_size[0], image_size[1])
            )
            print('Model: {}'.format(model_name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        device = torch.device(device)
        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)

            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, np.ndarray):
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            features = self.model(images)

        return features
