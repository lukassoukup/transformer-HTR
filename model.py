import torch
from torch import nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50, mobilenet_v2


class TransformerModel(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        model_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        patch_width,
        backbone='resnet50',
        pretrained=True,
        cnn_channels=4,
        img_height=64
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.model_dim = model_dim
        self.patch_width = patch_width

        # LAYERS
        if backbone == 'custom_cnn':
            self.backbone = CNN(cnn_channels)
            self.backbone_output_stride = 2
            input_dim = self.backbone.last_layer_channels * (img_height // self.backbone_output_stride)
        elif backbone == 'resnet50':
            # self.backbone = timm.create_model('resnet50', pretrained=True,  # num_classes=0, global_pool='',
            #                                   features_only=True, out_indices=[3])
            resnet = resnet50(pretrained=pretrained)
            self.backbone = torch.nn.Sequential(*(list(resnet.children())[:-3]))
            # modules = self.backbone.modules()
            # set stride to 1 in last block of resnet
            self.backbone._modules['6']._modules['0'].conv2.stride = (1, 1)
            self.backbone._modules['6']._modules['0'].downsample._modules['0'].stride = (1, 1)

            self.backbone_output_stride = 8
            input_dim = 1024 * (img_height // self.backbone_output_stride)

        elif backbone == 'mobilenetv2':
            mobilenet = mobilenet_v2(pretrained=pretrained)

            # block 14 has stride 2
            self.backbone = torch.nn.Sequential(*(list(mobilenet.children())[0][:14]))
            self.backbone._modules['7']._modules['conv']._modules['1']._modules['0'].stride = (1, 1)
            self.backbone_output_stride = 8
            input_dim = 96 * (img_height // self.backbone_output_stride)
        else:
            raise NotImplementedError()

        # self.input_embedding = nn.Conv1d(input_dim, model_dim, patch_width, stride=patch_width)
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.output_embedding = nn.Embedding(num_tokens, model_dim)

        self.positional_encoder = PositionalEncoding(
            dim_model=model_dim, max_len=5000
        )

        self.weird_linear = nn.Linear(model_dim, model_dim)

        # self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.classification = nn.Linear(model_dim, num_tokens)

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
        source_len=None,
        tgt_pad_mask=None
    ):
        # Src size must be (batch_size, channels, height, width)
        # Tgt size must be (batch_size, tgt sequence length)

        src = self.backbone(src)
        # calculate length of image without padding
        if source_len is not None:
            # max pooling twice
            source_len = source_len // self.backbone_output_stride
        else:
            # set whole image if no length provided
            source_len = torch.ones((src.shape[0]), dtype=torch.int) * src.shape[-1]
        # Src processed by CNN creates (batch_size, channels, height, width)
        src = self.make_patches(src)
        src = src.permute(0, 2, 1)
        # Src patches created by flattening channels and height (batch_size, c * h * patch_width, w / patch_width)

        # shrink dimensions of input from c * h * patch_width to model_dim
        src = self.input_embedding(src)
        tgt = self.output_embedding(tgt)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = src.permute(0, 2, 1)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = self.weird_linear(src)

        # create padding mask of input image
        source_len = source_len // self.patch_width
        src_pad_mask = torch.ones(src.shape[:2], dtype=torch.bool, device=src.device)
        for i in range(src.shape[0]):
            src_pad_mask[i, :source_len[i]] = False

        # we permute to obtain size (sequence length, batch_size, model_dim),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.classification(transformer_out)

        return out

    @staticmethod
    def get_tgt_mask(size) -> torch.tensor:
        # Generates a square matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    @staticmethod
    def create_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token

    @staticmethod
    def make_patches(feature_map):
        bs, c, h, w = feature_map.shape

        patch_dim = c * h
        patches = torch.zeros((bs, patch_dim, w), device=feature_map.device)
        for b in range(bs):
            for pos, i in enumerate(range(0, w)):
                patch = feature_map[b, :, :, i:(i + 1)]
                patch_f = patch.flatten()
                patches[b, :, pos] = patch_f

        return patches


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        # return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        return token_embedding + self.pos_encoding[:token_embedding.size(0), :]


class CNN(nn.Module):
    def __init__(self, last_layer_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.last_layer_channels = 16
        self.conv4 = nn.Conv2d(16, self.last_layer_channels, (3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
