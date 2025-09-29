import torch
import os

from torch import nn
from . import slide_encoder


def reshape_input(imgs, coords, pad_mask=None):
    if len(imgs.shape) == 4:
        imgs = imgs.squeeze(0)
    if len(coords.shape) == 4:
        coords = coords.squeeze(0)
    if pad_mask is not None:
        if len(pad_mask.shape) != 2:
            pad_mask = pad_mask.squeeze(0)
    return imgs, coords, pad_mask


class ClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="gigapath_slide_enc12l768d",
        pretrained="",
        freeze=False,
        ark_pretrained_ckpt: str = '',
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # Optionally load ARK pretraining checkpoint (MultiTaskHead) and extract slide_encoder weights
        if ark_pretrained_ckpt:
            if os.path.isfile(ark_pretrained_ckpt):
                try:
                    ckpt = torch.load(ark_pretrained_ckpt, map_location='cpu')
                    state_dict = ckpt.get('state_dict', ckpt)
                    # Filter keys that belong to slide_encoder inside MultiTaskHead: e.g., 'slide_encoder.xxx'
                    slide_keys = {k.split('slide_encoder.',1)[1]: v for k, v in state_dict.items() if k.startswith('slide_encoder.')}
                    missing, unexpected = self.slide_encoder.load_state_dict(slide_keys, strict=False)
                    print(f"Loaded ARK slide_encoder weights from {ark_pretrained_ckpt}. Missing={len(missing)} Unexpected={len(unexpected)}")
                except Exception as e:
                    print(f"Failed to load ARK pretrained checkpoint {ark_pretrained_ckpt}: {e}")
            else:
                print(f"ARK pretrained checkpoint not found: {ark_pretrained_ckpt}")

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])

    def forward(self, images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
        logits = self.classifier(h)
        return logits


def get_model(**kwargs):
    model = ClassificationHead(**kwargs)
    return model
