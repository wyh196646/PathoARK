import torch
from torch import nn

from . import slide_encoder
from typing import Optional


class MultiTaskHead(nn.Module):
    """
    Shared slide encoder + optional projector + multiple classification heads (one per dataset/task).

    Arguments:
    ----------
    input_dim: int
        Input tile-embedding dimension (e.g., 1536)
    latent_dim: int
        Hidden dimension of slide encoder
    feat_layer: str
        Layers to concatenate as features, e.g., "5-11"
    num_classes_list: list[int]
        Number of classes per dataset/task (head)
    model_arch: str
        Slide encoder architecture registered in slide_encoder (e.g., 'gigapath_slide_enc12l768d')
    pretrained: str
        Pretrained slide encoder weights (path or hf_hub:prov-gigapath/prov-gigapath)
    freeze: bool
        Whether to freeze slide encoder
    projector_dim: int | None
        If provided, add a linear or MLP projector from feature dim -> projector_dim
    use_mlp: bool
        If True, projector is 2-layer MLP with ReLU
    global_pool: bool
        Whether slide_encoder uses global pooling instead of [CLS]
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        feat_layer: str,
        num_classes_list,
        model_arch: str = "gigapath_slide_enc12l768d",
        pretrained: str = "hf_hub:prov-gigapath/prov-gigapath",
        freeze: bool = False,
    projector_dim: Optional[int] = None,
        use_mlp: bool = False,
        global_pool: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # slide encoder
        self.feat_layer_idx = [eval(x) for x in feat_layer.split("-")]
        self.enc_out_dim = latent_dim * len(self.feat_layer_idx)
        self.slide_encoder = slide_encoder.create_model(
            pretrained, model_arch, in_chans=input_dim, global_pool=global_pool,
            dropout=dropout, drop_path_rate=drop_path_rate, **kwargs
        )

        # freeze encoder if requested
        if freeze:
            for p in self.slide_encoder.named_parameters():
                p[1].requires_grad = False

        # optional projector for consistency loss space
        self.projector = None
        proj_in = self.enc_out_dim
        if projector_dim is not None and projector_dim > 0:
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(proj_in, projector_dim), nn.ReLU(inplace=True), nn.Linear(projector_dim, projector_dim)
                )
                self.feat_dim = projector_dim
            else:
                self.projector = nn.Linear(proj_in, projector_dim)
                self.feat_dim = projector_dim
        else:
            self.feat_dim = self.enc_out_dim

        # multi heads
        heads = []
        for nc in num_classes_list:
            if nc > 0:
                heads.append(nn.Linear(self.feat_dim, nc))
            else:
                heads.append(nn.Identity())
        self.heads = nn.ModuleList(heads)

    def _extract_features(self, images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # ensure batch dimension
        if images.dim() == 2:
            images = images.unsqueeze(0)
        # encoder returns list of layer outputs
        enc_list = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        enc_list = [enc_list[i] for i in self.feat_layer_idx]
        feats = torch.cat(enc_list, dim=-1)
        feats = feats.reshape([-1, feats.size(-1)])
        if self.projector is not None:
            feats = self.projector(feats)
        return feats

    def forward(self, images: torch.Tensor, coords: torch.Tensor, head_n: Optional[int] = None):
        """
        If head_n is not None: returns (features, logits) for that head.
        Else: returns list of logits for all heads.
        """
        feats = self._extract_features(images, coords)
        if head_n is not None:
            return feats, self.heads[head_n](feats)
        else:
            return [head(feats) for head in self.heads]

    def generate_embeddings(self, images: torch.Tensor, coords: torch.Tensor, after_proj: bool = True) -> torch.Tensor:
        feats = self._extract_features(images, coords)
        return feats if after_proj else feats  # currently same, projector already handled


def get_model(**kwargs):
    # kept for parity with existing get_model pattern
    return MultiTaskHead(**kwargs)
