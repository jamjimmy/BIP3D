from torch import nn

from mmdet.models.detectors import BaseDetector
from mmdet.models.detectors.deformable_detr import (
    MultiScaleDeformableAttention,
)

from bip3d.registry import MODELS
import torch.nn.functional as F
import torch
@MODELS.register_module()
class BIP3DConv(BaseDetector):
    def __init__(
        self,
        backbone,
        decoder,
        neck=None,
        text_encoder=None,
        feature_enhancer=None,
        spatial_enhancer=None,
        data_preprocessor=None,
        backbone_3d=None,
        neck_3d=None,
        init_cfg=None,
        input_2d="imgs",
        input_3d="depth_img",
        embed_dims=256,
        use_img_grid_mask=False,
        use_depth_grid_mask=False,
        slam3r_proj=None,
    ):
        super().__init__(data_preprocessor, init_cfg)

        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)

        self.backbone = build(backbone)
        self.decoder = build(decoder)
        self.neck = build(neck)
        self.text_encoder = build(text_encoder)
        self.feature_enhancer = build(feature_enhancer)
        self.spatial_enhancer = build(spatial_enhancer)
        self.backbone_3d = build(backbone_3d)
        self.neck_3d = build(neck_3d)
        self.input_2d = input_2d
        self.input_3d = input_3d
        self.embed_dims = embed_dims
        if text_encoder is not None:
            self.text_feat_map = nn.Linear(
                self.text_encoder.language_backbone.body.language_dim,
                self.embed_dims,
                bias=True,
            )

        self.use_img_grid_mask = use_img_grid_mask
        self.use_depth_grid_mask = use_depth_grid_mask
        if use_depth_grid_mask or use_img_grid_mask:
            from ..grid_mask import GridMask

            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        self.slam3r_proj = build(slam3r_proj)
        # nn.Sequential(
        #     nn.Conv2d(4, 16, kernel_size=3, padding=1),  # [h, w, 4] -> [h, w, 16]
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1), # [h, w, 16] -> [h, w, 32]
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1) # [h, w, 32] -> [h, w, 1]
        # )

    def init_weights(self):
        """Initialize weights for Transformer and other components."""
        for p in self.feature_enhancer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.decoder.init_weights()
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.feature_enhancer.level_embed)
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        imgs = batch_inputs_dict.get(self.input_2d)
        if imgs.dim() == 5:
            bs, num_cams = imgs.shape[:2]
            imgs = imgs.flatten(end_dim=1)
        else:
            bs = imgs.shape[0]
            num_cams = 1

        if self.use_img_grid_mask and self.training:
            img = self.grid_mask(
                img,
                offset=-self.data_preprocessor.mean
                / self.data_preprocessor.std,
            )
        feature_maps = self.backbone(imgs)
        if self.neck is not None:
            feature_maps = self.neck(feature_maps)
        feature_maps = [x.unflatten(0, (bs, num_cams)) for x in feature_maps]
        slam3r_feature = None
        
        slam3r_feature = torch.stack(batch_inputs_dict['slam3r_feature'])
        B, N, _ = slam3r_feature.shape
        # print('***************************************')
        # print(B,N)
        slam3r_feature = slam3r_feature.view(-1, 1024, 14 , 14) # [50, 1024, 14, 14]
        slam3r_feature = F.pixel_shuffle(slam3r_feature, upscale_factor=16)  # [B*N, 4, 224, 224]
        slam3r_feature = self.slam3r_proj(slam3r_feature)  # [B*N, 1, 224, 224]
        

        slam3r_feature = F.interpolate(slam3r_feature, size=(512, 512), mode='bilinear', align_corners=False)
        slam3r_feature = slam3r_feature.view(B, N, 1, 512, 512)

        # input_3d = batch_inputs_dict.get(self.input_3d) # [1, 50, 1, 512, 512]

        input_3d = slam3r_feature
        if self.backbone_3d is not None and input_3d is not None:
            if self.input_3d == "depth_img" and input_3d.dim() == 5:
                assert input_3d.shape[1] == num_cams
                input_3d = input_3d.flatten(end_dim=1)
            if self.use_depth_grid_mask and self.training:
                input_3d = self.grid_mask(input_3d)
            feature_3d = self.backbone_3d(input_3d) # [50, 8, 64, 64] [50, 16, 32, 32] [50, 32, 16, 16]
            if self.neck_3d is not None:
                feature_3d = self.neck_3d(feature_3d) # [50, 32, 64, 64] [50, 32, 32, 32] [50, 32, 16, 16] [50,32,8,8]
            feature_3d = [x.unflatten(0, (bs, num_cams)) for x in feature_3d]
        else:
            feature_3d = None
        return feature_maps, feature_3d

    def extract_text_feature(self, batch_inputs_dict):
        if self.text_encoder is not None:
            text_dict = self.text_encoder(batch_inputs_dict["text"])
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])
        else:
            text_dict = None
        return text_dict

    def loss(self, batch_inputs, batch_data_samples):
        model_outs, text_dict, loss_depth = self._forward(
            batch_inputs, batch_data_samples
        )
        loss = self.decoder.loss(model_outs, batch_inputs, text_dict=text_dict)
        if loss_depth is not None:
            loss["loss_depth"] = loss_depth
        return loss

    def predict(self, batch_inputs, batch_data_samples):
        model_outs, text_dict = self._forward(batch_inputs, batch_data_samples)
        results = self.decoder.post_process(
            model_outs, text_dict, batch_inputs, batch_data_samples
        )
        return results

    def _forward(self, batch_inputs, batch_data_samples):
        feature_maps, feature_3d = self.extract_feat(
            batch_inputs, batch_data_samples
        )
        text_dict = self.extract_text_feature(batch_inputs)
        if self.feature_enhancer is not None:
            feature_maps, text_feature = self.feature_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                batch_inputs=batch_inputs,
                batch_data_samples=batch_data_samples,
            )
            text_dict["embedded"] = text_feature
        if self.spatial_enhancer is not None:
            feature_maps, depth_prob, loss_depth = self.spatial_enhancer(
                feature_maps=feature_maps,
                feature_3d=feature_3d,
                text_dict=text_dict,
                batch_inputs=batch_inputs,
                batch_data_samples=batch_data_samples,
            )
        else:
            loss_depth = depth_prob = None
        model_outs = self.decoder(
            feature_maps=feature_maps,
            feature_3d=feature_3d,
            text_dict=text_dict,
            batch_inputs=batch_inputs,
            batch_data_samples=batch_data_samples,
            depth_prob=depth_prob,
        )
        if self.training:
            return model_outs, text_dict, loss_depth
        return model_outs, text_dict
