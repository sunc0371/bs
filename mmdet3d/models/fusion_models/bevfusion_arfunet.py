import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models import FUSIONMODELS
from mmcv.cnn import build_norm_layer
from .mytransformer import CustonDATransformer
from .bev_net import BEVFusion

class UNetARF(nn.Module):
    """
    该类实现了 U-Net 和 ARF 融合策略，
    其中 UNet 负责对多个输入特征（例如 BEV、语义特征等）进行融合，
    ARFModule 负责基于注意力机制对融合后的特征进行动态加权调整。
    """
    def __init__(self, in_channels, out_channels, arf_weight_adjustment=True, **kwargs):
        super(UNetARF, self).__init__()

        # U-Net Architecture: Encoder-Decoder Structure
        self.unet = UNet(in_channels, out_channels)
        self.arf = ARFModule(arf_weight_adjustment)  # ARF module for dynamic weight adjustment

    def forward(self, features):
        """
        Forward pass for U-Net + ARF Fusion.
        features: List of input features from different modalities.
        """
        # Pass the features through the U-Net part for fusion
        fused_features = self.unet(features)
        
        # Apply ARF module to adjust the weights dynamically
        adjusted_features = self.arf(fused_features)
        
        return adjusted_features

class BEVFusionARF(UNetARF):
    """
    继承自 UNetARF，这个类主要处理多尺度特征的融合。
    它使用 U-Net 来对不同尺度的特征进行融合，然后利用 ARF 模块对其进行动态调整。
    """
    def __init__(self, in_channels, out_channels, use_DA=False, **kwargs):
        super(BEVFusionARF, self).__init__(in_channels, out_channels, **kwargs)
        self.use_DA = use_DA
        
        # Custom Dynamic Adjustment (ARF)
        if self.use_DA:
            self.DA = CustonDATransformer(embed_dims=out_channels)

    def forward(self, bev_features, sem_features, com_features, img_features):
        """
        Forward pass for BEVFusion with dynamic ARF adjustment.
        Fuses BEV, semantic, commercial, and image features using dynamic attention.
        """
        # First, apply U-Net + ARF fusion on the BEV, semantic, and commercial features
        features = [bev_features, sem_features, com_features]

        # Use U-Net to combine the features
        fused_features = self.unet(features)
        
        # If DA (Dynamic Adjustment) is used, apply it to the fused features
        if self.use_DA:
            fused_features = self.DA(img_features, fused_features)

        # Return the final fused features after ARF adjustment
        adjusted_features = self.arf(fused_features)
        return adjusted_features

@FUSIONMODELS.register_module()
class BEVFusionARFunetModule(nn.Module):
    """
    用U-Net + ARF融合取代原有BEV融合逻辑的融合模块。
    该模块包括多尺度融合和ARF动态调整。
    """
    def __init__(self, encoders, fuser, decoder, heads, use_DA=False, **kwargs):
        super().__init__()

        # Initialize the U-Net + ARF fusion module for different scales
        self.fusion_scale_1 = BEVFusionARF(in_channels=128, out_channels=256, use_DA=use_DA)
        self.fusion_scale_2 = BEVFusionARF(in_channels=256, out_channels=512, use_DA=use_DA)
        self.fusion_scale_3 = BEVFusionARF(in_channels=512, out_channels=1024, use_DA=use_DA)

        # Decoder and heads (same as existing DAocc structure)
        self.decoder = decoder
        self.heads = heads
        self.fuser = fuser

        # Initialize weights (optional)
        self.init_weights()

    def init_weights(self):
        # Initialize decoder and heads weights if necessary
        pass

    def forward(self, img_features, lidar_features, sem_features, com_features):
        """
        和 DAOcc 中的 BEVFusion 保持一致，如果 fuser 配置存在，它将对不同尺度的特征进行融合。
        如果有解码器 (decoder)，则继续进行解码操作
        Perform forward pass through the BEVFusionARF module.
        img_features, lidar_features, sem_features, and com_features are the input features.
        """

        # Perform multi-scale fusion using the U-Net + ARF structure
        fusion_1 = self.fusion_scale_1(lidar_features, sem_features, com_features, img_features)
        fusion_2 = self.fusion_scale_2(fusion_1, sem_features, com_features, img_features)
        fusion_3 = self.fusion_scale_3(fusion_2, sem_features, com_features, img_features)

        # Optionally use fuser (if provided) for final fusion
        if self.fuser is not None:
            final_fusion = self.fuser([fusion_1, fusion_2, fusion_3])
        else:
            final_fusion = fusion_3

        # Decode the fused features if a decoder is available
        if self.decoder is not None:
            x = self.decoder["backbone"](final_fusion)
            x = self.decoder["neck"](x)

        # Pass through heads for final predictions
        outputs = {}
        for head_type, head in self.heads.items():
            if head_type == "object":
                pred_dict = head(x)
                outputs.update(pred_dict)
            elif head_type == "map":
                outputs.update(head(x))

        return outputs
