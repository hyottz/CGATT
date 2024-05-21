import math

from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.models.utils import set_requires_grad


class ModelCombined(nn.Module):
    def __init__(self, cfg=None, scale=64):
        super().__init__()

        self.backbone = instantiate(cfg.model.visual_backbone.obj)
        self.df_head = instantiate(cfg.model.df_predictor, cfg.model.visual_backbone.output_dim, scale=scale) # df_head(predictor) : 2048 -> 1
        self.scale = scale

        self.ssl_weight = cfg.model.ssl_weight
        if not math.isclose(self.ssl_weight, 0.0):
            target_backbone = instantiate(cfg.model.visual_backbone.obj)
            self.target_encoder = nn.Sequential( # sequential: target_backbone: input -> 2048, projector: 2048 -> 2048
                target_backbone, instantiate(cfg.model.projector, in_dim=cfg.model.visual_backbone.output_dim)
            )
            set_requires_grad(self.target_encoder, False)
            self.ssl_head = nn.Sequential(
                instantiate(cfg.model.projector, in_dim=cfg.model.visual_backbone.output_dim),
                instantiate(cfg.model.predictor, in_dim=cfg.model.projection_size),
            )

        num_fakes = len(cfg.data.dataset_df.fake_types_train) # 4
        prior_fake = num_fakes / ((num_fakes + 1) * (cfg.model.relative_bs + 1)) # 4 / ((4+1) * (8+1)) = 4/45 = 0.0888

        self.logit_adj = (
            torch.log(torch.tensor(prior_fake) / (1.0 - torch.tensor(prior_fake))) if cfg.model.logit_adj else 0.0
        ) # torch.log (0.0888 / (1-0.0888)) = -2.32

    @torch.no_grad()
    def get_targets(self, x_ssl):
        #print(f"X_SSL shape: {x_ssl.shape}")
        return self.target_encoder(x_ssl)

    def forward(self, videos_df, labels_df, videos_df_clean=None, videos_ssl=None, videos_ssl_clean=None):
        #print(f"\nInitial videos_df shape: {videos_df.shape}")  ### 입력 차원 확인
        videos, videos_clean = videos_df, videos_df_clean
        if videos_ssl is not None:  # Treat SSL videos as real
            videos, videos_clean = torch.cat([videos_df, videos_ssl]), torch.cat([videos_df_clean, videos_ssl_clean])
            zeros = torch.zeros(videos_ssl.size(0), dtype=labels_df.dtype, device=labels_df.device)
            labels_df = torch.cat([labels_df, zeros])

        # DeepFake detection loss
        features = self.backbone(videos)
        #print(f"Features shape after backbone: {features.shape}")  ### backbone을 통과한 후 차원 확인
        logits = self.df_head(features)
        #print("Logits shape: ", logits.shape)  ### logits 차원 확인
        loss_df = F.binary_cross_entropy_with_logits(logits.squeeze(-1) + self.logit_adj, labels_df.float())
        #print("pass")
        # SSL loss
        loss_ssl = 0.0
        if not math.isclose(self.ssl_weight, 0.0):
            if torch.all(labels_df == 0) or torch.all(labels_df == 1):
                # 모든 레이블이 같은 경우, SSL 처리를 건너뛰고 loss_ssl을 0으로 유지
                print("All labels in the batch are the same. Skipping SSL processing for this batch.")
            else:
                #print("[before get_targets]\nvideo_clean shape:", videos_clean.shape,"\nlabels_df.shape", labels_df.shape)
                targets = self.get_targets(videos_clean[~labels_df.bool()])  # Only for real videos
                #print("[after get_targets]\ntargets shape:", targets.shape)
                #print("[before ssl]\nfeatures shape:", features[~labels_df.bool()].shape, "\nlabels_df.shape", labels_df.shape)
                predictions = self.ssl_head(features[~labels_df.bool()])
                #print("[after ssl]\npredictions shape:", predictions.shape,'\n')
                loss_ssl = F.cosine_similarity(predictions, targets, dim=-1).mean()
                #print(f"Targets shape: {targets.shape},\nPredictions shape: {predictions.shape}\n") ### target과 prediction의 차원 확인

        return loss_df, loss_ssl
    