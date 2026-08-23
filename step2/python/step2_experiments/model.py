"""Thin continuous-token adapter around the maintained Hugging Face Llama core."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class Step2Config(PretrainedConfig):
    """Serializable configuration for the project-owned physical-token adapter."""

    model_type = "step2_trajectory"

    def __init__(
        self,
        *,
        hidden_size: int = 384,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 12,
        attention_heads: int = 6,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1.0e-5,
        rope_theta: float = 10_000.0,
        initializer_range: float = 0.02,
        num_roles: int = 10,
        payload_dim: int = 8,
        action_horizon: int = 16,
        action_loss_weight: float = 1.0,
        outcome_loss_weight: float = 0.5,
        vision_channels: int = 3,
        vision_patch_size: int = 16,
        vision_resampler_tokens: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for deterministic Fourier keys")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_heads = attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.num_roles = num_roles
        self.payload_dim = payload_dim
        self.action_horizon = action_horizon
        self.action_loss_weight = action_loss_weight
        self.outcome_loss_weight = outcome_loss_weight
        self.vision_channels = vision_channels
        self.vision_patch_size = vision_patch_size
        self.vision_resampler_tokens = vision_resampler_tokens

    @classmethod
    def from_project_json(cls, path: str | Path) -> "Step2Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            vocab_size=1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.attention_heads,
            num_key_value_heads=self.attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            initializer_range=self.initializer_range,
            tie_word_embeddings=False,
            use_cache=False,
        )


@dataclass
class Step2Output(ModelOutput):
    loss: torch.Tensor | None = None
    action_loss: torch.Tensor | None = None
    outcome_loss: torch.Tensor | None = None
    action_predictions: torch.Tensor | None = None
    outcome_predictions: torch.Tensor | None = None


class Step2ForTrajectoryPrediction(PreTrainedModel):
    """ICRT-derived trajectory learner with variable public readout queries.

    Transformer blocks, causal masking, RoPE, RMSNorm, and SwiGLU are owned by
    ``transformers.LlamaModel``. This class owns only continuous event adapters,
    variable action/outcome readouts, and the random visual tokenizer.
    """

    config_class = Step2Config
    base_model_prefix = "backbone"
    main_input_name = "role_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: Step2Config) -> None:
        super().__init__(config)
        h = config.hidden_size
        self.backbone = LlamaModel(config.llama_config())
        self.role_embedding = nn.Embedding(config.num_roles, h, padding_idx=0)
        self.payload_projector = nn.Linear(config.payload_dim, h, bias=False)
        half = h // 2
        key_frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        # Persist this deterministic buffer. Transformers' low-memory
        # ``from_pretrained`` construction may allocate non-persistent buffers
        # without running their ordinary value construction, which would make
        # an otherwise exact checkpoint reload change the public-key encoding.
        self.register_buffer("key_frequencies", key_frequencies, persistent=True)

        self.action_head = nn.Linear(h, config.action_horizon)
        self.outcome_head = nn.Linear(h, 1)

        patch = config.vision_patch_size
        self.visual_patch_stem = nn.Conv2d(
            config.vision_channels,
            h,
            kernel_size=patch,
            stride=patch,
        )
        self.visual_coordinate_projector = nn.Linear(2, h, bias=False)
        self.visual_queries = nn.Parameter(torch.empty(config.vision_resampler_tokens, h))
        self.visual_resampler = nn.MultiheadAttention(
            h,
            config.attention_heads,
            batch_first=True,
        )
        self.visual_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=self.config.initializer_range)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        if module is self:
            nn.init.normal_(self.visual_queries, mean=0.0, std=self.config.initializer_range)

    def deterministic_key_embedding(self, key_ids: torch.Tensor) -> torch.Tensor:
        angles = key_ids.to(dtype=self.key_frequencies.dtype).unsqueeze(-1) * self.key_frequencies
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoded

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return eight random-trainable visual tokens per image by default."""
        patches = self.visual_patch_stem(images)
        batch, hidden, height, width = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        ys = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=patches.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=patches.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height * width, 2)
        patches = patches + self.visual_coordinate_projector(coordinates).expand(batch, -1, -1)
        queries = self.visual_queries.to(dtype=patches.dtype).unsqueeze(0).expand(batch, -1, -1)
        resampled, _ = self.visual_resampler(queries, patches, patches, need_weights=False)
        return self.visual_norm(resampled)

    def embed_events(
        self,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        payloads: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor | None = None,
        visual_token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = (
            self.role_embedding(role_ids)
            + self.payload_projector(payloads)
            + self.deterministic_key_embedding(key_ids).to(dtype=payloads.dtype)
        )
        if images is not None or visual_token_positions is not None:
            if images is None or visual_token_positions is None:
                raise ValueError("images and visual_token_positions must be supplied together")
            if images.ndim != 5:
                raise ValueError("images must have shape [batch, views, channels, height, width]")
            batch, views = images.shape[:2]
            encoded = self.encode_images(images.flatten(0, 1)).reshape(
                batch, views, self.config.vision_resampler_tokens, self.config.hidden_size
            )
            if visual_token_positions.shape != encoded.shape[:3]:
                raise ValueError("visual_token_positions must have shape [batch, views, visual_tokens]")
            embeddings = embeddings.clone()
            for batch_index in range(batch):
                for view_index in range(views):
                    positions = visual_token_positions[batch_index, view_index]
                    valid = positions >= 0
                    if valid.any():
                        embeddings[batch_index, positions[valid]] += encoded[batch_index, view_index, valid]
        return embeddings * attention_mask.unsqueeze(-1).to(dtype=embeddings.dtype)

    @staticmethod
    def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=prediction.dtype)
        denominator = mask.sum()
        if denominator.item() == 0:
            return prediction.sum() * 0.0
        return ((prediction - target.to(dtype=prediction.dtype)).abs() * mask).sum() / denominator

    def forward(
        self,
        role_ids: torch.Tensor,
        key_ids: torch.Tensor,
        position_ids: torch.Tensor,
        payloads: torch.Tensor,
        attention_mask: torch.Tensor,
        action_targets: torch.Tensor | None = None,
        action_target_mask: torch.Tensor | None = None,
        outcome_targets: torch.Tensor | None = None,
        outcome_target_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        visual_token_positions: torch.Tensor | None = None,
        **_: Any,
    ) -> Step2Output:
        inputs_embeds = self.embed_events(
            role_ids,
            key_ids,
            payloads,
            attention_mask,
            images,
            visual_token_positions,
        )
        hidden = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        action_predictions = torch.tanh(self.action_head(hidden))
        outcome_predictions = torch.tanh(self.outcome_head(hidden).squeeze(-1))

        action_loss = None
        outcome_loss = None
        total_loss = None
        if action_targets is not None and action_target_mask is not None:
            action_loss = self._masked_l1(action_predictions, action_targets, action_target_mask)
        if outcome_targets is not None and outcome_target_mask is not None:
            outcome_loss = self._masked_l1(outcome_predictions, outcome_targets, outcome_target_mask)
        if action_loss is not None or outcome_loss is not None:
            total_loss = hidden.sum() * 0.0
            if action_loss is not None:
                total_loss = total_loss + self.config.action_loss_weight * action_loss
            if outcome_loss is not None:
                total_loss = total_loss + self.config.outcome_loss_weight * outcome_loss

        return Step2Output(
            loss=total_loss,
            action_loss=action_loss,
            outcome_loss=outcome_loss,
            action_predictions=action_predictions,
            outcome_predictions=outcome_predictions,
        )


def parameter_report(model: Step2ForTrajectoryPrediction) -> dict[str, int]:
    groups = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "backbone": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "visual": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("visual_")
        ),
    }
    groups["adapter_and_heads"] = groups["total"] - groups["backbone"] - groups["visual"]
    return groups


def assert_selected_parameter_report(report: dict[str, int]) -> None:
    expected = {
        "total": 22_147_985,
        "trainable": 22_147_985,
        "backbone": 21_243_648,
        "visual": 890_880,
        "adapter_and_heads": 13_457,
    }
    if report != expected:
        raise AssertionError(f"selected parameterization drift: expected {expected}, got {report}")


def assert_selected_profile(config: Step2Config) -> None:
    expected = {
        "hidden_size": 384,
        "intermediate_size": 1024,
        "num_hidden_layers": 12,
        "attention_heads": 6,
        "max_position_embeddings": 2048,
        "num_roles": 10,
        "payload_dim": 8,
        "action_horizon": 16,
        "vision_resampler_tokens": 8,
    }
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        raise AssertionError(f"selected architecture drift: expected {expected}, got {actual}")
    llama = config.llama_config()
    if llama.num_key_value_heads != llama.num_attention_heads:
        raise AssertionError("GQA is not selected")
    rope_theta = getattr(llama, "rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(llama, "rope_parameters", {}).get("rope_theta")
    if rope_theta != 10_000.0 or llama.rms_norm_eps != 1.0e-5:
        raise AssertionError("RoPE/RMSNorm profile drift")
