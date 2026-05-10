from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from ivideogpt.transformer import SurgWMTrajectoryHead, load_trajectory_head, save_trajectory_head


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, output_hidden_states=False, return_dict=True):
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


def test_trajectory_head_token_positions_match_ivideogpt_layout():
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)

    assert head.context_prefix_len == 1284
    assert head.context_frame_token_slices() == [
        (0, 256),
        (256, 513),
        (513, 770),
        (770, 1027),
        (1027, 1284),
    ]
    assert head.future_sdf_positions(sequence_length=1539).tolist()[:3] == [1284, 1301, 1318]
    assert head.future_sdf_positions(sequence_length=1539).tolist()[-1] == 1522


def test_trajectory_head_forward_predicts_future_anchor_points():
    torch.manual_seed(0)
    base_model = TinyCausalLM()
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels[:, :1285] = -100
    context_trajectory = torch.rand(2, 5, 2)
    future_trajectory = torch.rand(2, 15, 2)

    outputs = head(
        base_model,
        input_ids=input_ids,
        labels=labels,
        context_trajectory_norm=context_trajectory,
        future_trajectory_norm=future_trajectory,
    )

    assert outputs.image_loss.ndim == 0
    assert outputs.trajectory_loss.ndim == 0
    assert outputs.velocity_loss.ndim == 0
    assert outputs.trajectory_pred_norm.shape == (2, 15, 2)
    assert torch.all(outputs.trajectory_pred_norm >= 0.0)
    assert torch.all(outputs.trajectory_pred_norm <= 1.0)


def test_mask_embedding_is_used_without_mask_for_ddp():
    torch.manual_seed(0)
    base_model = TinyCausalLM()
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels[:, :1285] = -100
    context_trajectory = torch.rand(2, 5, 2)
    future_trajectory = torch.rand(2, 15, 2)

    outputs = head(
        base_model,
        input_ids=input_ids,
        labels=labels,
        context_trajectory_norm=context_trajectory,
        future_trajectory_norm=future_trajectory,
    )
    (outputs.image_loss + outputs.trajectory_loss + outputs.velocity_loss).backward()

    assert head.mask_embedding.grad is not None


def test_trajectory_head_accepts_masked_context_conditions():
    torch.manual_seed(0)
    base_model = TinyCausalLM()
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)
    input_ids = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels = torch.randint(0, base_model.config.vocab_size, (2, 1539))
    labels[:, :1285] = -100
    context_trajectory = torch.rand(2, 5, 2)
    future_trajectory = torch.rand(2, 15, 2)
    context_mask = torch.tensor([
        [False, True, False, False, True],
        [True, False, False, True, False],
    ])

    outputs = head(
        base_model,
        input_ids=input_ids,
        labels=labels,
        context_trajectory_norm=context_trajectory.masked_fill(context_mask.unsqueeze(-1), 0.0),
        future_trajectory_norm=future_trajectory,
        context_trajectory_mask=context_mask,
        loss_context_trajectory_norm=context_trajectory,
    )

    assert outputs.trajectory_pred_norm.shape == (2, 15, 2)
    assert outputs.trajectory_loss.ndim == 0
    assert outputs.velocity_loss.ndim == 0


def test_trajectory_head_save_load_round_trip(tmp_path):
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)
    save_trajectory_head(tmp_path, head)

    loaded = load_trajectory_head(tmp_path)

    assert loaded.config == head.config
    assert (tmp_path / "trajectory_head.pt").exists()
    assert (tmp_path / "trajectory_head_config.json").exists()


def test_trajectory_head_loads_pre_mask_checkpoint(tmp_path):
    head = SurgWMTrajectoryHead(hidden_size=8, context_length=5, segment_length=20)
    save_trajectory_head(tmp_path, head)
    state_dict = torch.load(tmp_path / "trajectory_head.pt")
    state_dict.pop("mask_embedding")
    torch.save(state_dict, tmp_path / "trajectory_head.pt")

    loaded = load_trajectory_head(tmp_path)

    assert loaded.config == head.config
    assert loaded.mask_embedding.shape == head.mask_embedding.shape
