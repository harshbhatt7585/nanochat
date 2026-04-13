import torch

from nanochat.gpt import GPT, GPTConfig


def _small_config(**overrides):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_iha_forward_backward_cpu():
    torch.manual_seed(0)
    model = GPT(_small_config(attention_type="iha", iha_num_pseudo_heads=2))
    model.init_weights()

    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(idx, targets)
    loss.backward()

    assert loss.ndim == 0
    alpha_grad = model.transformer.h[0].attn.alpha_q.grad
    assert alpha_grad is not None
    assert torch.isfinite(alpha_grad).all()


def test_iha_matches_mha_with_single_pseudo_head():
    torch.manual_seed(1)
    mha = GPT(_small_config(attention_type="mha"))
    iha = GPT(_small_config(attention_type="iha", iha_num_pseudo_heads=1))
    mha.init_weights()
    iha.init_weights()

    iha_state = iha.state_dict()
    for name, tensor in mha.state_dict().items():
        if name in iha_state and iha_state[name].shape == tensor.shape:
            iha_state[name].copy_(tensor)
    iha.load_state_dict(iha_state, strict=False)

    idx = torch.randint(0, mha.config.vocab_size, (2, 8))
    mha.eval()
    iha.eval()
    with torch.no_grad():
        mha_logits = mha(idx)
        iha_logits = iha(idx)

    torch.testing.assert_close(mha_logits, iha_logits, rtol=1e-5, atol=1e-5)


def test_iha_non_matrix_params_stay_off_muon():
    model = GPT(_small_config(attention_type="iha", iha_num_pseudo_heads=2))
    optimizer = model.setup_optimizer()

    muon_params = []
    adamw_params = []
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            muon_params.extend(group["params"])
        else:
            adamw_params.extend(group["params"])

    assert all(param.ndim == 2 for param in muon_params)
    assert any(param is model.transformer.h[0].attn.alpha_q for param in adamw_params)
