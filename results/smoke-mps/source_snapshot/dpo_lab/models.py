"""Full-parameter causal models: no quantization or adapter changes to DPO."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from dpo_lab.runtime import seed_everything


def build_fixture_model(rows, path, seed=0):
    """A randomly initialized local transformer, only for engineering validation."""
    seed_everything(seed)
    backend = Tokenizer(WordLevel(unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]'])
    backend.train_from_iterator((r[k] for r in rows for k in ('prompt', 'chosen', 'rejected')), trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=256, n_embd=48, n_layer=2, n_head=4,
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    GPT2LMHeadModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_model(path, device, revision=None, checkpointing=False):
    model = AutoModelForCausalLM.from_pretrained(str(path), revision=revision, torch_dtype=torch.float32,
                                               attn_implementation='eager').to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def load_tokenizer(path, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(str(path), revision=revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None: raise ValueError('An EOS token is required')
    return tokenizer


def save_model(model, tokenizer, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    # State tensors are copied to CPU without moving the active model/optimizer.
    model.save_pretrained(path, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})
    tokenizer.save_pretrained(path)
