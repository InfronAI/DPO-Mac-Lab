import json
from pathlib import Path
import pytest
import torch
from dpo_lab.pipeline import mark_complete,is_complete,validate_config
from dpo_lab.runtime import select_device


def test_completion_receipt_detects_changed_checkpoint(tmp_path):
    (tmp_path/'weights').write_bytes(b'checkpoint')
    mark_complete(tmp_path)
    assert is_complete(tmp_path)
    (tmp_path/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Corrupt'): is_complete(tmp_path)


def test_device_validation_and_config():
    assert select_device('cpu').type=='cpu'
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):select_device('cuda')
    cfg=json.loads(Path('experiments/smoke.json').read_text())
    validate_config(cfg)
    cfg['max_new_tokens']=999
    with pytest.raises(ValueError):validate_config(cfg)
