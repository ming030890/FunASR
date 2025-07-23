import torch
import torch.nn as nn
from funasr.models.sense_voice.model import SenseVoiceSmall
from funasr.tokenizer.char_tokenizer import CharTokenizer
from funasr.register import tables
from pathlib import Path

class DummyEncoder(nn.Module):
    def __init__(self, input_size, output_size=5, **kwargs):
        super().__init__()
        self._output_size = output_size
    def output_size(self):
        return self._output_size
    def forward(self, x, lengths):
        batch = x.size(0)
        T = 2
        return torch.zeros(batch, T, self._output_size), torch.tensor([T]*batch)

tables.encoder_classes['DummyEncoder'] = DummyEncoder

def test_inference_shallow_fusion(tmp_path):
    arpa_src = Path('runtime/onnxruntime/third_party/kaldi/lm/test_data/input.arpa')
    lm_path = tmp_path / 'small.arpa'
    lm_path.write_text(arpa_src.read_text())

    tokens = ["<blank>", "<s>", "</s>", "a", "b"]
    tokenizer = CharTokenizer(token_list=tokens, unk_symbol="<blank>")

    model = SenseVoiceSmall(
        specaug=None,
        normalize=None,
        encoder='DummyEncoder',
        encoder_conf={'input_size': len(tokens), 'output_size': len(tokens)},
        ctc_conf={'dropout_rate': 0.0},
        input_size=len(tokens),
        vocab_size=len(tokens),
        blank_id=0,
    )

    log_probs = torch.log_softmax(
        torch.tensor([[[0.1, 0.0, 0.0, 5.0, -2.0], [0.1, 0.0, 0.0, -2.0, 5.0]]]),
        dim=-1,
    )
    model.encoder = lambda x, y: (torch.zeros(1, 2, len(tokens)), torch.tensor([2]))
    model.ctc.log_softmax = lambda x: log_probs

    out, _ = model.inference(
        torch.zeros(1, 2, len(tokens)),
        data_lengths=torch.tensor([2]),
        tokenizer=tokenizer,
        device='cpu',
        data_type='fbank',
        beam_size=2,
        lm_file=str(lm_path),
        lm_weight=0.5,
    )
    assert out[0]['text'] == 'ab'
