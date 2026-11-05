"""Build a tiny, randomly initialised RoBERTa + tokenizer for offline smoke tests.

No network access needed. The resulting model is meaningless; it only proves the
training / evaluation code paths run end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build(out: str | Path, corpus: list[str]) -> Path:
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=800, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=bpe, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>", cls_token="<s>", sep_token="</s>",
    )
    tok.save_pretrained(out)
    cfg = RobertaConfig(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=200, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    RobertaForMaskedLM(cfg).save_pretrained(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/tiny-roberta")
    a = p.parse_args()
    corpus = ["the debate was great", "terrible speech by the candidate", "election in november"] * 50
    print(build(a.out, corpus))
