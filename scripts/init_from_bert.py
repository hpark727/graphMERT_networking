"""
Initialize a GraphMERT model from bert-base-uncased weights.

Every-other-layer approach: BERT layers 0,2,4,6,8,10 -> GraphMERT layers 0-5.

Weights transferred:
  - Token embeddings  (atom_encoder <- word_embeddings)
  - Per-layer Q/K/V/out projections, FC1/FC2, LayerNorms

GraphMERT-specific weights (relation embeddings, H-GAT, graph token) remain
randomly initialized since BERT has no equivalent.

Usage:
  python3 scripts/init_from_bert.py \
      --bert_path /path/to/bert-base-uncased-snapshot \
      --output_path /path/to/save/initialized/checkpoint \
      --config_overrides 'max_nodes=512,vocab_size=30522,num_hidden_layers=6,num_attention_heads=12,hidden_size=768,intermediate_size=3072,pretrained_emb_dim=0,num_relationships=22'
"""

import argparse
import os
import sys

import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphmert_model import GraphMertConfig, GraphMertForMaskedLM

BERT_LAYERS = [0, 2, 4, 6, 8, 10]  # every-other-layer

LAYER_MAP = {
    "attention.self.query":        "self_attn.q_proj",
    "attention.self.key":          "self_attn.k_proj",
    "attention.self.value":        "self_attn.v_proj",
    "attention.output.dense":      "self_attn.out_proj",
    "intermediate.dense":          "fc1",
    "output.dense":                "fc2",
    "attention.output.LayerNorm":  "self_attn_layer_norm",
    "output.LayerNorm":            "final_layer_norm",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", required=True,
                        help="Local path to bert-base-uncased snapshot directory")
    parser.add_argument("--output_path", required=True,
                        help="Where to save the initialized GraphMERT checkpoint")
    parser.add_argument("--config_overrides", default="",
                        help="GraphMERT config overrides (comma-separated key=value)")
    args = parser.parse_args()

    print(f"Loading BERT from {args.bert_path}")
    bert = AutoModel.from_pretrained(args.bert_path)
    bert_sd = bert.state_dict()

    config = GraphMertConfig()
    if args.config_overrides:
        config.update_from_string(args.config_overrides)
        print(f"Config overrides applied: {args.config_overrides}")

    print(f"Creating GraphMERT: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"layers={config.num_hidden_layers}, ffn={config.intermediate_size}")
    model = GraphMertForMaskedLM(config)
    gm_sd = model.state_dict()

    transferred, skipped = 0, []

    # Token embeddings
    src = "embeddings.word_embeddings.weight"
    dst = "graphmert.graph_encoder.graph_node_feature.atom_encoder.weight"
    if src in bert_sd and dst in gm_sd:
        assert bert_sd[src].shape == gm_sd[dst].shape, \
            f"Embedding shape mismatch: {bert_sd[src].shape} vs {gm_sd[dst].shape}"
        gm_sd[dst].copy_(bert_sd[src])
        transferred += 1
        print(f"  [ok] token embeddings {bert_sd[src].shape}")
    else:
        skipped.append(src)

    # Transformer layers
    for gm_idx, bert_idx in enumerate(BERT_LAYERS[:config.num_hidden_layers]):
        for bert_suffix, gm_suffix in LAYER_MAP.items():
            for param in ("weight", "bias"):
                src = f"encoder.layer.{bert_idx}.{bert_suffix}.{param}"
                dst = f"graphmert.graph_encoder.layers.{gm_idx}.{gm_suffix}.{param}"
                if src not in bert_sd or dst not in gm_sd:
                    continue
                if bert_sd[src].shape != gm_sd[dst].shape:
                    print(f"  [shape mismatch] {src} {bert_sd[src].shape} vs {dst} {gm_sd[dst].shape}")
                    skipped.append(src)
                    continue
                gm_sd[dst].copy_(bert_sd[src])
                transferred += 1

        print(f"  [ok] layer {gm_idx} <- BERT layer {bert_idx}")

    model.load_state_dict(gm_sd)
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)

    print(f"\nTransferred {transferred} tensors.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {skipped}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
