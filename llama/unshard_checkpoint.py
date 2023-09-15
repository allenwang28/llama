import torch
import json
from google.cloud import storage
from io import BytesIO


def load_unsharded_model(ckpt_dir: str) -> "checkpoint":
    split_name = ckpt_dir[5:].split("/")
    bucket_name = split_name[0]
    folder_prefix = ("/").join(split_name[1:])
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.get_blob(f"{folder_prefix}/params.json")
    print("Loading params")
    params = json.loads(blob.download_as_text())

    blobs = bucket.list_blobs(prefix=folder_prefix)

    checkpoints = [blob.name for blob in blobs if blob.name.endswith(".pth")]

    if len(checkpoints) == 0:
        print(f"no checkpoint files found in {ckpt_dir}, init model "
              "without loading checkpoint.")
        checkpoint = None
    else:
        num_shards = len(checkpoints)
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        n_heads_per_shard = n_heads // num_shards
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = params.get("rope_theta", 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
        print("DEBUG: n_heads: ", n_heads)
        print("DEBUG: n_heads_per_shard: ", n_heads_per_shard)
        print("DEBUG: dim: ", dim)
        print("DEBUG: dims_per_head: ", dims_per_head)

        if "n_kv_heads" in params:
            """
            num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
            num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
            key_value_dim = dim // num_key_value_heads
            """
            num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
            num_local_key_value_heads = num_key_value_heads // num_shards
            key_value_dim = dim // n_heads * num_key_value_heads
        else:  # compatibility with other checkpoints
            num_key_value_heads = n_heads
            num_local_key_value_heads = n_heads_per_shard
            key_value_dim = dim

        checkpoint_shards = []
        for i, checkpoint in enumerate(checkpoints):
            ckpt_path = checkpoints[i]
            blob = bucket.get_blob(ckpt_path)
            print(f"loading {ckpt_path}")
            model_bytes = blob.download_as_bytes()
            checkpoint_shards.append(
                torch.load(BytesIO(model_bytes), map_location="cpu"))

        checkpoint = {}
        if num_shards == 1:
            checkpoint.update({
                "tok_embeddings.weight": checkpoint_shards[0]["tok_embeddings.weight"],
                "norm.weight": checkpoint_shards[0]["norm.weight"],
                "output.weight": checkpoint_shards[0]["output.weight"],
                "rope.freqs": checkpoint_shards[0]["rope.freqs"],
            })
        else:
            checkpoint.update({
                "norm.weight": checkpoint_shards[0]["norm.weight"],
                "tok_embeddings.weight": torch.cat(
                    [checkpoint_shards[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
                ),
                "output.weight": torch.cat([checkpoint_shards[i]["output.weight"] for i in range(num_shards)], dim=0),
                # cat on dim 0 or 1?
                "rope.freqs": torch.cat([checkpoint_shards[i]["rope.freqs"] for i in range(num_shards)], dim=0),
            })

        for layer_i in range(n_layers):
            if num_shards == 1:
                # 7B, already unsharded
                checkpoint_shard = checkpoint_shards[0]
                checkpoint.update({
                    f"layers.{layer_i}.attention.wq.weight": checkpoint_shard[f"layers.{layer_i}.attention.wq.weight"],
                    f"layers.{layer_i}.attention.wk.weight": checkpoint_shard[f"layers.{layer_i}.attention.wk.weight"],
                    f"layers.{layer_i}.attention.wv.weight": checkpoint_shard[f"layers.{layer_i}.attention.wv.weight"],
                    f"layers.{layer_i}.attention.wo.weight": checkpoint_shard[f"layers.{layer_i}.attention.wo.weight"],
                    f"layers.{layer_i}.feed_forward.w1.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w1.weight"],
                    f"layers.{layer_i}.feed_forward.w2.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w2.weight"],
                    f"layers.{layer_i}.feed_forward.w3.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w3.weight"],
                    f"layers.{layer_i}.attention_norm.weight": checkpoint_shard[f"layers.{layer_i}.attention_norm.weight"],
                    f"layers.{layer_i}.ffn_norm.weight": checkpoint_shard[f"layers.{layer_i}.ffn_norm.weight"],
                })
            else:
                # Sharded
                checkpoint.update({
                    f"layers.{layer_i}.attention_norm.weight": checkpoint_shards[0][
                        f"layers.{layer_i}.attention_norm.weight"
                    ].clone(),
                    f"layers.{layer_i}.ffn_norm.weight": checkpoint_shards[0][
                        f"layers.{layer_i}.ffn_norm.weight"
                    ].clone(),
                })
                checkpoint[f"layers.{layer_i}.attention.wq.weight"] = torch.cat(
                        [
                            checkpoint_shards[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(dim, dim)
                checkpoint[f"layers.{layer_i}.attention.wk.weight"] = torch.cat(
                        [
                            checkpoint_shards[i][f"layers.{layer_i}.attention.wk.weight"].view(
                                num_local_key_value_heads, dims_per_head, dim
                            )
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(key_value_dim, dim)
                print("DEBUG: wk shape: ", checkpoint[f"layers.{layer_i}.attention.wk.weight"].shape)
                checkpoint[f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
                    [
                        checkpoint_shards[i][f"layers.{layer_i}.attention.wv.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim)

                checkpoint[f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
                )
                checkpoint[f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
                )
                checkpoint[f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
                )
                checkpoint[f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
                )

    for key in checkpoint.keys():
      print(f"{key}: {checkpoint[key].shape}")

    return params, checkpoint

