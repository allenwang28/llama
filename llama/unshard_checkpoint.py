

from google.cloud import storage
from io import BytesIO


def load_unsharded_model(ckpt_dir: str) -> "checkpoint":
    split_name = ckpt_dir[5:].split("/")
    bucket_name = split_name[0]
    folder_prefix = ("/").join(split_name[1:])

    blob = bucket.get_blob(f"{folder_prefix}/params.json")
    print("Loading params")
    params = json.loads(blob.download_as_text())

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
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
        print("num shards: ", num_shards)
        print("num layers: ", n_layers)
        print("n_heads: ", n_heads)
        print("n_heads_per_shard: ", n_heads_per_shard)

        checkpoint_shards = []
        for i, checkpoint in enumerate(checkpoints):
            ckpt_path = checkpoints[rank]
            blob = bucket.get_blob(ckpt_path)
            model_bytes = blob.download_as_bytes()
            print(f"loading {ckpt_path}")
            checkpoint_shards.append(
                torch.load(BytesIO(model_bytes), map_location="cpu"))
        for layer_i in range(n_layers):
            if num_shards == 1:
                # 7B, already unsharded
                checkpoint_shard = checkpoint_shards[0]
                checkpoint = {
                    f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                        checkpoint_shard[f"layers.{layer_i}.attention.wq.weight"]
                    ),
                    f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                        checkpoint_shard[f"layers.{layer_i}.attention.wk.weight"]
                    ),
                    f"model.layers.{layer_i}.self_attn.v_proj.weight": checkpoint_shard[f"layers.{layer_i}.attention.wv.weight"],
                    f"model.layers.{layer_i}.self_attn.o_proj.weight": checkpoint_shard[f"layers.{layer_i}.attention.wo.weight"],
                    f"model.layers.{layer_i}.mlp.gate_proj.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w1.weight"],
                    f"model.layers.{layer_i}.mlp.down_proj.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w2.weight"],
                    f"model.layers.{layer_i}.mlp.up_proj.weight": checkpoint_shard[f"layers.{layer_i}.feed_forward.w3.weight"],
                    f"model.layers.{layer_i}.input_layernorm.weight": checkpoint_shard[f"layers.{layer_i}.attention_norm.weight"],
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": checkpoint_shard[f"layers.{layer_i}.ffn_norm.weight"],
                }
            else:
                # Sharded
                # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
                # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
                # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

                checkpoint = {
                    f"model.layers.{layer_i}.input_layernorm.weight": checkpoint_shards[0][
                        f"layers.{layer_i}.attention_norm.weight"
                    ].clone(),
                    f"model.layers.{layer_i}.post_attention_layernorm.weight": checkpoint_shards[0][
                        f"layers.{layer_i}.ffn_norm.weight"
                    ].clone(),
                }
                checkpoint[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                    torch.cat(
                        [
                            checkpoint_shards[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(dim, dim)
                )
                checkpoint[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                    torch.cat(
                        [
                            checkpoint_shards[i][f"layers.{layer_i}.attention.wk.weight"].view(
                                num_local_key_value_heads, dims_per_head, dim
                            )
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(key_value_dim, dim),
                    num_key_value_heads,
                    key_value_dim,
                    dim,
                )
                checkpoint[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                    [
                        checkpoint_shards[i][f"layers.{layer_i}.attention.wv.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim)

                checkpoint[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
                )
                checkpoint[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
                )
                checkpoint[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
                )
                checkpoint[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                    [checkpoint_shards[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
                )
            checkpoint[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
    return params, checkpoint



