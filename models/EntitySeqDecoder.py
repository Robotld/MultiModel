import torch
import torch.nn as nn
import torch.nn.functional as F


class EntitySeqDecoder(nn.Module):
    """
    实体序列生成 Decoder (安全版)
    实体 id: 0..num_entities-1
    PAD: num_entities
    BOS: num_entities+1
    EOS: num_entities+2
    vocab_size = num_entities + 3
    """

    def __init__(self,
                 d_model: int = 384,
                 num_layers: int = 3,
                 num_heads: int = 6,
                 dim_feedforward: int = 1024,
                 num_entities: int = 24,   # 实体类别数(不含BOS/EOS/PAD)
                 dropout: float = 0.1,
                 max_seq_len: int = 10):
        super().__init__()

        # 实体 id 范围: [0, num_entities-1]
        # 特殊 token:
        self.PAD_ID = num_entities          # padding
        self.BOS_ID = num_entities + 1      # 序列开始
        self.EOS_ID = num_entities + 2      # 序列结束

        self.vocab_size = num_entities + 3  # 实体 + PAD + BOS + EOS
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Embedding
        self.entity_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.PAD_ID
        )
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

        print(f"✅ EntitySeqDecoder 初始化: num_entities={num_entities}, "
              f"PAD={self.PAD_ID}, BOS={self.BOS_ID}, EOS={self.EOS_ID}, vocab={self.vocab_size}")

    def _generate_square_subsequent_mask(self, sz: int, device):
        # shape [sz, sz], 上三角为 -inf
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self,
                encoder_context,
                target_entity_ids=None,
                context_mask=None):
        """
        encoder_context: [B, N_ctx, D]
        target_entity_ids: [B, K]，每个位置是实体 id (0..num_entities-1) 或 -1(pad)
        """
        device = encoder_context.device
        B, N_ctx, D = encoder_context.shape

        # ====== 训练模式: teacher forcing ======
        if target_entity_ids is not None:
            batch_dec_inputs = []
            batch_labels = []
            max_len = 0

            for b in range(B):
                ent_ids = target_entity_ids[b]  # [K]
                # 过滤 -1，保留真实实体 id
                valid = ent_ids[ent_ids >= 0].tolist()

                if len(valid) == 0:
                    # 无实体: BOS -> EOS
                    dec_in = [self.BOS_ID]
                    lab = [self.EOS_ID]
                else:
                    # 实体 id 必须在 [0, num_entities-1]
                    # 若你后续修改过 entity2id，这里可加断言:
                    # assert all(0 <= x < self.PAD_ID for x in valid)
                    dec_in = [self.BOS_ID] + valid
                    lab = valid + [self.EOS_ID]

                batch_dec_inputs.append(dec_in)
                batch_labels.append(lab)
                max_len = max(max_len, len(dec_in), len(lab))

            # padding 到 max_len
            dec_input_ids = torch.full((B, max_len), self.PAD_ID, dtype=torch.long, device=device)
            label_ids = torch.full((B, max_len), self.PAD_ID, dtype=torch.long, device=device)

            for b in range(B):
                di = batch_dec_inputs[b]
                lb = batch_labels[b]
                dec_input_ids[b, :len(di)] = torch.tensor(di, dtype=torch.long, device=device)
                label_ids[b, :len(lb)] = torch.tensor(lb, dtype=torch.long, device=device)

            L_dec = dec_input_ids.size(1)
            pos_ids = torch.arange(L_dec, device=device).unsqueeze(0).expand(B, -1)

            dec_embeds = self.entity_embedding(dec_input_ids)   # [B, L_dec, D]
            pos_embeds = self.pos_embedding(pos_ids)            # [B, L_dec, D]
            dec_embeds = self.dropout(dec_embeds + pos_embeds)

            tgt_mask = self._generate_square_subsequent_mask(L_dec, device=device)

            decoder_out = self.transformer_decoder(
                tgt=dec_embeds,
                memory=encoder_context,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=context_mask  # 可为 None
            )  # [B, L_dec, D]

            logits = self.output_projection(decoder_out)  # [B, L_dec, vocab]

            # 调试信息：检查 label 范围
            # （debug 时先用 CPU 跑几步，或者打印一下）
            if torch.any(label_ids < 0) or torch.any(label_ids >= self.vocab_size):
                print("❌ label_ids 越界: min=", label_ids.min().item(),
                      "max=", label_ids.max().item(), "vocab=", self.vocab_size)
                raise RuntimeError("label_ids out of range")

            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                label_ids.view(-1),
                ignore_index=self.PAD_ID
            )
            return logits, loss

        # ====== 推理模式 ======
        else:
            generated = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)  # [B, 1]

            for step in range(self.max_seq_len):
                L_dec = generated.size(1)
                pos_ids = torch.arange(L_dec, device=device).unsqueeze(0).expand(B, -1)

                dec_embeds = self.entity_embedding(generated)
                pos_embeds = self.pos_embedding(pos_ids)
                dec_embeds = self.dropout(dec_embeds + pos_embeds)

                tgt_mask = self._generate_square_subsequent_mask(L_dec, device=device)

                decoder_out = self.transformer_decoder(
                    tgt=dec_embeds,
                    memory=encoder_context,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=context_mask
                )
                logits = self.output_projection(decoder_out)  # [B, L_dec, vocab]
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == self.EOS_ID).all():
                    break

            return generated, None