from torch.utils.data import Dataset
import torch
import numpy as np

class T5Dataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        annotations=None,  # list of lists of char-level spans [(start, end), ...] per text
        max_length=512,
        corruption_rate=0.15,
        mean_span_length=3
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        # annotations as char-based spans to highlight domain entities
        self.annotations = annotations if annotations is not None else [None] * len(texts)
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.mean_span_length = mean_span_length

    def corrupt_text(self, tokens, annotation_token_spans=None):
        # tokens: list of token strings
        # annotation_token_spans: list of (token_start, token_end) indices
        mask = self.generate_span_mask(len(tokens), annotation_token_spans)

        input_tokens = []
        target_tokens = []
        sentinel = 0
        in_span = False

        for i, token in enumerate(tokens):
            if mask[i]:
                if not in_span:
                    input_tokens.append(f"<extra_id_{sentinel}>")
                    target_tokens.append(f"<extra_id_{sentinel}>")
                    sentinel += 1
                    in_span = True
                target_tokens.append(token)
            else:
                input_tokens.append(token)
                in_span = False

        # close final span
        if in_span:
            target_tokens.append(f"<extra_id_{sentinel}>")
        else:
            input_tokens.append(f"<extra_id_{sentinel}>")
            target_tokens.append(f"<extra_id_{sentinel}>")

        # reconstruct as whitespace-joined strings
        corrupted_input = " ".join(input_tokens)
        corrupted_target = " ".join(target_tokens)
        return corrupted_input, corrupted_target

    def corrupt_ids(self, ids, annotation_token_spans):
        PAD = self.tokenizer.pad_token_id
        EOS = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else PAD
        sequence = ids[:self.max_length]

        # Build the mask (True = will be corrupted)
        mask = self.generate_span_mask(len(sequence), annotation_token_spans)

        # Skip from masking PAD/EOS
        for i,t in enumerate(sequence):
            if t in (PAD,EOS): mask[i] = False

        input_ids, target_ids = [], []
        sentinel = 0
        i = 0

        while i < len(sequence):
            if mask[i]:
                sentinel_id = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{sentinel}>")
                input_ids.append(sentinel_id)
                target_ids.append(sentinel_id)

                # Coppying the whole corrupted ids to target
                j = i
                while j < len(sequence) and mask[j]:
                    target_ids.append(sequence[j])
                    j += 1
                i = j
            else:
                input_ids.append(sequence[i])
                i += 1
        
        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]

        in_ids = input_ids
        tgt_ids = target_ids
        seq = sequence
        L = len(seq)
        if len(tgt_ids) == 0:
            # pick a non-special position near the end
            cand = [p for p,t in enumerate(seq) if t not in (PAD, EOS)]
            if cand:
                p = cand[-1]
                sid = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
                in_ids = seq[:p] + [sid] + seq[p+1: L]
                tgt_ids = [sid, seq[p]]
                in_ids  = in_ids[:self.max_length]
                tgt_ids = tgt_ids[:128]

        return in_ids, tgt_ids

        # return input_ids, target_ids

    def generate_span_mask(self, seq_len, annotation_token_spans=None, specials = None):
        # create boolean mask over tokens
        num_to_mask = max(1, int(self.corruption_rate * seq_len))
        mask = np.zeros(seq_len, dtype=bool)
        num_masked = 0

        # force mask domain spans first
        if annotation_token_spans:
            for ts, te in annotation_token_spans:
                mask[ts:te] = True
            num_masked = mask.sum()

        
        # then mask random spans until target corruption rate
        # print("NUMS MASKED IS", num_masked)
        # print("NUMS TO MASK IS", num_to_mask)
        while num_masked < num_to_mask:
            span_start = np.random.randint(0, seq_len)
            span_length = max(1, np.random.poisson(self.mean_span_length))
            span_end = min(seq_len, span_start + span_length)
            if mask[span_start:span_end].any():
                continue
            mask[span_start:span_end] = True
            num_masked += (span_end - span_start)
        
        if mask.sum() == 0: 
            cand = [i for i in range(seq_len) if i not in (specials or [])]
            if cand:
                mask[np.random.choice(cand)] = True

        return mask

    def __getitem__(self, idx):
        # print("HELLO")
        text = self.texts[idx]
        ann_char_spans = self.annotations[idx]
        # print(len(text))
        if not text.strip():
            return self.__getitem__((idx + 1) % len(self.texts))
        # first, tokenize with offsets to align char spans to token indices
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )
        offsets = encoding.pop("offset_mapping")
        ids = encoding["input_ids"]
        # tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        # ─── DEBUG START ──────────────────────────────────────────────────────────
        # print each token with its character span
        # print("=== TOKEN ⇆ OFFSET MAPPING ===")
        # for tok, (s,e) in zip(tokens, offsets):
        #     print(f"  {tok!r}: ({s},{e})")
        # print("=============================\n")
        # ─── DEBUG END ───────────────────────────────────────────────────────────

        # map char-based annotations to token-based spans
        annotation_token_spans = []
        if ann_char_spans:
            for cstart, cend in ann_char_spans:
                # find tokens whose span overlaps the char span
                token_idxs = [i for i, (s, e) in enumerate(offsets)
                              if not (s == e == 0) and e > cstart and s < cend]
                if token_idxs:
                    annotation_token_spans.append((min(token_idxs), max(token_idxs) + 1))

        # perform corruption on token strings
        corrupted_input, corrupted_target = self.corrupt_ids(ids, annotation_token_spans)
        # corrupted_input, corrupted_target = self.corrupt_text(tokens, annotation_token_spans)
        # finally, tokenize corrupted sequences for model
        # print(input_ids)
        # print(labels)
        # input_ids = self.tokenizer.encode(
        #     corrupted_input,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors="pt"
        # ).squeeze(0)
        # labels = self.tokenizer.encode(
        #     corrupted_target,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_tensors="pt"
        # ).squeeze(0)

        # return {"input_ids": input_ids, "decoder_input_ids": labels}
        # return {"input_ids": input_ids, "labels": labels}
        return {
            "input_ids": torch.tensor(corrupted_input, dtype= torch.long),
            "labels": torch.tensor(corrupted_target, dtype= torch.long),
            "attention_mask": torch.ones(len(corrupted_input), dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


def collator(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id)
    }
