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

    def generate_span_mask(self, seq_len, annotation_token_spans=None):
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
        while num_masked < num_to_mask:
            span_start = np.random.randint(0, seq_len)
            span_length = max(1, np.random.poisson(self.mean_span_length))
            span_end = min(seq_len, span_start + span_length)
            if mask[span_start:span_end].any():
                continue
            mask[span_start:span_end] = True
            num_masked += (span_end - span_start)
        return mask

    def __getitem__(self, idx):
        text = self.texts[idx]
        ann_char_spans = self.annotations[idx]
        # first, tokenize with offsets to align char spans to token indices
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )
        offsets = encoding.pop("offset_mapping")
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

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
        corrupted_input, corrupted_target = self.corrupt_text(tokens, annotation_token_spans)
        # finally, tokenize corrupted sequences for model
        input_ids = self.tokenizer.encode(
            corrupted_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).squeeze(0)
        labels = self.tokenizer.encode(
            corrupted_target,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).squeeze(0)

        return {"input_ids": input_ids, "labels": labels}

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
