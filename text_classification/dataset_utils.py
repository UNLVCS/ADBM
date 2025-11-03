from torch.utils.data import Dataset
import torch
import numpy as np


"""
 Setting up training dataset for Flan-T5 supervised finetuning.
    The dataset is of the form (input, label) where input is the instruction with the context in the form:
        ** input: "Instruction": '<question regarding the context> <context>'
        ** label: "Answer": '<yes/no answer to question>''<Justification for answer>' 
"""


"""
 Datasources: Synthetic Data Generated for enhancing classification capabilities on domain corpora
               Format:
                {"samples":[
                    {
                      "input": "Instruction: <some form of instruction about classifying the following text based on a given criteria>? Text: <Actual text to be analyzed>",
                      "output": <Yes/No>. <Justification>"
                    }
                ]}
              PubmedQA dataset.
               Format:
                {"pubid: <articleID>",
                 "question: <question that can be answered from context>",
                 "context": <article extract>",
                 "long_answer: <long answer provided to question">,
                 "final_decision": <yes/no>
                }
 Goal: Even though the two dataset slightly differ in their structure, processing is done to flatten both of them into the format specified above
       This dataset class does that.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class T5Dataset(Dataset):
    """
    A unified supervised dataset for Flan-T5 fine-tuning.
    Each example is a (input_text, target_text) pair.
    """
    def __init__(
        self,
        *,
        synthetic_data: dict = None,
        pubmed_data: list = None,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # flatten into list of (inp, out) tuples
        self.examples = []
        self.n_synt = 0
        self.n_pqa = 0

        # 1) synthetic_data: {"samples": [ {"input":..., "output":...}, ... ]}
        if synthetic_data:
            for sample in synthetic_data.get("samples", []):
                inp = sample["input"]
                out = sample["output"]
                self.examples.append((inp, out))
                self.n_synt += 1

        # 2) pubmed_data: list of dicts
        if pubmed_data:
            # for doc in pubmed_data:
            #     # handle context either as str or list of strs
            #     raw_ctx = doc.get("context", doc.get("contexts", ""))
            #     if isinstance(raw_ctx, list):
            #         context = " ".join(raw_ctx)
            #     else:
            #         context = raw_ctx

            #     inp = f"Instruction: {doc.get('question','')}  Text: {context}"
            #     ans = str(doc.get("final_decision", "")).strip().lower()
            #     just = str(doc.get("long_answer", "")).strip()
            #     out = f"Answer: {ans}. {just}"
            #     self.examples.append((inp, out))
            for doc in pubmed_data:
                # build the instruction + context
                context = "".join(doc['context']['contexts'])
                inp = f"Instruction: {doc['question']} Text: {context}"
                # yes/no answer plus the justification (long_answer)
                ans = doc["final_decision"].strip().lower()
                just = doc.get("long_answer", "").strip()
                out = f"Answer: {ans}. {just}"
                self.examples.append((inp, out))
                self.n_pqa += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]

        # tokenize *without* padding (we'll pad in the collator)
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        # squeeze off the batch dim
        input_ids = inputs["input_ids"].squeeze(0)            # (seq_len)
        labels    = targets["input_ids"].squeeze(0)           # (tgt_len)

        # ignore pad tokens in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":    input_ids,
            "labels":       labels,
        }


# def collator(batch, tokenizer):
#     """
#     Returns a collate_fn(batch) that will dynamically pad input_ids + labels,
#     rebuild attention_mask from the padded input_ids, and hand back the dict.
#     """
#     def collate_fn(batch):
#         input_ids = [ex["input_ids"] for ex in batch]
#         labels    = [ex["labels"]    for ex in batch]

#         # pad to longest in batch
#         input_ids = pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=tokenizer.pad_token_id
#         )
#         labels = pad_sequence(
#             labels,
#             batch_first=True,
#             padding_value=-100
#         )

#         attention_mask = input_ids.ne(tokenizer.pad_token_id)

#         return {
#             "input_ids":      input_ids,
#             "attention_mask": attention_mask,
#             "labels":         labels,
#         }

#     return collate_fn


import torch
from torch.nn.utils.rnn import pad_sequence

class DataCollatorForT5:
    """
    A picklable collator for T5-style supervised batches.
    Pads input_ids & labels, builds attention_mask.
    """
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        # batch is a list of dicts w/ keys "input_ids" and "labels"
        input_ids = [ex["input_ids"] for ex in batch]
        labels    = [ex["labels"]    for ex in batch]

        # pad both to the max length in this batch
        input_ids = pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(
            labels, batch_first=True,
            padding_value=self.label_pad_token_id
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }