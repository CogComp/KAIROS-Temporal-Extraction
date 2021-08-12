import torch
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)
from torch.utils.data.sampler import SequentialSampler
from transformers.data.data_collator import DataCollator
from torch.utils.data.dataset import Dataset
from typing import Dict
from dataclasses import dataclass
import math


@dataclass
class CustomDataCollator(DataCollator):
    def collate_batch(self, features) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        lm_labels = torch.tensor([f['lm_labels'] for f in features], dtype=torch.long)
        decoder_attention_mask = []
        decoder_inputs = []
        for i in range(0, len(features)):
            decoder_inputs.append([0, 1525, 10] + [0] * (input_ids.size()[1] - 3))
            decoder_attention_mask.append([1, 1, 1, 1, 1] + [0] * (input_ids.size()[1] - 5))
        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
        decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_input_ids": decoder_inputs,
            "decoder_attention_mask": decoder_attention_mask,
        }


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, lines):

        originals = []
        labels = []
        for l in lines:
            originals.append(l.split("\t")[0])
            labels.append(l.split("\t")[1])

        self.inputs = tokenizer.batch_encode_plus(originals, pad_to_max_length=True)
        self.labels = tokenizer.batch_encode_plus(labels, pad_to_max_length=True)

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, i):
        source_ids = self.inputs["input_ids"][i]
        target_ids = self.labels["input_ids"][i]
        src_mask = self.inputs["attention_mask"][i]
        target_mask = self.labels["attention_mask"][i]
        return {"input_ids": source_ids, "attention_mask": src_mask, "lm_labels": target_ids, "decoder_attention_mask": target_mask}


def get_dataset(lines, tokenizer):
    ret = LineByLineTextDataset(tokenizer=tokenizer, lines=lines)
    return ret


class RelationOnlyPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = T5ForConditionalGeneration.from_pretrained(
            "models/final_matres"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.data_collator = CustomDataCollator()

    def softmax(self, a_list):
        a_sum = 0.0
        for a in a_list:
            a_sum += math.exp(a)
        return [math.exp(x) / a_sum for x in a_list]

    # Input: a list of lines
    # Return: a list of [pos, neg] probabilities
    def predict(self, lines, query_type="order"):
        eval_dataset = get_dataset(lines, self.tokenizer)
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=8,
            collate_fn=self.data_collator.collate_batch,
        )
        ret = []
        for inputs in data_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=inputs['decoder_input_ids'],
                    decoder_attention_mask=inputs['decoder_attention_mask'],
                )[0].cpu().numpy()
                for output in outputs:
                    ret.append(self.softmax([output[2][1465], output[2][2841]]))
        return ret


class Predictor:

    def __init__(self, config_lines=None):
        if config_lines is None:
            config_lines = [x.strip() for x in open("configs/model_paths.txt").readlines()]
        config_map = {}
        for l in config_lines:
            config_map[l.split("\t")[0]] = l.split("\t")[1]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = ""
        self.model = T5ForConditionalGeneration.from_pretrained(
            config_map['order_model']
        ).to(self.device)
        self.model_distance = T5ForConditionalGeneration.from_pretrained(
            config_map['distance_model']
        ).to(self.device)
        self.model_duration = T5ForConditionalGeneration.from_pretrained(
            config_map['duration_model']
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.model_distance.resize_token_embeddings(len(self.tokenizer))
        self.model_distance.eval()
        self.model_duration.resize_token_embeddings(len(self.tokenizer))
        self.model_duration.eval()
        self.data_collator = CustomDataCollator()

    def softmax(self, a_list):
        a_sum = 0.0
        for a in a_list:
            a_sum += math.exp(a)
        return [math.exp(x) / a_sum for x in a_list]

    # Input: a list of lines
    # Return: a list of [pos, neg] probabilities
    def predict(self, lines, query_type="order"):
        if len(lines) == 0:
            return None
        eval_dataset = get_dataset(lines, self.tokenizer)
        sampler = SequentialSampler(eval_dataset)
        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=4,
            collate_fn=self.data_collator.collate_batch,
        )
        ret = []
        for inputs in data_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)
            with torch.no_grad():
                if query_type == "order":
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=inputs['decoder_input_ids'],
                        decoder_attention_mask=inputs['decoder_attention_mask'],
                    )[0].cpu().numpy()
                    for output in outputs:
                        ret.append(self.softmax([output[2][1465], output[2][2841]]))
                elif query_type == "distance":
                    outputs = self.model_distance(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=inputs['decoder_input_ids'],
                        decoder_attention_mask=inputs['decoder_attention_mask'],
                    )[0].cpu().numpy()
                    for output in outputs:
                        arr = []
                        for val in [32000, 32001, 32002, 32003, 32004, 32005, 32006]:
                            arr.append(output[3][val])
                        ret.append(self.softmax(arr))
                else:
                    outputs = self.model_duration(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_input_ids=inputs['decoder_input_ids'],
                        decoder_attention_mask=inputs['decoder_attention_mask'],
                    )[0].cpu().numpy()
                    for output in outputs:
                        arr = []
                        for val in [32000, 32001, 32002, 32003, 32004, 32005, 32006]:
                            arr.append(output[2][val])
                        ret.append(self.softmax(arr))
        return ret


if __name__ == "__main__":
    p = Predictor()
    r = p.predict([
        "event: i went home starts before i left home story: \tnothing\n",
        "event: i left home starts before i get back home story: \tnothing\n",
    ])
    print(r)

