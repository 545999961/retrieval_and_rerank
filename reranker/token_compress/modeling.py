import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput
import torch.distributed as dist

logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)

class BiEncoderModel(nn.Module):
    def __init__(self,
                 model: None,
                 tokenizer: AutoTokenizer = None,
                 compress_method: str = 'mean',
                 train_batch_size: int = 4,
                 ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size
        self.compress_method = compress_method

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

        self.average_nums = [1, 2]


    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features, query_lengths, prompt_lengths):
        # input('continue?')
        if features is None:
            return None

        all_scores = []
        for average_num in self.average_nums:

            # print(self.tokenizer.batch_decode(features['input_ids']))

            outputs = self.model(input_ids=features['input_ids'],
                                 attention_mask=features['attention_mask'],
                                 position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                                 output_hidden_states=True,
                                 layer_sep=6,
                                 average_num=average_num,
                                 query_lengths=query_lengths,
                                 prompt_lengths=prompt_lengths,
                                 compress_method=self.compress_method)
            logits = last_logit_pool(outputs.logits, outputs.attention_masks)
            scores = logits[:, self.yes_loc]
            # if dist.get_rank() == 0:
            #     print(scores)
            all_scores.append(scores.contiguous())
        return all_scores

    def forward(self,
                pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
                query_lengths: List[int] = None,
                prompt_lengths: List[int] = None,
                teacher_scores: List[int] = None):
        all_ranker_logits = self.encode(pair, query_lengths, prompt_lengths) # (batch_size * num, dim)

        if self.training:
            loss = 0
            for ranker_logits in all_ranker_logits:
                grouped_logits = ranker_logits.view(self.train_batch_size, -1)
                target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
                loss += self.compute_loss(grouped_logits, target)

            # teacher_scores = all_ranker_logits[0].view(
            #     self.train_batch_size,
            #     -1
            # )
            teacher_scores = torch.tensor(teacher_scores, device=all_ranker_logits[0].device)
            teacher_targets = torch.softmax(teacher_scores.detach(), dim=-1)
            for logits in all_ranker_logits:
                student_scores = logits.view(
                    self.train_batch_size,
                    -1
                )
                loss += - torch.mean(torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))

        else:
            loss = None

        # print(loss)
        return RerankerOutput(
            loss=loss,
            scores=all_ranker_logits,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        # self.model.save_pretrained(output_dir)
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, **kwargs):
        return self.model.save_pretrained(**kwargs)
