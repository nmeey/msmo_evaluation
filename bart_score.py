import os
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
import pickle


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


def read_example_data(readfrom_folder, hash_model_folder):
    doc = ""
    summary = ""
    source_img_list = [] # list of directory to the img file
    summary_img_list = []

    with open(os.path.join(readfrom_folder, hash_model_folder, "source_doc.txt")) as file:
        doc = file.read().rstrip()

    with open(os.path.join(readfrom_folder, hash_model_folder, "summary.txt")) as file:
        summary = file.read().rstrip()

    source_img_folder = os.path.join(readfrom_folder, hash_model_folder, "source_img")
    for source_img in os.listdir(source_img_folder):
        source_img_list.append(os.path.join(source_img_folder, source_img))


    summary_img_folder = os.path.join(readfrom_folder, hash_model_folder, "summary_img")
    for summary_img in os.listdir(summary_img_folder):
        summary_img_list.append(os.path.join(summary_img_folder, summary_img))
      
    return doc, summary, source_img_list, summary_img_list

def get_the_ref(ori_folder, hash_model_folder):
    hash_only = hash_model_folder[:40]
    hash_ref = hash_only + "_ref"
    doc, summary, source_img_list, summary_img_list = read_example_data(ori_folder, hash_ref)
    return summary, summary_img_list

if __name__ == '__main__':

    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

    old_folder = "./data_pre"
    examples_folder = "./data_pre_new"

    mergedpkl_path = "./eval_results_pkl_merged"

    old_list = os.listdir(old_folder)
    example_list = os.listdir(examples_folder)
    print(len(old_list))
    print(len(example_list))
    text_bart_dic = {}
    for example in old_list:
        #print(example)
        doc, summary, source_img_list, summary_img_list = read_example_data(old_folder, example)
        bart_result_list = bart_scorer.score([summary], [doc], batch_size=4)
        text_bart_dic[example] = bart_result_list[0]

    for example in example_list:
        #print(example)
        doc, summary, source_img_list, summary_img_list = read_example_data(examples_folder, example)
        bart_result_list = bart_scorer.score([summary], [doc], batch_size=4)
        text_bart_dic[example] = bart_result_list[0]


    text_ref_bart_dic = {}
    for example in old_list:
        doc, summary, source_img_list, summary_img_list = read_example_data(old_folder, example)
        ref_summary, ref_summary_img_list = get_the_ref(old_folder, example)
        bart_result_list = bart_scorer.score([summary], [ref_summary], batch_size=4)
        text_ref_bart_dic[example] = bart_result_list[0]

    for example in example_list:
        doc, summary, source_img_list, summary_img_list = read_example_data(examples_folder, example)
        ref_summary, ref_summary_img_list = get_the_ref(old_folder, example)
        bart_result_list = bart_scorer.score([summary], [ref_summary], batch_size=4)
        text_ref_bart_dic[example] = bart_result_list[0]


    print(len(text_bart_dic))
    print(len(text_ref_bart_dic))

    with open('text_bart_dic.pkl', 'wb') as f:
        pickle.dump(text_bart_dic, f)

    with open('text_ref_bart.pkl', 'wb') as f:
        pickle.dump(text_ref_bart_dic, f)


    print("=========== success =============")
