from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import random
import numpy as np
import torch
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from torch.utils.data import Dataset,DataLoader
from torch.nn import Linear
import torch.nn.functional as F

from tqdm import tqdm

#from training.train_roberta_mlm_with_sub_Qs import (set_seed)
from training.data_utils import (load_optimizer_scheduler,count_parameters,MODEL_TYPES)

MODEL_CLASSES = {
    'roberta-cls': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synthesize_train_data_path",
                        default='pretrained_data/synthesize_QA_full_files/synthesize_QA_full_train.jsonl',
                        type=str, required=False,
                        help="The dev file name")
    '''parser.add_argument("--synthesize_train_data_path",
                        default='pretrained_data/synthesize_QA_full_files/synthesize_QA_full_test.jsonl',
                        type=str, required=False,
                        help="The dev file name")'''
    parser.add_argument("--output_dir", default="pretrained_data/Q_only_based_synthesize_QA_filtered",
                        required=False)
    parser.add_argument("--save_path", default='Q_only_based_synthesize_QA_full_filtered_train.jsonl', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    '''parser.add_argument("--save_path", default='Q_only_based_synthesize_QA_full_filtered_test.jsonl', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")'''

    parser.add_argument("--model_type", default='roberta-cls', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='model_save/question_filtering_Q_only/roberta-base-cls', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            MODEL_TYPES))

    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=1, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=100, type=int, required=False)
    parser.add_argument("--max_seq_length", default=110, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    args = parser.parse_args()

    return args


def preprocessed_and_tokenized_datas(args):
    preprocessed_datas=[]

    with open(args.synthesize_train_data_path,"r") as input_file:
        synthesize_datas=[json.loads(line) for line in input_file.readlines()]
        for sample in synthesize_datas:
            stem = " ".join(sample['question'].strip().split())
            preprocessed_datas.append([stem, sample])

    return preprocessed_datas


def cal_quality(args, tokenizer, model, cls_classifier, dataset):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.eval_batch_size,
                                 collate_fn=lambda x:collate_fn(args,tokenizer,x))

    print("***** Running evaluation *****")
    model.eval()
    with torch.no_grad():
        with open(os.path.join(args.output_dir,args.save_path),"w") as output_file:
            for input_tensor,samples in tqdm(dataloader):
                for key in input_tensor.keys():
                    input_tensor[key]=input_tensor[key].to(model.device)
                output = model(**input_tensor)
                logits = cls_classifier(output[1]).squeeze(-1)
                pros=F.sigmoid(logits)
                for sample,pro in zip(samples,pros):
                    sample['quality']=round(pro.item(),2)
                    output_file.write(json.dumps(sample)+"\n")

    #return eval_acc


def collate_fn(args,tokenizer:RobertaTokenizer,batch):
    input_seq=[]
    samples=[]
    for example in batch:
        stem,sample=example
        input_seq.append(stem)
        samples.append(sample)
    input_ids=tokenizer(input_seq,padding=True,truncation=True,max_length=args.max_seq_length,return_tensors='pt')
    return input_ids,samples


class MyDataset(Dataset):
    def __init__(self,datas):
        self.datas=datas

    def __getitem__(self, item):
        return self.datas[item]

    def __len__(self):
        return len(self.datas)


def load_config_tokenizer_model(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    return config,tokenizer,model


def main_synthesize():
    args=set_args()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    #set_seed(args)

    config,tokenizer,model=load_config_tokenizer_model(args)
    model.to(args.device)
    cls_classifier=Linear(config.hidden_size,1)
    cls_classifier.load_state_dict(torch.load(os.path.join(args.model_name_or_path,'cls_classifier.bin')))
    cls_classifier.to(args.device)

    count = count_parameters(model)
    print("parameters count {}".format(count))

    all_datas = preprocessed_and_tokenized_datas(args)
    #all_datas=preprocessed_and_tokenized_ATOMIC_datas(args)
    dataset=MyDataset(all_datas)
    cal_quality(args, tokenizer, model, cls_classifier, dataset)


if __name__ == "__main__":
    main_synthesize()