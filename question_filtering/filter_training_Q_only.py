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

from training.data_utils import (load_optimizer_scheduler,count_parameters,MODEL_TYPES)

MODEL_CLASSES = {
    'roberta-cls': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--synthesize_dev_data_path",
                        default='pretrained_data/synthesize_QA_full_files/synthesize_QA_full_test.jsonl',
                        type=str, required=False,
                        help="The dev file name")
    parser.add_argument("--ComQA_train_path",default="data/commonsenseqa/train_data.jsonl",type=str,required=False)
    parser.add_argument('--SocialiQA_train_path', default='data/socialiqa/socialiqa-train-dev/train.jsonl',
                        type=str,
                        required=False)

    parser.add_argument("--output_dir", default='model_save/question_filtering_Q_only/roberta-base-cls', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default='roberta-cls', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            MODEL_TYPES))

    parser.add_argument("--loss_type",default='CE_margin') #mlm_CE/mlm_margin/binary_CE/CE_margin
    parser.add_argument("--margin",default=1.0)

    # 暂时
    #parser.add_argument('--save_steps', type=int, default=10,
    #                    help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    #暂时
    #2555
    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)

    #parser.add_argument('--balanced_train_data',action='store_true')
    parser.add_argument('--balanced_train_data', default=True,type=bool,required=False)
    parser.add_argument('--resample_balanced_train_data', action='store_true', required=False)
    parser.add_argument('--max_data_num_per_set', default=20000 ,type=int, required=False)


    parser.add_argument('--train_batch_size', default=50, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=50, type=int, required=False)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_seq_length", default=110, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Linear warmup over warmup proportion.")

    args = parser.parse_args()

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def preprocessed_and_tokenized_filter_datas(args):
    positive_preprocessed_datas=[]
    negative_preprocessed_datas=[]

    if args.ComQA_train_path!="None":
        with open(args.ComQA_train_path,"r") as input_file:
            ComQA_datas=[json.loads(line) for line in input_file.readlines()]
            for sample in ComQA_datas:
                stem = " ".join(sample['question']['stem'].strip().split())
                positive_preprocessed_datas.append([stem, 1])
    if args.OpenbookQA_train_path!="None":
        with open(args.OpenbookQA_train_path,"r") as input_file:
            OpenbookQA_datas=[json.loads(line) for line in input_file.readlines()]
            for sample in OpenbookQA_datas:
                stem = " ".join(sample['question']['stem'].strip().split())
                if stem.endswith('?'):
                    positive_preprocessed_datas.append([stem, 1])
    if args.SocialiQA_train_path!="None":
        with open(args.SocialiQA_train_path,"r") as input_file:
            SocialiQA_datas=[json.loads(line) for line in input_file.readlines()]
            for sample in SocialiQA_datas:
                stem = " ".join((sample['context'].strip()+" "+sample['question'].strip()).split())
                positive_preprocessed_datas.append([stem, 1])

    with open(args.synthesize_dev_data_path,"r") as input_file:
        synthesize_datas=[json.loads(line) for line in input_file.readlines()]
        if args.max_data_num_per_set>0:
            random.shuffle(synthesize_datas)
            synthesize_datas=synthesize_datas[:args.max_data_num_per_set]
        for sample in synthesize_datas:
            stem = " ".join(sample['question'].strip().split())
            negative_preprocessed_datas.append([stem, 0])

    if args.ATOMIC_dev_data_path!="None":
        with open(args.ATOMIC_dev_data_path, "r") as input_file:
            ATOMIC_datas = [json.loads(line) for line in input_file.readlines()]
            if args.max_data_num_per_set > 0:
                random.shuffle(ATOMIC_datas)
                ATOMIC_datas = ATOMIC_datas[:args.max_data_num_per_set]
            for sample in ATOMIC_datas:
                stem = " ".join(sample['context'].replace("___","").strip().split())
                negative_preprocessed_datas.append([stem, 0])
    if args.CWWV_dev_data_path!="None":
        with open(args.CWWV_dev_data_path, "r") as input_file:
            CWWV_datas = [json.loads(line) for line in input_file.readlines()]
            if args.max_data_num_per_set > 0:
                random.shuffle(CWWV_datas)
                CWWV_datas = CWWV_datas[:args.max_data_num_per_set]
            for sample in CWWV_datas:
                stem = " ".join(sample['question']['stem'].replace("___","").strip().split())
                negative_preprocessed_datas.append([stem, 0])

    return positive_preprocessed_datas, negative_preprocessed_datas


def evaluate(args, tokenizer, model, cls_classifier, eval_dataset):
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size,
                                 collate_fn=lambda x:collate_fn(args,tokenizer,x))

    print("***** Running evaluation *****")
    preds = []
    gold_label_ids = []
    model.eval()
    with torch.no_grad():
        for input_tensor,labels_ids in tqdm(eval_dataloader):
            #batch_cls_input_ids, batch_cls_input_mask, batch_label_ids = batch
            #input_tensor = {'input_ids': batch_cls_input_ids, 'attention_mask': batch_cls_input_mask}
            labels_list=[]
            for one_index in labels_ids:
                label=torch.zeros(2,dtype=torch.long)
                label[one_index]=1.
                labels_list.append(label)
            labels_ids=torch.cat(labels_list,dim=0)
            for key in input_tensor.keys():
                input_tensor[key]=input_tensor[key].to(model.device)
            output = model(**input_tensor)
            # (batch_size,1)
            logits = cls_classifier(output[1]).squeeze(-1)
            pro=F.sigmoid(logits)
            preds.append(pro)
            gold_label_ids.append(labels_ids.numpy())
        preds = torch.cat(preds, dim=0).cpu().numpy()
        eval_acc = ((preds>0.5) == np.concatenate(gold_label_ids, axis=0)).mean()
        print('eval acc:{}'.format(eval_acc))

    return eval_acc


def collate_fn(args,tokenizer:RobertaTokenizer,batch):
    input_seq=[]
    labels=[]
    for example in batch:
        stems,tags=example
        input_seq.extend([stem for stem in stems])
        labels.append(tags.index(1))
    input_ids=tokenizer(input_seq,padding=True,truncation=True,max_length=args.max_seq_length,return_tensors='pt')
    labels=torch.tensor(labels,dtype=torch.long)
    return input_ids,labels


def train(args, train_dataset, eval_dataset, model, cls_classifier, tokenizer, loss_fn):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  collate_fn=lambda x:collate_fn(args,tokenizer,x))
    optimizer,scheduler=load_optimizer_scheduler(args, len(train_dataloader), (model,args.learning_rate), (cls_classifier,args.learning_rate))

    #set_seed(args)
    global_step = 0
    print_loss=0.0
    model.zero_grad()
    curr_best = 0.0
    model.train()
    for _ in range(args.num_train_epochs):
        pbar=tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(pbar):
            input_tensor,labels_ids = batch
            for key in input_tensor.keys():
                input_tensor[key]=input_tensor[key].to(model.device)
            output=model(**input_tensor)
            #(batch_size,1)
            logits=cls_classifier(output[1]).view(-1,2)
            loss = loss_fn(logits, labels_ids.to(args.device))


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            print_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                pbar.set_description('loss:{}'.format(print_loss))
                print_loss=0.0

                if global_step % args.save_steps == 0:
                    eval_acc = evaluate(args, tokenizer, model, cls_classifier, eval_dataset)
                    model.eval()

                    if eval_acc > curr_best:
                        curr_best = eval_acc
                        # Save model checkpoint
                        output_dir = args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        #暂时
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(cls_classifier.state_dict(),os.path.join(output_dir,'cls_classifier.bin'))
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        print("Saving model checkpoint to {}".format(output_dir))

                    model.train()

    eval_acc = evaluate(args, tokenizer, model, cls_classifier, eval_dataset)
    model.eval()
    if eval_acc > curr_best:
        # Save model checkpoint
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        #暂时
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(cls_classifier.state_dict(), os.path.join(output_dir, 'cls_classifier.bin'))
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        print("Saving model checkpoint to {}".format(output_dir))


class MyDataset(Dataset):
    def __init__(self,positive_datas,negative_datas):
        self.positive_datas=positive_datas
        self.negative_datas = negative_datas

    def __getitem__(self, item):
        p_data=self.positive_datas[item]
        n_data=self.negative_datas[item]
        if random.random()<0.5:
            data=[[p_item,n_item] for p_item,n_item in zip(p_data,n_data)]
        else:
            data = [[n_item,p_item] for n_item,p_item in zip(n_data,p_data)]
        return data

    def __len__(self):
        return len(self.positive_datas)


def load_config_tokenizer_model(args):
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    return config,tokenizer,model


def load_loss_fn(args):
    loss_fns={
        'binary_CE': torch.nn.BCELoss(),
        'CE_margin': torch.nn.MultiMarginLoss(margin=args.margin)
    }
    return loss_fns[args.loss_type]


def main():
    args=set_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    config,tokenizer,model=load_config_tokenizer_model(args)
    model.to(args.device)
    cls_classifier=Linear(config.hidden_size,1)
    cls_classifier.to(args.device)
    loss_fn=load_loss_fn(args)

    count = count_parameters(model)
    print("parameters count {}".format(count))

    positive_preprocessed_datas, negative_preprocessed_datas = preprocessed_and_tokenized_filter_datas(args)
    random.shuffle(positive_preprocessed_datas)
    random.shuffle(negative_preprocessed_datas)

    #平衡正例和负利数量？
    if args.resample_balanced_train_data:
        data_len = max(len(positive_preprocessed_datas), len(negative_preprocessed_datas))
        while len(positive_preprocessed_datas)<data_len:
            if data_len>len(positive_preprocessed_datas)*2:
                positive_preprocessed_datas.extend(positive_preprocessed_datas)
            else:
                positive_preprocessed_datas.extend(random.sample(positive_preprocessed_datas,data_len-len(positive_preprocessed_datas)))
        if len(negative_preprocessed_datas)<data_len:
            if data_len > len(negative_preprocessed_datas) * 2:
                negative_preprocessed_datas.extend(negative_preprocessed_datas)
            else:
                negative_preprocessed_datas.extend(random.sample(negative_preprocessed_datas,data_len-len(negative_preprocessed_datas)))
        #positive_preprocessed_datas = positive_preprocessed_datas[:data_len]
        #negative_preprocessed_datas = negative_preprocessed_datas[:data_len]
    elif args.balanced_train_data:
        data_len=min(len(positive_preprocessed_datas),len(negative_preprocessed_datas))
        positive_preprocessed_datas=positive_preprocessed_datas[:data_len]
        negative_preprocessed_datas=negative_preprocessed_datas[:data_len]


    positive_train_len=int(len(positive_preprocessed_datas)*0.8)
    positive_train_datas=positive_preprocessed_datas[:positive_train_len]
    positive_test_datas=positive_preprocessed_datas[positive_train_len:]

    negative_train_len = int(len(negative_preprocessed_datas) * 0.8)
    negative_train_datas = negative_preprocessed_datas[:negative_train_len]
    negative_test_datas = negative_preprocessed_datas[negative_train_len:]

    train_dataset = MyDataset(positive_train_datas,negative_train_datas)
    test_dataset = MyDataset(positive_test_datas,negative_test_datas)

    train(args, train_dataset, test_dataset, model, cls_classifier, tokenizer, loss_fn)


if __name__ == "__main__":
    main()