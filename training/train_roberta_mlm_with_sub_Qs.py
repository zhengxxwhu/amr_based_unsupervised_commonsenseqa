from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import random
import numpy as np
import torch
from transformers.models.roberta.modeling_roberta import RobertaModel
from torch.utils.data import DataLoader

from tqdm import tqdm
from data_utils_with_sub_Q import (MODEL_TYPES,MODEL_CLASSES,
                        load_config_tokenizer_model,
                        count_parameters,
                        load_loss_fn,
                        load_optimizer_scheduler,
                        MLMDataset,
                        mlm_CollateFn,
                        mlm_step,
                        cls_step,
                        mlm_evaluate_dataset_acc)
from modeling import AttentionClassifier
import json

from CQA_evaluate_roberta_with_sub_Q import evaluate_CQA_datasets
from multiprocessing import Pool


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_cached_data",default=True,type=str, required=False)
    parser.add_argument("--ComQA_dev_path",default="data/commonsenseqa/dev_data_with_sub_Q.jsonl",type=str,required=False)
    parser.add_argument('--SocialiQA_dev_path', default='data/socialiqa/socialiqa-train-dev/dev_data_with_sub_Q_v2.jsonl',
                        type=str,
                        required=False)

    parser.add_argument("--output_dir", default='model_save/Q_only_based_synthesize_filtered_0.7_with_hard_candidates/roberta-mlm-margin', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default='Q_only_based_synthesize_QA_full_filtered_0.7_with_hard_candidates_roberta', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default='roberta-mlm', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            MODEL_TYPES))
    parser.add_argument("--config_name", default=None, type=str,required=False,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,required=False,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    #暂时
    #修改loss函数
    parser.add_argument("--loss_type",default='mlm_margin') #mlm_CE/mlm_margin
    #parser.add_argument("--loss_type", default='mlm_CE')  # mlm_CE/mlm_margin
    parser.add_argument("--margin", default=1.0, type=float,
                        help="The margin for ranking loss")  # param for mlm_margin loss function
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A
    #parser.add_argument("--with_sub_Q", default=False, type=bool)
    parser.add_argument("--with_sub_Q", action="store_true")
    #parser.add_argument("--eval_skip_stopwords", default=False, type=bool)
    parser.add_argument("--eval_skip_stopwords", action="store_true")
    parser.add_argument("--fix_PLM", action="store_true")

    #parser.add_argument("--candidates_num", default=3, type=int, required=False)
    parser.add_argument("--max_loop_times", default=20, type=int, required=False,help="构建候选选项的最多尝试尝试次数")
    parser.add_argument("--max_words_to_mask", default=6, type=int,
                        help="The maximum number of tokens to mask when computing scores")

    # 暂时
    #parser.add_argument('--save_steps', type=int, default=10,
    #                    help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--print_steps', type=list, default=None, required=False)
    parser.add_argument('--step_constraint', type=int, default=None, required=False)
    #暂时
    #2555
    #parser.add_argument('--seed', type=int, default=12345,
    #                    help="random seed for initialization")
    parser.add_argument('--seed', type=int, default=17293,
                        help="random seed for initialization")
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)

    parser.add_argument("--max_sequence_per_time", default=10, type=int,
                        help="The maximum number of sequences to feed into the model")
    parser.add_argument('--train_batch_size', default=1, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=1, type=int, required=False)
    #parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
    #                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_seq_length", default=90, type=int,
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

    parser.add_argument("--with_sep", action='store_true', required=False)

    args = parser.parse_args()

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer):
    cached_file = os.path.join(args.cache_dir, 'cached_{}_{}'.format(
        str(args.model_type),
        str(args.max_seq_length)))
    if args.load_cached_data and os.path.exists(cached_file):
        print("load data from cache path")
        tokenized_datas_dict = torch.load(cached_file)
        tokenized_train_datas=tokenized_datas_dict['train']
        tokenized_test_datas = tokenized_datas_dict['test']

        #暂时
        random.shuffle(tokenized_test_datas)
        tokenized_test_datas=tokenized_test_datas[:10000]
        #tokenized_test_datas = tokenized_test_datas[:5000]
    else:
        raise Exception("need to preprocess data first")

    print('max_words_to_mask is %s' % (args.max_words_to_mask))
    return MLMDataset(tokenized_train_datas, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask),\
           MLMDataset(tokenized_test_datas, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask)


def train(args, train_dataset, eval_dataset, model, cls_classifier, embedder, tokenizer, loss_fn):
    """ Train the model """
    if args.step_constraint:
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.train_batch_size,
                                      collate_fn=mlm_CollateFn)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                      collate_fn=mlm_CollateFn)
    if args.fix_PLM:
        optimizer, scheduler = load_optimizer_scheduler(args, len(train_dataloader), (embedder,args.learning_rate), (cls_classifier,1e-3))
    else:
        optimizer,scheduler=load_optimizer_scheduler(args, len(train_dataloader), (model,args.learning_rate), (cls_classifier,1e-3))

    set_seed(args)
    global_step = 0
    print_loss=0.0
    model.zero_grad()
    curr_best = 0.0
    CE = torch.nn.CrossEntropyLoss(reduction='none')
    #loss_fct = torch.nn.MultiMarginLoss(margin=args.margin)
    #loss_fct=torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.num_train_epochs):
        pbar=tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(pbar):
            if args.step_constraint and step>args.step_constraint:
                continue

            batch_input_ids, batch_input_mask, batch_input_labels, batch_cls_input_ids, batch_cls_input_mask, batch_label_ids = batch
            if len(batch_input_ids[0])==0:
                continue

            if args.with_sub_Q:
                if args.fix_PLM:
                    num_cand = len(batch_cls_input_ids[0])
                    input_ids = torch.cat([q.unsqueeze(0) for sample in batch_cls_input_ids for c in sample for q in c],
                                          dim=0).to(args.device)
                    att_mask = torch.cat([q.unsqueeze(0) for sample in batch_cls_input_mask for c in sample for q in c],
                                         dim=0).to(args.device)
                    inputs = {'input_ids': input_ids,
                              'attention_mask': att_mask}
                    outputs = embedder(**inputs)
                    #cls_embed = outputs[1]
                    cls_embed = outputs[0][:,0,:]

                    batch_cls_embed = []
                    # [batch_size,candidate_num*question_num]
                    batch_choice_seq_lens = np.array(
                        [0] + [sum([len(c) for c in sample]) for sample in batch_cls_input_ids])
                    batch_choice_seq_lens = np.cumsum(batch_choice_seq_lens)
                    for b_i in range(len(batch_choice_seq_lens) - 1):
                        start = batch_choice_seq_lens[b_i]
                        end = batch_choice_seq_lens[b_i + 1]
                        # [batch_size,(candidate_num,question_num,embedding_size)]
                        batch_cls_embed.append(cls_embed[start:end, :].view(num_cand, -1, cls_embed.shape[-1]))
                    #batch_cls_embed = cls_step(batch_cls_input_ids, batch_cls_input_mask, args, embedder)

                    torch.no_grad()
                else:
                    # batch_cls_embed [batch_size,(candidate_num,question_num,embedding_size)]
                    batch_cls_embed = cls_step(batch_cls_input_ids, batch_cls_input_mask, args, model)
                # batch_choice_loss [batch_size,(candidate_num,question_num)]
                batch_choice_loss=mlm_step(batch_input_ids, batch_input_mask, batch_input_labels, args, model, CE)

                if args.fix_PLM:
                    torch.enable_grad()

                batch_weighted_sum_l=[]
                for cls_embed,choice_loss in zip(batch_cls_embed,batch_choice_loss):
                    #(cand_num)
                    weighted_sum_l=cls_classifier(cls_embed,choice_loss)
                    batch_weighted_sum_l.append(weighted_sum_l.unsqueeze(0))
                #(batch_size,cand_num)
                batch_weighted_sum_l=torch.cat(batch_weighted_sum_l,dim=0)

                #loss = loss_fn(choice_loss, batch[3].to(args.device))
                loss = loss_fn(batch_weighted_sum_l, batch_label_ids.to(args.device))
            else:
                single_batch_input_ids=[[[c[0]]  for c in sample] for sample in batch_input_ids]
                single_batch_input_mask=[[[c[0]]  for c in sample] for sample in batch_input_mask]
                single_batch_input_labels=[[[c[0]]  for c in sample] for sample in batch_input_labels]
                # batch_choice_loss [batch_size,(candidate_num,1)]
                single_batch_choice_loss = mlm_step(single_batch_input_ids, single_batch_input_mask, single_batch_input_labels, args, model, CE)
                # (batch_size,cand_num)
                batch_choice_loss=torch.cat([choice_loss.squeeze(-1).unsqueeze(0) for choice_loss in single_batch_choice_loss])
                #batch_choice_loss=torch.tensor([[c[0] for c in sample] for sample in single_batch_choice_loss]).to(model.device)
                loss=loss_fn(batch_choice_loss, batch_label_ids.to(args.device))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            print_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #model.zero_grad()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_description('loss:{}'.format(print_loss))
                print_loss=0.0

                if (args.print_steps==None and global_step % args.save_steps == 0) or (args.print_steps and global_step in args.print_steps):
                    eval_acc = mlm_evaluate_dataset_acc(args, model, cls_classifier, embedder, eval_dataset)
                    model.eval()
                    with torch.no_grad():
                        evaluate_CQA_datasets(args, tokenizer, model, cls_classifier, embedder)

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
                        if args.with_sub_Q:
                            torch.save(cls_classifier.state_dict(),os.path.join(output_dir,'cls_classifier.bin'))
                            if embedder!=None:
                                #torch.save(cls_classifier.state_dict(), os.path.join(output_dir, 'embedder.bin'))
                                embedder.save_pretrained(os.path.join(output_dir,'embedder'))
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        print("Saving model checkpoint to {}".format(output_dir))

                    model.train()

    eval_acc = mlm_evaluate_dataset_acc(args, model, cls_classifier, embedder, eval_dataset)
    model.eval()
    with torch.no_grad():
        evaluate_CQA_datasets(args, tokenizer, model, cls_classifier, embedder)
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
        if args.with_sub_Q:
            torch.save(cls_classifier.state_dict(), os.path.join(output_dir, 'cls_classifier.bin'))
            if embedder != None:
                #torch.save(cls_classifier.state_dict(), os.path.join(output_dir, 'embedder.bin'))
                embedder.save_pretrained(os.path.join(output_dir, 'embedder'))
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        print("Saving model checkpoint to {}".format(output_dir))


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
    if args.fix_PLM:
        model.requires_grad_(False)

    if args.fix_PLM and args.with_sub_Q:
        #仅限于tokenizer为roberta的情况
        embedder=RobertaModel.from_pretrained('roberta-base',add_pooling_layer=False)
        embedder.to(args.device)
    else:
        embedder=None

    if args.with_sub_Q:
        if embedder!=None:
            cls_classifier = AttentionClassifier(embedder.config.hidden_size)
        else:
            cls_classifier=AttentionClassifier(config.hidden_size)
        cls_classifier.to(args.device)
    else:
        cls_classifier=None
    loss_fn=load_loss_fn(args)

    count = count_parameters(model)
    print("parameters count {}".format(count))

    train_dataset, test_dataset = load_and_cache_examples(args, tokenizer)
    train(args, train_dataset, test_dataset, model, cls_classifier, embedder, tokenizer, loss_fn)


if __name__ == "__main__":
    main()