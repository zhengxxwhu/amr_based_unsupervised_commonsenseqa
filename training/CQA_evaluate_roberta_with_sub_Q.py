import json
import os
import argparse
import torch
from data_utils_with_sub_Q import (MODEL_CLASSES,
                                   #mlm_data_tokenized,
                                   MLMDataset,
                                   mlm_evaluate_dataset_acc)
#import train_roberta_mlm
from modeling import AttentionClassifier
from tokenized_data import mlm_data_tokenized
from transformers.models.roberta import RobertaModel

tokenized_fns={
    'roberta-mlm':mlm_data_tokenized,
    'albert-mlm':mlm_data_tokenized
}
DATASET_TYPES={
    'roberta-mlm':MLMDataset,
    'albert-mlm':MLMDataset
}
cal_acc_fns={
    'roberta-mlm':mlm_evaluate_dataset_acc,
    'albert-mlm':mlm_evaluate_dataset_acc
}

def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_cuda', default=True, type=bool, required=False)
    parser.add_argument('--gpu_id', default=0, type=int, required=False)
    parser.add_argument("--model_type", default='roberta-mlm', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='model_save/Q_only_based_synthesize_filtered_0.7_with_hard_candidates/roberta-mlm-margin', type=str, required=False)
    parser.add_argument('--vocab_path', default='roberta-large', type=str, required=False)
    parser.add_argument('--model_config', default='roberta-large', type=str, required=False)
    parser.add_argument('--pretrained_model', default='roberta-large', type=str,
                        required=False)

    #暂时
    parser.add_argument('--ComQA_dev_path', default='data/commonsenseqa/dev_data_with_sub_Q.jsonl', type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_path', default='data/socialiqa/socialiqa-train-dev/dev_data_with_sub_Q_v2.jsonl', type=str,
                        required=False)
    #parser.add_argument('--OpenbookQA_dev_path',default='data/openbookqa/dev.jsonl')dev_data_with_sub_Q.jsonl
    parser.add_argument('--OpenbookQA_dev_path', default='data/openbookqa/dev_data_with_sub_Q.jsonl')
    #parser.add_argument('--ComQA_dev_path', default='data/commonsenseqa/bart-base/dev_data_with_sub_Q.jsonl', type=str,
    #                    required=False)
    #parser.add_argument('--random_seed',default=12345,type=int,required=False)

    #parser.add_argument("--loss_type", default='mlm_CE')  # mlm_CE/mlm_margin
    parser.add_argument("--loss_type", default='mlm_margin')  # mlm_CE/mlm_margin
    parser.add_argument("--margin", default=1.0, type=float,
                        help="The margin for ranking loss")  # param for mlm_margin loss function
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A
    parser.add_argument("--with_sub_Q", action="store_true")
    parser.add_argument("--fix_PLM", action="store_true")
    #parser.add_argument("--eval_skip_stopwords", default=False, type=bool)
    #parser.add_argument("--eval_skip_stopwords", default=True, type=bool)
    parser.add_argument("--eval_skip_stopwords", action='store_true')
    parser.add_argument("--max_words_to_mask", default=6, type=int,
                        help="The maximum number of tokens to mask when computing scores")

    parser.add_argument('--seed', type=int, default=2555,
                        help="random seed for initialization")

    parser.add_argument("--max_sequence_per_time", default=10, type=int,
                        help="The maximum number of sequences to feed into the model")
    parser.add_argument('--eval_batch_size', default=24, type=int, required=False)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--with_sep", action='store_true', required=False)


    return parser.parse_args()


def load_tokenizer_and_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    #config = config_class.from_pretrained(args.model_name_or_path)
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    return config, tokenizer, model


def load_CQA_datasets(args):
    CQA_datasets_list=[]
    CQA_datasets_names=[]
    if args.ComQA_dev_path!='None' and args.ComQA_dev_path!=None:
        with open(args.ComQA_dev_path, "r") as ComQA_dev_file:
            ComQA_dev_raw_list = [json.loads(line) for line in ComQA_dev_file.readlines()]
            CQA_datasets_list.append(ComQA_dev_raw_list)
            CQA_datasets_names.append("ComQA")

    if args.SocialiQA_dev_path!='None' and args.SocialiQA_dev_path!=None:
        with open(args.SocialiQA_dev_path, "r") as SocialiQA_dev_file:
            SocialiQA_dev_raw_list = [json.loads(line) for line in SocialiQA_dev_file.readlines()]
            CQA_datasets_list.append(SocialiQA_dev_raw_list)
            CQA_datasets_names.append("SocialiQA")

    if args.OpenbookQA_dev_path!='None' and args.OpenbookQA_dev_path!=None:
        with open(args.OpenbookQA_dev_path,"r") as OpenbookQA_dev_file:
            OpenbookQA_dev_raw_list=[json.loads(line) for line in OpenbookQA_dev_file.readlines()]
            CQA_datasets_list.append(OpenbookQA_dev_raw_list)
            CQA_datasets_names.append("OpenbookQA")

    #return CQA_datasets_list,["ComQA","SocialiQA","OpenbookQA"]
    return CQA_datasets_list, CQA_datasets_names
    #暂时
    #return [OpenbookQA_dev_raw_list], ["OpenbookQA"]


def preprocessed_and_tokenized_CQA_datas(args,CQA_datasets_list,tokenizer):
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    preprocessed_CQA_datasets_list=[]
    for data_raw_list in CQA_datasets_list:
        preprocessed_datas=[]
        for sample in data_raw_list:
            answerKey = answerMap[sample['answerKey']]
            '''stem = 'Q: ' + " ".join(sample['question']['stem'].strip().split())
            choices = ['A: ' + c['text'][0].lower() + c['text'][1:] for c in sample['question']['choices']]
            preprocessed_datas.append([
                stem, choices, answerKey
            ])'''
            stem = " ".join(sample['question']['stem'].strip().split())
            choices = [c['text'][0].lower() + c['text'][1:] for c in sample['question']['choices']]
            stems = [stem]

            if 'sub_Q_texts' in sample.keys():
                #暂时
                compositions = [" ".join(composition.strip().split()) for composition in sample['sub_Q_texts']]
                if len(compositions) <= 1:
                    compositions = []
                compositions = list(filter(lambda x: x.endswith('?') and x.index('?') == len(x) - 1, compositions))
                #暂时
                #if len(compositions)>1 and compositions[0][:-1] in compositions[1][-1] or compositions[1][-1] in compositions[0][-1]:
                #compositions = []
                #
                stems.extend(compositions)

                #stems.extend(sample['sub_Q_texts'])
            preprocessed_datas.append([
                stems, choices, answerKey, None
            ])
        preprocessed_CQA_datasets_list.append(preprocessed_datas)

    #此处和训练时不同，应为所有位都mask计算得到
    tokenized_CQA_datasets_list = []
    for preprocessed_datas in preprocessed_CQA_datasets_list:
        tokenized_datas = tokenized_fns[args.model_type](preprocessed_datas, args, tokenizer,True,args.eval_skip_stopwords)
        tokenized_CQA_datasets_list.append(tokenized_datas)

    return tokenized_CQA_datasets_list


def evaluate_CQA_datasets(args, tokenizer, model, cls_classifier, embedder):
    accuracy_list = []
    CQA_datasets_list,datasets_names=load_CQA_datasets(args)
    tokenized_CQA_datasets_list=preprocessed_and_tokenized_CQA_datas(args, CQA_datasets_list, tokenizer)
    CQA_datasets = [DATASET_TYPES[args.model_type](dataset, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask) for dataset in
        tokenized_CQA_datasets_list]
    for dataset_name,dataset in zip(datasets_names,CQA_datasets):
        print("***** CQA dataset {} evaluation *****".format(dataset_name))
        acc = cal_acc_fns[args.model_type](args, model, cls_classifier, embedder, dataset)
        accuracy_list.append(acc)

    return accuracy_list


def main():
    args = setup_args()
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    config, tokenizer, model = load_tokenizer_and_model(args)
    model.to(args.device)

    if args.fix_PLM and args.with_sub_Q:
        #仅限于tokenizer为roberta的情况
        embedder=RobertaModel.from_pretrained(os.path.join(args.model_name_or_path,"embedder"),add_pooling_layer=False)
        embedder.to(args.device)
    else:
        embedder=None

    if args.with_sub_Q:
        if embedder != None:
            cls_classifier = AttentionClassifier(embedder.config.hidden_size)
        else:
            cls_classifier = AttentionClassifier(config.hidden_size)
        cls_classifier.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "cls_classifier.bin"),map_location='cpu'))
        cls_classifier.to(args.device)
    else:
        cls_classifier=None

    with torch.no_grad():
        model.eval()
        with torch.no_grad():
            evaluate_CQA_datasets(args, tokenizer, model, cls_classifier, embedder)


if __name__ == '__main__':
    main()