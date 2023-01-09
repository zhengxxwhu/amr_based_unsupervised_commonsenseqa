"""
可拓展为使用所有情况的eval函数，当前只实现了mlm_cls的
"""
import json
import os
import argparse
import torch
from data_utils import (MODEL_CLASSES,
                                   #mlm_data_tokenized,
                        MLMDataset,
                        mlm_evaluate_dataset_acc)
#import train_roberta_mlm
from tokenized_data import mlm_data_tokenized
from transformers.models.roberta import RobertaModel
import re

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
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=False)
    parser.add_argument('--vocab_path', default='roberta-large', type=str, required=False)
    parser.add_argument('--model_config', default='roberta-large', type=str, required=False)
    parser.add_argument('--pretrained_model', default='model_save/Q_only_based_synthesize_filtered_0.8_with_hard_candidates/roberta-mlm-margin', type=str,
                        required=False)

    #暂时
    parser.add_argument('--ComQA_dev_path', default='data/commonsenseqa/dev_data.jsonl', type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_path', default='data/socialiqa/socialiqa-train-dev/dev.jsonl', type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_label_path', default='data/socialiqa/socialiqa-train-dev/dev-labels.lst', type=str,
                        required=False)

    #parser.add_argument("--loss_type", default='mlm_CE')  # mlm_CE/mlm_margin
    parser.add_argument("--loss_type", default='mlm_margin')  # mlm_CE/mlm_margin
    parser.add_argument("--margin", default=1.0, type=float,
                        help="The margin for ranking loss")  # param for mlm_margin loss function
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A
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
    config = config_class.from_pretrained(args.pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(args.vocab_path)
    model = model_class.from_pretrained(args.pretrained_model)
    return config, tokenizer, model


def preprocessed_and_tokenized_CQA_datas(args,tokenizer):
    Q_prefix="Q: "
    A_prefix=" A: "
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    tokenized_CQA_datasets_list=[]
    CQA_datasets_names=["CommonsenseQA","SocialiQA"]
    tokenized_CQA_datasets_list.append(preprocessed_and_tokenized_CommonsenseQA(args, tokenizer))
    tokenized_CQA_datasets_list.append(preprocessed_and_tokenized_SocialiQA(args,tokenizer))

    return tokenized_CQA_datasets_list,CQA_datasets_names


def preprocessed_and_tokenized_SocialiQA(args,tokenizer):
    Q_prefix = ""
    A_prefix = " "
    QUESTION_TO_ANSWER_PREFIX = {
        "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
        "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
        "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
        "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
        "What will (.*) do next?": r"[SUBJ] then",
        "How would (.*) feel after?": r"[SUBJ] then",
        "How would you describe (.*)?": r"[SUBJ] is seen as",
        "What kind of person is (.*)?": r"[SUBJ] is seen as",
        "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
        "Why did (.*) do that?": r"Before, [SUBJ] wanted",
        "Why did (.*) do this?": r"Before, [SUBJ] wanted",
        "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
        "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
        "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
        "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
        "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
        "What will happen to (.*)?": r"[SUBJ] then",
        "What will happen to (.*) next?": r"[SUBJ] then"
    }
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}

    def load_data():
        with open(args.SocialiQA_dev_path, "r") as SocialiQA_dev_file,\
                open(args.SocialiQA_dev_label_path,"r") as SocialiQA_dev_label_file:
            SocialiQA_dev_raw_list = [json.loads(line) for line in SocialiQA_dev_file.readlines()]
            SocialiQA_dev_label_list=list(map(lambda x:x.strip(),SocialiQA_dev_label_file.readlines()))
            for data,label in zip(SocialiQA_dev_raw_list,SocialiQA_dev_label_list):
                data['correct']=label
        return SocialiQA_dev_raw_list

    def to_uniform_fields(fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c[0].lower() + c[1:] for c in choices]
        choices = [c + "." if not c.endswith(".") else c for c in choices]
        #label = ord(label) - 65
        return context, question, label, choices

    def convert_choice(choice, answer_prefix):
        if answer_prefix.endswith('wanted to') and choice.startswith('wanted to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('needed to') and choice.startswith('needed to'):
            choice = choice[9:].strip()
        if answer_prefix.endswith('to') and choice.startswith('to'):
            choice = choice[2:].strip()
        choice = choice[0].lower() + choice[1:]
        return choice

    datas=load_data()

    preprocessed_datas = []
    for sample in datas:
        context, question, label, choices = to_uniform_fields(sample)
        answerKey = answerMap[label]

        answer_prefix = ""
        for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                subj = m.group(1)
                if subj.endswith('?'):
                    subj = subj[:-1]
                answer_prefix = ans_prefix.replace("[SUBJ]", subj)
                break

        if answer_prefix == "":
            answer_prefix = question.replace("?", "is")

        stem = context + ' ' + answer_prefix
        choices = [convert_choice(choice, answer_prefix) for choice in choices]

        preprocessed_datas.append([
            stem, choices, answerKey, None, Q_prefix, A_prefix
        ])

    tokenized_datas = tokenized_fns[args.model_type](preprocessed_datas, args, tokenizer, True,
                                                     args.eval_skip_stopwords)

    return tokenized_datas


def preprocessed_and_tokenized_CommonsenseQA(args,tokenizer):
    def load_data():
        with open(args.ComQA_dev_path, "r") as ComQA_dev_file:
            ComQA_dev_raw_list = [json.loads(line) for line in ComQA_dev_file.readlines()]
        return ComQA_dev_raw_list

    datas=load_data()

    Q_prefix = "Q: "
    A_prefix = " A: "
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}

    preprocessed_datas = []
    for sample in datas:
        answerKey = answerMap[sample['answerKey']]
        stem = " ".join(sample['question']['stem'].strip().split())
        if stem.endswith('.'):
            stem=stem[:-1].strip()
        if not stem.endswith('?'):
            stem=stem+'?'

        choices = [c['text'][0].lower() + c['text'][1:] for c in sample['question']['choices']]
        choices = [c.strip() if c.strip().endswith('.') else c.strip() + '.' for c in choices]

        preprocessed_datas.append([
            stem, choices, answerKey, None, Q_prefix, A_prefix
        ])

    tokenized_datas = tokenized_fns[args.model_type](preprocessed_datas, args, tokenizer, True,
                                                     args.eval_skip_stopwords)

    return tokenized_datas


def preprocessed_and_tokenized_OtherCommonQA(args,data_path,tokenizer):
    def load_data():
        with open(data_path, "r") as ComQA_dev_file:
            raw_list = [json.loads(line) for line in ComQA_dev_file.readlines()]
        return raw_list

    datas = load_data()

    q_Q_prefix = "Q: "
    q_A_prefix = " A: "
    s_Q_prefix = ""
    s_A_prefix = " "
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}

    preprocessed_datas = []
    for sample in datas:
        answerKey = answerMap[sample['answerKey']]
        stem = " ".join(sample['question']['stem'].strip().split())
        if stem.endswith('.'):
            stem = stem[:-1].strip()+' '+'is'

        choices = [c['text'][0].lower() + c['text'][1:] for c in sample['question']['choices']]
        choices = [c.strip() if c.strip().endswith('.') else c.strip() + '.' for c in choices]

        if stem.endswith('?'):
            preprocessed_datas.append([
                stem, choices, answerKey, None, q_Q_prefix, q_A_prefix
            ])
        else:
            preprocessed_datas.append([
                stem, choices, answerKey, None, s_Q_prefix, s_A_prefix
            ])

    tokenized_datas = tokenized_fns[args.model_type](preprocessed_datas, args, tokenizer, True,
                                                     args.eval_skip_stopwords)

    return tokenized_datas


def evaluate_CQA_datasets(args, tokenizer, model):
    accuracy_list = []
    #CQA_datasets_list,datasets_names=load_CQA_datasets(args)
    #tokenized_CQA_datasets_list=preprocessed_and_tokenized_CQA_datas(args, CQA_datasets_list, tokenizer)
    tokenized_CQA_datasets_list,CQA_datasets_names = preprocessed_and_tokenized_CQA_datas(args, tokenizer)
    CQA_datasets = [DATASET_TYPES[args.model_type](dataset, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask) for dataset in
        tokenized_CQA_datasets_list]
    for dataset_name,dataset in zip(CQA_datasets_names,CQA_datasets):
        print("***** CQA dataset {} evaluation *****".format(dataset_name))
        acc = cal_acc_fns[args.model_type](args, model, dataset)
        accuracy_list.append(acc)

    return accuracy_list


def main():
    args = setup_args()
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    config, tokenizer, model = load_tokenizer_and_model(args)
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        with torch.no_grad():
            evaluate_CQA_datasets(args, tokenizer, model)


if __name__ == '__main__':
    main()