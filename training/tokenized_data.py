import argparse
import json
import os
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
from multiprocessing import Pool
import math
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
import torch
import random
#from sentence_transformers import SentenceTransformer, util
from generate_candidates import generate_hard_candidates
import re


def set_args():
    parser = argparse.ArgumentParser()
    #暂时
    parser.add_argument('--declarative_sent',action='store_true')
    parser.add_argument("--num_workers",default=20,type=int,required=False)
    #parser.add_argument("--multi_processing",default=False,required=False)
    parser.add_argument("--multi_processing", action='store_true', required=False)

    parser.add_argument("--model_type", default='roberta-mlm', type=str, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=False)
    parser.add_argument("--max_seq_length", default=90, type=int)
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A
    #parser.add_argument("--with_sub_Q", default=True,type=bool)

    '''parser.add_argument("--tokenize_synthesize_QA",default=True,required=False)
    parser.add_argument("--tokenize_ATOMIC", default=False, required=False)
    parser.add_argument("--tokenize_CWWV", default=False, required=False)'''
    parser.add_argument("--tokenize_synthesize_QA", action="store_true")
    parser.add_argument("--tokenize_ATOMIC", action="store_true")
    parser.add_argument("--tokenize_CWWV", action="store_true")
    parser.add_argument("--tokenize_ComQA", action="store_true")
    parser.add_argument("--tokenize_SocialiQA", action="store_true")

    parser.add_argument("--hard_candidates",action='store_true')
    parser.add_argument("--no_rel_type", action='store_true')
    '''parser.add_argument("--candidates_source_path",
                        default="pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.3.jsonl",
                        required=False)'''
    parser.add_argument("--sentence_embedding_model_name",default='all-mpnet-base-v2')
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)
    #parser.add_argument("--batch_size", default=120, type=int, required=False)

    parser.add_argument("--train_sub_Qs_data_path",
                        default='pretrained_data/synthesize_QA_filtered/synthesize_QA_full_train_high_quality.jsonl',
                        type=str, required=False,
                        help="The train file name")
    parser.add_argument("--dev_sub_Qs_data_path",
                        default='pretrained_data/synthesize_QA_full_files/synthesize_QA_full_test.jsonl',
                        type=str, required=False,
                        help="The dev file name")
    parser.add_argument("--candidates_num", default=3, type=int, required=False)
    parser.add_argument("--max_loop_times", default=20, type=int, required=False, help="构建候选选项的最多尝试尝试次数")

    parser.add_argument("--ATOMIC_train_data_path",default='pretrained_data/synthesize_QA_from_ATOMIC/train_random.jsonl',required=False)
    parser.add_argument("--ATOMIC_dev_data_path",
                        default='pretrained_data/synthesize_QA_from_ATOMIC/dev_random.jsonl', required=False)
    parser.add_argument("--CWWV_train_data_path",
                        default='pretrained_data/synthesize_QA_from_CWWV/train_random.jsonl', required=False)
    parser.add_argument("--CWWV_dev_data_path",
                        default='pretrained_data/synthesize_QA_from_CWWV/dev_random.jsonl', required=False)

    parser.add_argument("--ComQA_train_path", default="data/commonsenseqa/train_data.jsonl", type=str,
                        required=False)
    parser.add_argument("--ComQA_dev_path", default="data/commonsenseqa/dev_data.jsonl", type=str, required=False)
    parser.add_argument('--SocialiQA_train_path',
                        default='data/socialiqa/socialiqa-train-dev/train.jsonl',
                        type=str,
                        required=False)
    parser.add_argument('--SocialiQA_train_label_path', default='data/socialiqa/socialiqa-train-dev/train-labels.lst',
                        type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_path',
                        default='data/socialiqa/socialiqa-train-dev/dev.jsonl',
                        type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_label_path', default='data/socialiqa/socialiqa-train-dev/dev-labels.lst',
                        type=str,
                        required=False)

    #parser.add_argument("--output_dir", default='pretrained_data/synthesize_QA_knowledge_sub_Qs_joined_files/', type=str, required=False,
    #                    help="The output directory where the model predictions and checkpoints will be written.")
    #暂时
    parser.add_argument("--output_dir", default='pretrained_data/synthesize_QA_knowledge_files/', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #parser.add_argument("--output_dir", default='model_save/amr_synthesize_QA_with_sub_Q/roberta-mlm-margin', type=str,
    #                    required=False,
    #                    help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--with_sep",action='store_true',required=False)
    parser.add_argument("--QA_templated_based_concat", action='store_true', required=False)

    parser.add_argument("--save_candidates_file",action='store_true',required=False)
    parser.add_argument("--candidates_output_dir",default="pretrained_data/Q_only_based_synthesize_QA_filtered/candidates", type=str, required=False)
    parser.add_argument("--candidates_output_file",
                        default="Q_only_based_synthesize_QA_full_test_high_quality_0.7.jsonl", type=str,
                        required=False)
    parser.add_argument("--candidates_train_output_file",
                        default="Q_only_based_synthesize_QA_full_train_high_quality_0.7.jsonl", type=str,
                        required=False)
    parser.add_argument("--candidates_test_output_file",
                        default="Q_only_based_synthesize_QA_full_test_high_quality_0.7.jsonl", type=str,
                        required=False)

    args = parser.parse_args()

    return args

skip_words = set(stopwords.words('english'))
skip_words.add('\'s')
skip_words.add('.')
skip_words.add(',')
nlp=spacy.load("en_core_web_sm")
PERSON_NAMES = ['Alex', 'Ash', 'Aspen', 'Bali', 'Berkeley', 'Cameron', 'Chris', 'Cody', 'Dana', 'Drew', 'Emory', 'Flynn', 'Gale', 'Jamie', 'Jesse',
'Kai', 'Kendall', 'Kyle', 'Lee', 'Logan', 'Max', 'Morgan', 'Nico', 'Paris', 'Pat', 'Quinn', 'Ray', 'Robin', 'Rowan', 'Rudy', 'Sam', 'Skylar', 'Sydney',
'Taylor', 'Tracy', 'West', 'Wynne']
MODEL_CLASSES = {
    'roberta-mlm': (RobertaTokenizer),
    'albert-mlm': (AlbertTokenizer)
}


def handle_words(span, tokenizer, keywords=None, is_start=False,skip_stopwords=True):
    inputs = []
    labels = []
    #words = nltk.word_tokenize(span)
    words = [word.text_with_ws for word in nlp(span)]
    # 添加前缀
    #words.insert(0, prefix)

    for w_i, w in enumerate(words):
        #暂时
        if isinstance(tokenizer,AlbertTokenizer):
            if w_i > 0 and not words[w_i - 1].endswith(' '):
                w_bpes = tokenizer.tokenize(w.strip())
                if len(w_bpes)==0:
                    continue
                w_bpes=list(filter(lambda x:len(x)>0,[w_bpes[0][1:]]+w_bpes[1:]))
            else:
                w_bpes = tokenizer.tokenize(w.strip())
        else:
            #if (w_i == 0 and is_start) or w == '.' or w == ',' or w.startswith('\''):
            if (w_i == 0 and is_start) or (w_i > 0 and not words[w_i - 1].endswith(' ')):
                w_bpes = tokenizer.tokenize(w.strip())
            else:
                w_bpes = tokenizer.tokenize(w.strip(), add_prefix_space=True)


        inputs.extend(w_bpes)
        if keywords != None:
            if w.strip() in keywords:
                labels.extend(w_bpes)
            else:
                labels.extend([-100]*len(w_bpes))
        else:
            if not skip_stopwords or (w.strip() not in PERSON_NAMES and w.strip() not in skip_words and w.lower().strip() not in skip_words):
                labels.extend(w_bpes)
            else:
                labels.extend([-100]*len(w_bpes))
    return inputs, labels


def handle_underscores(suffixs, tokenizer, prefix:str, keywords=None, skip_stopwords=True):
    inputs_list=[]
    labels_list=[]
    for suffix in suffixs:
        inputs = []
        labels = []
        suffix=suffix.strip()
        if isinstance(tokenizer,AlbertTokenizer):
            w_bpes = tokenizer.tokenize(prefix.strip())
        else:
            if prefix.startswith(' '):
                w_bpes = tokenizer.tokenize(prefix.strip(), add_prefix_space=True)
            else:
                w_bpes = tokenizer.tokenize(prefix.strip())
        inputs+=w_bpes
        labels+=[-100]*len(w_bpes)
        if '_' in suffix:
            suffix_parts = [i.strip() for i in suffix.split('___')]
            for i, part in enumerate(suffix_parts):
                if part:
                    tmp_inputs, tmp_labels = handle_words(part, tokenizer, keywords=keywords, is_start=(i==0 and not prefix.endswith(' ')),skip_stopwords=skip_stopwords)
                    inputs += tmp_inputs
                    labels += tmp_labels

                    if i != len(suffix_parts) - 1 and suffix_parts[i+1]:
                        inputs.append(tokenizer.mask_token)
                        labels.append(-100)
                else:
                    inputs.append(tokenizer.mask_token)
                    labels.append(-100)
        else:
            tmp_inputs, tmp_labels = handle_words(suffix, tokenizer, keywords=keywords, is_start=not prefix.endswith(' '),skip_stopwords=skip_stopwords)
            inputs += tmp_inputs
            labels += tmp_labels

        inputs_list.append(inputs)
        labels_list.append(labels)

    return inputs_list, labels_list


def mlm_data_tokenized(preprocessed_datas,args,tokenizer,is_CQA_dataset=False,skip_stopwords=True):
    """
    实现方法基于Pre-training Is (Almost) All You Need的target premise score，即只mask问句/假设部分
    :param preprocessed_datas:
    :param tokenizer:
    :return:
    """
    tokenized_datas = []
    # 暂时
    #preprocessed_datas = preprocessed_datas[:2000]
    #此处应屏蔽掉停用词，参照HyKAS代码的实现
    for stems, candidates, answer_key, keywords in tqdm(preprocessed_datas):
        #inputs_list, labels_list = handle_text(stems, tokenizer, prefix="Q: ")
        '''if args.QA_templated_based_concat:   
        else:'''
        if isinstance(stems,list):
            inputs_list, labels_list = handle_underscores(stems, tokenizer, prefix="Q: ", keywords=keywords, skip_stopwords=skip_stopwords)
        else:
            inputs_list, labels_list = handle_underscores([stems], tokenizer, prefix="Q: ", keywords=keywords, skip_stopwords=skip_stopwords)

        #if args.with_sub_Q:
        question_ids_list = [tokenizer.convert_tokens_to_ids(inputs) for inputs in inputs_list]
        '''else:
            question_ids=tokenizer.convert_tokens_to_ids(inputs_list[0])'''

        if args.mask_type == 'Q_and_A':
            #choices_list, choice_labels_list = [handle_underscores(choice, tokenizer, prefix="A: ")
            #                                    for choice in candidates]
            choices_list, choice_labels_list = handle_underscores(candidates, tokenizer, prefix=" A: ", skip_stopwords=skip_stopwords)
            choice_ids = [tokenizer.convert_tokens_to_ids(choice)
                          for choice in choices_list]
            if args.with_sep:
                label_ids_list = [[[-100] + (labels + [-100] +choice_labels) + [-100]
                                   for labels in labels_list]
                                  for choice_labels in choice_labels_list]
            else:
                label_ids_list = [[[-100] + (labels + choice_labels) + [-100]
                                   for labels in labels_list]
                                  for choice_labels in choice_labels_list]
        else:
            if isinstance(tokenizer, AlbertTokenizer):
                choice_ids = [tokenizer.encode("A: " + choice)[1:-1]
                              for choice in candidates]
            else:
                choice_ids = [tokenizer.encode("A: " + choice, add_prefix_space=True)[1:-1]
                              for choice in candidates]
            if args.with_sep:
                label_ids_list = [[-100] + (labels_list[0] + [-100] +[-100] * len(cand)) + [-100]
                                  for cand in choice_ids]
            else:
                label_ids_list = [[-100] + (labels_list[0] + [-100] * len(cand)) + [-100]
                                  for cand in choice_ids]

        #choice_ids = [tokenizer.encode("A: "+choice, add_prefix_space=True)[1:-1] for choice in candidates]
        #if args.with_sub_Q:
        if args.with_sep:
            sequences_list = [
                [[tokenizer.cls_token_id] + (question_ids + [tokenizer.sep_token_id] +choice_ids[i]) + [tokenizer.sep_token_id]
                 for question_ids in question_ids_list]
                for i in range(len(choice_ids))]
        else:
            sequences_list = [
                [[tokenizer.cls_token_id] + (question_ids + choice_ids[i]) + [tokenizer.sep_token_id]
                 for question_ids in question_ids_list]
                for i in range(len(choice_ids))]
        max_length=max([len(sq) for cand in sequences_list for sq in cand])
        if max_length > args.max_seq_length and not is_CQA_dataset:
            continue

        label_ids_list = [[[t if t == -100 else sequences_list[i][j][t_i]
                            for t_i, t in enumerate(cand)]
                           for j, cand in enumerate(Q_cands)]
                          for i, Q_cands in enumerate(label_ids_list)]
        tokenized_datas.append([
            sequences_list, label_ids_list, answer_key
        ])

    return tokenized_datas


def preprocessed_and_tokenized_ATOMIC(data_list, args, tokenizer):
    preprocessed_datas=[]
    for sample in data_list:
        answerKey = sample['correct']
        stem = " ".join(sample['context'].strip().split())
        choices = [c.strip() for c in sample['candidates']]
        stems = [stem]
        keywords=sample['keywords']

            #stems.extend(sample['sub_Q_texts'])
        preprocessed_datas.append([
            stems, choices, answerKey, keywords
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def preprocessed_and_tokenized_CWWV(data_list, args, tokenizer):
    preprocessed_datas=[]
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    for sample in data_list:
        answerKey = answerMap[sample['answerKey']]
        stem = " ".join(sample['question']['stem'].strip().split())
        choices = [c['text'].strip() for c in sample['question']['choices']]
        stems = [stem]

            #stems.extend(sample['sub_Q_texts'])
        preprocessed_datas.append([
            stems, choices, answerKey, None
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def processed_and_tokenized_synthesize_QA_data(datas, args, tokenizer: RobertaTokenizer):
                                               #candidates_origin_sents=None,candidates_questions=None,
                                               #candidates_answers=None,candidates_answer_types=None,
                                               #candidates_questions_embeddings=None):
    relation_type_datas = {}
    answer_type_datas = {}

    clean_datas = []
    for data in datas:
        if 'context' in data.keys():
            stem = " ".join((data['context'] + " " + data['question']).strip().split())
            data.pop('context')
            data.pop('question')
        else:
            stem = " ".join(data['question'].strip().split())
            data.pop('question')

        if not args.declarative_sent and (not stem.endswith('?') or stem.index('?') != len(stem) - 1):
            # bad_stem_num+=1
            continue

        data['question'] = stem

        if 'sub_Q_texts' in data.keys():
            compositions = [" ".join(composition.strip().split()) for composition in data['sub_Q_texts']]
            data.pop('sub_Q_texts')
            if len(compositions) <= 1:
                compositions = []
            compositions = list(filter(lambda x: x.endswith('?') and x.index('?') == len(x) - 1, compositions))
            data['sub_Q_texts'] = compositions

        clean_datas.append(data)

    for data in tqdm(clean_datas):
        if 'answer_type' not in data.keys():
            data['answer_type']='no_answer_type'

        if data['answer_type'] not in answer_type_datas.keys():
            answer_type_datas[data['answer_type']] = [data]
        else:
            answer_type_datas[data['answer_type']].append(data)

        if 'relation_type' not in data.keys():
            data['relation_type']='no_relation_type'

        if data['relation_type'] not in relation_type_datas.keys():
            relation_type_datas[data['relation_type']] = [data]
        else:
            relation_type_datas[data['relation_type']].append(data)

    preprocessed_datas = []

    if args.hard_candidates:

        origin_sents = []
        questions = []
        answers = []
        answer_types = []
        for data in datas:
            if "origin_snt" not in data.keys() or "question" not in data.keys() or \
                    "answer" not in data.keys():
                continue
            origin_sents.append(data["origin_snt"])
            questions.append(data["question"])
            answers.append(data["answer"])
            answer_types.append(data["answer_type"])

        hard_candidates_datas=generate_hard_candidates(args.sentence_embedding_model_name,
                                                       args.device,args.candidates_num,
                                                       origin_sents,questions,answers,answer_types)
        preprocessed_datas.extend(hard_candidates_datas)


    if not args.no_rel_type:
        for datas_dict in [relation_type_datas, answer_type_datas]:
            for key in datas_dict.keys():
                random.shuffle(datas_dict[key])
                for data in tqdm(datas_dict[key]):
                    stem = data['question']
                    answer = data['answer']
                    if len(answer.strip()) == 0 or len(stem.strip()) < 2 or len(datas_dict[key]) < args.candidates_num:
                        continue

                    candidates = [answer]
                    candidates_num = args.candidates_num
                    times = 0
                    while len(candidates) < candidates_num and times < args.max_loop_times:
                        times += 1
                        index = random.choice(range(len(datas_dict[key])))
                        candidate = datas_dict[key][index]['answer']
                        if candidate != answer and \
                                answer not in candidate and \
                                candidate not in answer:
                            candidates.append(candidate)
                    if len(candidates) < candidates_num and times >= args.max_loop_times:
                        continue

                    random.shuffle(candidates)
                    answer_key = candidates.index(answer)

                    candidates = [c[0].lower() + c[1:] for c in candidates]

                    stems = [stem]
                    if 'sub_Q_texts' in data.keys():
                        stems.extend(data['sub_Q_texts'])
                    preprocessed_datas.append([
                        stems, candidates, answer_key, None
                    ])

    #存储中间结果
    '''if args.save_candidates_file:
        if not os.path.exists(args.candidates_output_dir):
            os.makedirs(args.candidates_output_dir)
        with open(os.path.join(args.candidates_output_dir,args.candidates_output_file),"w") as output:
            for data in preprocessed_datas:
                sample={'question':data[0][0],'candidates':data[1],'answerKey':data[2]}
                output.write(json.dumps(sample)+"\n")'''

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def processed_and_tokenized_synthesize_QA_data_with_candidates_joined(datas, args,tokenizer: RobertaTokenizer,candidates_saved_file):

    relation_type_datas = {}
    answer_type_datas = {}

    clean_datas = []
    for data in datas:
        if 'context' in data.keys():
            stem = " ".join((data['context'] + " " + data['question']).strip().split())
            data.pop('context')
            data.pop('question')
        else:
            stem = " ".join(data['question'].strip().split())
            data.pop('question')

        if not args.declarative_sent and (not stem.endswith('?') or stem.index('?') != len(stem) - 1):
            # bad_stem_num+=1
            continue

        data['question'] = stem

        if 'sub_Q_texts' in data.keys():
            compositions = [" ".join(composition.strip().split()) for composition in data['sub_Q_texts']]
            data.pop('sub_Q_texts')
            if len(compositions) <= 1:
                compositions = []
            compositions = list(filter(lambda x: x.endswith('?') and x.index('?') == len(x) - 1, compositions))
            data['sub_Q_texts'] = compositions

        clean_datas.append(data)

    for data in tqdm(clean_datas):
        if 'answer_type' not in data.keys():
            data['answer_type'] = 'no_answer_type'

        if data['answer_type'] not in answer_type_datas.keys():
            answer_type_datas[data['answer_type']] = [data]
        else:
            answer_type_datas[data['answer_type']].append(data)

        if 'relation_type' not in data.keys():
            data['relation_type'] = 'no_relation_type'

        if data['relation_type'] not in relation_type_datas.keys():
            relation_type_datas[data['relation_type']] = [data]
        else:
            relation_type_datas[data['relation_type']].append(data)

    preprocessed_datas = []

    if args.hard_candidates:

        origin_sents = []
        questions = []
        answers = []
        answer_types = []
        for data in datas:
            if "origin_snt" not in data.keys() or "question" not in data.keys() or \
                    "answer" not in data.keys():
                continue
            origin_sents.append(data["origin_snt"])
            questions.append(data["question"])
            answers.append(data["answer"])
            answer_types.append(data["answer_type"])

        hard_candidates_datas = generate_hard_candidates(args.sentence_embedding_model_name,
                                                         args.device, args.candidates_num,
                                                         origin_sents, questions, answers, answer_types)
        preprocessed_datas.extend(hard_candidates_datas)

    rel_type_preprocessed_datas = []
    if not args.no_rel_type:
        for datas_dict in [relation_type_datas, answer_type_datas]:
            for key in datas_dict.keys():
                random.shuffle(datas_dict[key])
                for data in tqdm(datas_dict[key]):
                    stem = data['question']
                    answer = data['answer']
                    if len(answer.strip()) == 0 or len(stem.strip()) < 2 or len(datas_dict[key]) < args.candidates_num:
                        continue

                    candidates = [answer]
                    candidates_num = args.candidates_num
                    times = 0
                    while len(candidates) < candidates_num and times < args.max_loop_times:
                        times += 1
                        index = random.choice(range(len(datas_dict[key])))
                        candidate = datas_dict[key][index]['answer']
                        if candidate != answer and \
                                answer not in candidate and \
                                candidate not in answer:
                            candidates.append(candidate)
                    if len(candidates) < candidates_num and times >= args.max_loop_times:
                        continue

                    random.shuffle(candidates)
                    answer_key = candidates.index(answer)
                    answer=answer[0].lower() + answer[1:]

                    candidates = [c[0].lower() + c[1:] for c in candidates]

                    stems = [stem]

                    if args.save_candidates_file:
                        flag=False
                        for sample in preprocessed_datas:
                            if sample[0][0]==stem and candidates[answer_key]==sample[1][sample[2]]:
                                candidates.extend(sample[1])
                                candidates=list(set(candidates))
                                random.shuffle(candidates)
                                answer_key=candidates.index(answer)
                                sample[1]=candidates
                                sample[2]=answer_key
                                flag=True
                        if not flag:
                            rel_type_preprocessed_datas.append([
                                stems, candidates, answer_key, None
                            ])

    preprocessed_datas.extend(rel_type_preprocessed_datas)
    # 存储中间结果
    if args.save_candidates_file:
        if not os.path.exists(args.candidates_output_dir):
            os.makedirs(args.candidates_output_dir)

        with open(os.path.join(args.candidates_output_dir, candidates_saved_file), "w") as output:
            for data in preprocessed_datas:
                sample = {'question': data[0][0], 'candidates': data[1], 'answerKey': data[2]}
                output.write(json.dumps(sample) + "\n")

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def preprocessed_and_tokenized_ComQA(data_list, args, tokenizer):
    preprocessed_datas = []
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    for sample in data_list:
        answerKey = answerMap[sample['answerKey']]
        stem = " ".join(sample['question']['stem'].strip().split())
        choices = [c['text'].strip() for c in sample['question']['choices']]
        stems = [stem]

        preprocessed_datas.append([
            stems, choices, answerKey, None
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def preprocessed_and_tokenized_SocialiQA(data_list, args, tokenizer):
    preprocessed_datas = []
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    for sample in data_list:
        answerKey = sample['correct']-1
        stem=sample['context'].strip() + ' ' + sample['question'].strip()
        stem = " ".join(stem.strip().split())
        choices = [" ".join(sample["answerA"].strip().split()),
                   " ".join(sample["answerB"].strip().split()),
                   " ".join(sample["answerC"].strip().split())]
        stems = [stem]

        preprocessed_datas.append([
            stems, choices, answerKey, None
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def ATOMIC_data_processed_wrapper(t_params):
    return preprocessed_and_tokenized_ATOMIC(*t_params)

def CWWV_data_processed_wrapper(t_params):
    return preprocessed_and_tokenized_CWWV(*t_params)

def synthesize_QA_data_processed_wrapper(t_params):
    return processed_and_tokenized_synthesize_QA_data(*t_params)

def synthesize_QA_data_processed_with_candidates_joined_wrapper(t_params):
    return processed_and_tokenized_synthesize_QA_data_with_candidates_joined(*t_params)

def ComQA_data_processed_wrapper(t_params):
    return preprocessed_and_tokenized_ComQA(*t_params)

def SocialiQA_data_processed_wrapper(t_params):
    return preprocessed_and_tokenized_SocialiQA(*t_params)


def main():
    args=set_args()

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    tokenized_train_datas = []
    tokenized_test_datas = []

    #暂时
    preprocessed_and_tokenized_train_datasets=[]
    preprocessed_and_tokenized_test_datasets = []
    wrappers=[]
    if args.tokenize_synthesize_QA:
        with open(args.train_sub_Qs_data_path, "r") as input_file:
            train_sub_Qs_datas = [json.loads(line) for line in input_file.readlines()]
        with open(args.dev_sub_Qs_data_path, "r") as input_file:
            dev_sub_Qs_datas = [json.loads(line) for line in input_file.readlines()]

        preprocessed_and_tokenized_train_datasets.append(train_sub_Qs_datas)
        preprocessed_and_tokenized_test_datasets.append(dev_sub_Qs_datas)
        if not args.save_candidates_file:
            wrappers.append(synthesize_QA_data_processed_wrapper)
        else:
            wrappers.append(synthesize_QA_data_processed_with_candidates_joined_wrapper)

    if args.tokenize_ATOMIC:
        with open(args.ATOMIC_train_data_path, "r") as input_file:
            ATOMIC_train_datas = [json.loads(line) for line in input_file.readlines()]
            #train_datas.extend(ATOMIC_train_datas)
        with open(args.ATOMIC_dev_data_path, "r") as input_file:
            ATOMIC_test_datas = [json.loads(line) for line in input_file.readlines()]
            #dev_datas.extend(ATOMIC_test_datas)

        #暂时
        #ATOMIC_train_datas=random.sample(ATOMIC_train_datas,50000)
        #
        preprocessed_and_tokenized_train_datasets.append(ATOMIC_train_datas)
        preprocessed_and_tokenized_test_datasets.append(ATOMIC_test_datas)
        wrappers.append(ATOMIC_data_processed_wrapper)

    if args.tokenize_CWWV:
        with open(args.CWWV_train_data_path, "r") as input_file:
            CWWV_train_datas = [json.loads(line) for line in input_file.readlines()]
            #train_datas.extend(CWWV_train_datas)
        with open(args.CWWV_dev_data_path, "r") as input_file:
            CWWV_test_datas = [json.loads(line) for line in input_file.readlines()]
            #dev_datas.extend(CWWV_test_datas)

        # 暂时
        #CWWV_train_datas = random.sample(CWWV_train_datas, 50000)
        #
        preprocessed_and_tokenized_train_datasets.append(CWWV_train_datas)
        preprocessed_and_tokenized_test_datasets.append(CWWV_test_datas)
        wrappers.append(CWWV_data_processed_wrapper)

    if args.tokenize_ComQA:
        with open(args.ComQA_train_path, "r") as ComQA_train_file:
            ComQA_train_raw_list = [json.loads(line) for line in ComQA_train_file.readlines()]
        with open(args.ComQA_dev_path, "r") as ComQA_dev_file:
            ComQA_dev_raw_list = [json.loads(line) for line in ComQA_dev_file.readlines()]

        preprocessed_and_tokenized_train_datasets.append(ComQA_train_raw_list)
        preprocessed_and_tokenized_test_datasets.append(ComQA_dev_raw_list)
        wrappers.append(ComQA_data_processed_wrapper)

    if args.tokenize_SocialiQA:
        with open(args.SocialiQA_train_path, "r") as SocialiQA_train_file, \
                open(args.SocialiQA_train_label_path, "r") as SocialiQA_train_label_file:
            SocialiQA_train_raw_list = [json.loads(line) for line in SocialiQA_train_file.readlines()]
            SocialiQA_train_label_list = list(map(lambda x: int(x.strip()), SocialiQA_train_label_file.readlines()))
            for data, label in zip(SocialiQA_train_raw_list, SocialiQA_train_label_list):
                data['correct']=label

        with open(args.SocialiQA_dev_path, "r") as SocialiQA_dev_file, \
                open(args.SocialiQA_dev_label_path, "r") as SocialiQA_dev_label_file:
            SocialiQA_dev_raw_list = [json.loads(line) for line in SocialiQA_dev_file.readlines()]
            SocialiQA_dev_label_list = list(map(lambda x: int(x.strip()), SocialiQA_dev_label_file.readlines()))
            for data, label in zip(SocialiQA_dev_raw_list, SocialiQA_dev_label_list):
                data['correct']=label

        preprocessed_and_tokenized_train_datasets.append(SocialiQA_train_raw_list)
        preprocessed_and_tokenized_test_datasets.append(SocialiQA_dev_raw_list)
        wrappers.append(SocialiQA_data_processed_wrapper)


    if args.multi_processing:
        with Pool(args.num_workers) as p:
            for train_datas, dev_datas, wrapper in zip(preprocessed_and_tokenized_train_datasets,
                                                       preprocessed_and_tokenized_test_datasets,
                                                       wrappers):
                train_chunk_size = int(math.ceil(len(train_datas) / args.num_workers))
                dev_chunk_size = int(math.ceil(len(dev_datas) / args.num_workers))
                train_results = p.map(wrapper,
                                [(train_datas[i * train_chunk_size:(i + 1) * train_chunk_size], args, tokenizer) for i
                                 in range(args.num_workers)])
                dev_results = p.map(wrapper,
                                      [(
                                       dev_datas[i * dev_chunk_size:(i + 1) * dev_chunk_size], args, tokenizer)
                                       for i
                                       in range(args.num_workers)])
                for result in train_results:
                    tokenized_train_datas.extend(result)
                for result in dev_results:
                    tokenized_test_datas.extend(result)
    else:
        for train_datas, dev_datas, wrapper in zip(preprocessed_and_tokenized_train_datasets,
                                                   preprocessed_and_tokenized_test_datasets,
                                                   wrappers):
            if not args.save_candidates_file:
                train_results=wrapper([train_datas,args,tokenizer])
                dev_results = wrapper([dev_datas, args, tokenizer])
            else:
                train_results = wrapper([train_datas, args, tokenizer,args.candidates_train_output_file])
                dev_results = wrapper([dev_datas, args, tokenizer, args.candidates_test_output_file])
            tokenized_train_datas.extend(train_results)
            tokenized_test_datas.extend(dev_results)


    print("save tokenized data")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    cached_file = os.path.join(args.output_dir,
                               'cached_{}_{}'.format(
                                   str(args.model_type),
                                   str(args.max_seq_length)))

    torch.save({'train':tokenized_train_datas,'test':tokenized_test_datas}, cached_file)


if __name__ == "__main__":
    main()
