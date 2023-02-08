import argparse
import json
import os
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
from multiprocessing import Pool
import math
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
import torch
import random
from generate_candidates import generate_hard_candidates


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_workers",default=50,type=int,required=False)
    parser.add_argument("--multi_processing", action='store_true', required=False)

    parser.add_argument("--model_type", default='roberta-mlm', type=str, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=False)
    parser.add_argument("--max_seq_length", default=110, type=int)
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A

    parser.add_argument("--tokenize_synthesize_QA", action="store_true")

    parser.add_argument("--hard_candidates",action='store_true')
    parser.add_argument("--sentence_embedding_model_name",default='all-mpnet-base-v2')
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)

    parser.add_argument("--train_data_path",
                        default='pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.8.jsonl',
                        type=str, required=False,
                        help="The train file name")
    parser.add_argument("--dev_data_path",
                        default='pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_test_high_quality_0.8.jsonl',
                        type=str, required=False,
                        help="The dev file name")
    parser.add_argument("--candidates_num", default=3, type=int, required=False)
    parser.add_argument("--max_loop_times", default=20, type=int, required=False, help="构建候选选项的最多尝试尝试次数")

    parser.add_argument("--output_dir", default='Q_only_based_synthesize_QA_full_filtered_0.8_with_hard_candidates_roberta', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--save_candidates_file", action='store_true', required=False)
    parser.add_argument("--candidates_output_dir",
                        default="pretrained_data/Q_only_based_synthesize_QA_filtered/candidates", type=str,
                        required=False)
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
skip_words.add('?')
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
    words = [word.text_with_ws for word in nlp(span)]

    for w_i, w in enumerate(words):
        if isinstance(tokenizer,AlbertTokenizer):
            if w_i > 0 and not words[w_i - 1].endswith(' '):
                w_bpes = tokenizer.tokenize(w.strip())
                w_bpes=list(filter(lambda x:len(x)>0,[w_bpes[0][1:]]+w_bpes[1:]))
            else:
                w_bpes = tokenizer.tokenize(w.strip())
        else:
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
    for stem, candidates, answer_key, keywords, Q_prefix, A_prefix in tqdm(preprocessed_datas):
        inputs_list, labels_list = handle_underscores([stem], tokenizer, prefix=Q_prefix, keywords=keywords,
                                                      skip_stopwords=skip_stopwords)
        inputs=inputs_list[0]
        labels=labels_list[0]
        question_ids = tokenizer.convert_tokens_to_ids(inputs)

        if args.mask_type == 'Q_and_A':
            choices_list, choice_labels_list = handle_underscores(candidates, tokenizer, prefix=A_prefix,
                                                                  skip_stopwords=skip_stopwords)
            choice_ids = [tokenizer.convert_tokens_to_ids(choice)
                          for choice in choices_list]

            label_ids_list = [[-100] + (labels + choice_labels) + [-100]
                                  for choice_labels in choice_labels_list]
        else:
            if isinstance(tokenizer, AlbertTokenizer):
                choice_ids = [tokenizer.encode(A_prefix + choice)[1:-1]
                              for choice in candidates]
            else:
                choice_ids = [tokenizer.encode(A_prefix + choice, add_prefix_space=True)[1:-1]
                              for choice in candidates]

            label_ids_list = [[-100] + (labels + [-100] * len(cand)) + [-100]
                                  for cand in choice_ids]

        sequences_list = [
            [tokenizer.cls_token_id] + (question_ids + choice_ids[i]) + [tokenizer.sep_token_id]
            for i in range(len(choice_ids))]

        max_length=max([len(cand) for cand in sequences_list])
        if max_length > args.max_seq_length and not is_CQA_dataset:
            continue

        label_ids_list = [[cand if cand == -100 else sequences_list[i][j]
                           for j, cand in enumerate(Q_cands)]
                          for i, Q_cands in enumerate(label_ids_list)]

        tokenized_datas.append([
            sequences_list, label_ids_list, answer_key
        ])

    return tokenized_datas


def preprocessed_and_tokenized_ATOMIC(data_list, args, tokenizer):
    Q_prefix=""
    A_prefix=" "
    preprocessed_datas=[]
    for sample in data_list:
        answerKey = sample['correct']
        stem = " ".join(sample['context'].strip().split())
        choices = [c.strip() for c in sample['candidates']]
        choices = [c.strip() if c.strip().endswith('.') else c.strip() + '.' for c in choices]
        keywords=sample['keywords']

        preprocessed_datas.append([
            stem, choices, answerKey, keywords, Q_prefix, A_prefix
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def preprocessed_and_tokenized_CWWV(data_list, args, tokenizer):
    Q_prefix=""
    A_prefix=" "

    preprocessed_datas=[]
    answerMap = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, '1': 0, '2': 1, '3': 2, '4': 3,
                 '5': 4}
    for sample in data_list:
        answerKey = answerMap[sample['answerKey']]
        stem = " ".join(sample['question']['stem'].strip().split())
        choices = [c['text'].strip() for c in sample['question']['choices']]

        if stem.endswith('.'):
            stem = stem[:-1]
        if not stem.endswith('[MASK]'):
            stem_parts = stem.split('[MASK]')
            stem = stem_parts[0].strip()
            choices = [c + stem_parts[1] + '.' for c in choices]
        else:
            stem = stem.replace('[MASK]','').strip()
            choices = [c + '.' for c in choices]

        preprocessed_datas.append([
            stem, choices, answerKey, None, Q_prefix, A_prefix
        ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def processed_and_tokenized_synthesize_QA_data(datas, args, tokenizer: RobertaTokenizer):
    Q_prefix="Q: "
    A_prefix=" A: "

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

        if not stem.endswith('?') or stem.index('?') != len(stem) - 1:
            continue

        data['question'] = stem

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
                candidates = [c.strip() if c.strip().endswith('.') else c.strip()+'.' for c in candidates]

                preprocessed_datas.append([
                    stem, candidates, answer_key, None, Q_prefix, A_prefix
                ])

    tokenized_datas = mlm_data_tokenized(preprocessed_datas, args, tokenizer)

    return tokenized_datas


def processed_and_tokenized_synthesize_QA_data_with_candidates_joined(datas, args,tokenizer: RobertaTokenizer,candidates_saved_file):
    Q_prefix = "Q: "
    A_prefix = " A: "

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
                                stems, candidates, answer_key, None, Q_prefix, A_prefix
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


def synthesize_QA_data_processed_wrapper(t_params):
    return processed_and_tokenized_synthesize_QA_data(*t_params)


def main():
    args=set_args()

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    tokenized_train_datas = []
    tokenized_test_datas = []

    preprocessed_and_tokenized_train_datasets=[]
    preprocessed_and_tokenized_test_datasets = []
    wrappers=[]
    if args.tokenize_synthesize_QA:
        with open(args.train_data_path, "r") as input_file:
            train_datas = [json.loads(line) for line in input_file.readlines()]
        with open(args.dev_data_path, "r") as input_file:
            dev_datas = [json.loads(line) for line in input_file.readlines()]

        if not args.hard_candidates or not args.multi_processing:
            preprocessed_and_tokenized_train_datasets.append(train_datas)
            preprocessed_and_tokenized_test_datasets.append(dev_datas)
            wrappers.append(synthesize_QA_data_processed_wrapper)
        else:
            train_results = synthesize_QA_data_processed_wrapper([train_datas, args, tokenizer])
            dev_results = synthesize_QA_data_processed_wrapper([dev_datas, args, tokenizer])
            tokenized_train_datas.extend(train_results)
            tokenized_test_datas.extend(dev_results)

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
            train_results=wrapper([train_datas,args,tokenizer])
            dev_results = wrapper([dev_datas, args, tokenizer])
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
