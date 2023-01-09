import spacy
import json
import re
import os
from datasets import load_from_disk,DatasetDict,Dataset
import random
import argparse


def setup_train_args():
    parser = argparse.ArgumentParser()
    #暂时
    parser.add_argument('--preprocessing_num_workers', default=20, type=int, required=False)
    #parser.add_argument('--preprocessing_num_workers', default=1, type=int, required=False)
    parser.add_argument('--overwrite_cache', default=False, type=bool, required=False)
    parser.add_argument('--context_beside_last_sent',default=True,type=bool,required=False)

    parser.add_argument('--min_sent_length',default=3,type=int,required=False)
    #parser.add_argument('--max_last_sent_length', default=40, type=int, required=False)
    parser.add_argument('--max_last_sent_length', default=30, type=int, required=False)
    parser.add_argument('--random_max_last_sent_length', default=True, type=bool, required=False)
    parser.add_argument('--mean_last_sent_length', default=10, type=int, required=False)

    parser.add_argument('--max_sub_sents_num', default=3, type=int, required=False)
    parser.add_argument('--max_sents_num', default=3, type=int, required=False)

    #parser.add_argument('--max_tokens_length',default=80,type=int,required=False)
    parser.add_argument('--max_tokens_length', default=50, type=int, required=False)
    parser.add_argument('--random_max_tokens_length', default=True, type=bool, required=False)
    parser.add_argument('--mean_tokens_length', default=15, type=int, required=False)

    parser.add_argument('--dataset_load_path', default='pretrained_data/wiki40b',
                        type=str, required=False)
    parser.add_argument('--dataset_save_path', default='pretrained_data/wiki_sample_for_synthesize',
                        type=str, required=False)

    parser.add_argument('--preprocess_data', default=True, type=bool, required=False)
    #parser.add_argument('--train_sample_num', default=300000, type=int, required=False)
    parser.add_argument('--train_sample_num', default=50000, type=int, required=False)
    #parser.add_argument('--eval_sample_num', default=30000, type=int, required=False)
    parser.add_argument('--eval_sample_num', default=2500, type=int, required=False)

    return parser.parse_args()


def sample_sentences_from_wiki(args):
    nlp=spacy.load("en_core_web_sm")

    def check_en_str(text):
        #import re
        #此处^和$分别表示匹配字符串的开头和末尾
        pattern = re.compile('^[A-Za-z0-9.,:;!$%\' °•()_*^+=\-/]+$')
        if pattern.fullmatch(text):
            return True
        else:
            return False


    def preprocess_example(text: str):
        sections = text.replace('\n', ' ').replace('_START_ARTICLE_', '').split('_START_SECTION_')

        clean_sections = []
        for section in sections:
            start_index = section.find('_START_PARAGRAPH_')
            if start_index != -1:
                clean_sections.append(
                    section[start_index + len('_START_PARAGRAPH_'):].strip())

        processed_sentences = []
        for section in clean_sections:
            sentences = section.split('_NEWLINE_')
            line_sentences=sentences

            good_sents = []
            for sentences in line_sentences:
                sentences = re.sub(r"\(.*?\)", " ", sentences).strip()
                sentences=" ".join(sentences.strip().split())

                if len(sentences) > 0:
                    doc = nlp(sentences)
                    sents=[sent for sent in doc.sents]

                    for sent in sents:
                        #if len(sent)>args.min_sent_length and sent[0].text[0].upper()==sent[0].text[0] and check_en_str(sent.text):
                        if len(sent) > args.min_sent_length and check_en_str(sent.text):
                            good_sents.append(sent)

                        elif args.context_beside_last_sent:
                            while len(good_sents) > 0:
                                max_sents_num = min(args.max_sents_num, len(good_sents))
                                sents_num = random.choice(list(range(1, max_sents_num + 1, 1)))

                                max_tokens_length = args.max_tokens_length
                                if args.random_max_tokens_length:
                                    max_tokens_length = random.random() * (
                                                max_tokens_length - args.mean_tokens_length) + args.mean_tokens_length

                                while sents_num > 1 and sum(
                                        [len(sent) for sent in good_sents[:sents_num]]) > max_tokens_length:
                                    sents_num -= 1

                                if sum([len(sent) for sent in good_sents[:sents_num]]) < max_tokens_length:
                                    processed_sentences.append(good_sents[:sents_num])

                                good_sents = good_sents[sents_num:]

            while len(good_sents)>0:
                max_sents_num=min(args.max_sents_num,len(good_sents))
                sents_num=random.choice(list(range(1,max_sents_num+1,1)))

                max_tokens_length=args.max_tokens_length
                if args.random_max_tokens_length:
                    max_tokens_length=random.random()*(max_tokens_length-args.mean_tokens_length)+args.mean_tokens_length

                while sents_num>1 and sum([len(sent) for sent in good_sents[:sents_num]])>max_tokens_length:
                    sents_num-=1

                if sum([len(sent) for sent in good_sents[:sents_num]])<max_tokens_length:
                    processed_sentences.append(good_sents[:sents_num])

                good_sents=good_sents[sents_num:]


        contexts=[]
        prefix_sents=[]
        last_sents=[]
        for sents in processed_sentences:
            context_sents=sents[:-1]
            context=" ".join([sent.text for sent in context_sents])

            last_sent=sents[-1].text
            last_subsents=[subsent.strip() for subsent in last_sent.split(",")]
            #subsent_num = len(last_subsents)
            #修改
            max_sents_num=min(len(last_subsents),args.max_sents_num)
            subsent_num = random.choice(list(range(1,max_sents_num+1,1)))

            max_last_sent_length=args.max_last_sent_length
            if args.random_max_last_sent_length:
                max_last_sent_length=int(random.random()*(max_last_sent_length-args.mean_last_sent_length))+args.mean_last_sent_length

            while subsent_num>1 and sum([len(sent.split()) for sent in last_subsents[-subsent_num:]]) > max_last_sent_length:
                subsent_num -= 1

            if sum([len(sent.split()) for sent in last_subsents[-subsent_num:]]) > max_last_sent_length:
                continue

            last_sent=", ".join(last_subsents[-subsent_num:])
            prefix_sent=", ".join(last_subsents[:-subsent_num])
            context = " ".join(context.strip().split())
            #if len(prefix_sent)>0:
                #context=context+" "+prefix_sent+","
            #    context=" ".join(context.strip().split())

            contexts.append(context)
            prefix_sents.append(prefix_sent)
            last_sents.append(last_sent)

        return contexts,prefix_sents,last_sents


    def split_sentences_function(examples):
        contexts_list=[]
        prefix_sents_list=[]
        last_sents_list=[]
        for example in examples[text_column_name]:
            contexts,prefix_sents,last_sents=preprocess_example(example)
            contexts_list.extend(contexts)
            prefix_sents_list.extend(prefix_sents)
            last_sents_list.extend(last_sents)

        return {'contexts': contexts_list, 'prefix_sents':prefix_sents_list, 'last_sents':last_sents_list}


    datasets = load_from_disk(args.dataset_load_path)
    datasets = DatasetDict(
        {'train': datasets['train'].select(
            [int(random.random() * len(datasets['train'])) for i in range(args.train_sample_num)]),
            'test': datasets['test'].select(
                [int(random.random() * len(datasets['test'])) for i in range(args.eval_sample_num)])})


    if isinstance(datasets, DatasetDict):
        column_names = datasets["train"].column_names
    elif isinstance(datasets, Dataset):
        column_names = datasets.column_names
    else:
        raise RuntimeError("dataset load wrong")

    text_column_name = "text" if "text" in column_names else column_names[0]

    sentences_datasets = datasets.map(split_sentences_function,
                                      batched=True,
                                      num_proc=args.preprocessing_num_workers,
                                      remove_columns=column_names,
                                      load_from_cache_file=not args.overwrite_cache,
                                      batch_size=100)
    if not os.path.exists(args.dataset_save_path):
        os.makedirs(args.dataset_save_path)
    sentences_datasets.save_to_disk(args.dataset_save_path)


if __name__ == '__main__':
    args=setup_train_args()
    sample_sentences_from_wiki(args)
