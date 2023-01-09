import random

from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
import argparse
import json
import evaluate
from multiprocessing import Pool


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_embedding_model_name",default='all-mpnet-base-v2')
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)
    #parser.add_argument("--batch_size", default=120, type=int, required=False)
    parser.add_argument("--data_path",default="pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.8.jsonl",
                        type=str,required=False)
    #parser.add_argument("--data_path",default="pretrained_data/Q_only_based_synthesize_QA_filtered/Q_only_based_synthesize_QA_full_train_high_quality_0.7.jsonl",
    #                    type=str,required=False)
    #parser.add_argument("--data_path",default="data/commonsenseqa/dev_data.jsonl",
    #                    type=str,required=False)
    #parser.add_argument("--data_path",default="data/socialiqa/socialiqa-train-dev/dev.jsonl",
    #                    type=str,required=False)
    #parser.add_argument("--num_workers",default=100,type=int,required=False)
    parser.add_argument("--num_workers", default=1, type=int, required=False)

    args = parser.parse_args()

    return args


class MyDataset(Dataset):
    def __init__(self,questions, questions_embeddings):
        self.questions = questions
        self.questions_embeddings = questions_embeddings

    def __getitem__(self, item):
        return self.questions[item],self.questions_embeddings[item]

    def __len__(self):
        return len(self.questions)


def count_sim_score(sentence_embedding_model_name,device,questions):
    sentence_embedding_model = SentenceTransformer(sentence_embedding_model_name, device=device)
    questions_embeddings = sentence_embedding_model.encode(questions)

    def Collate_fn(batch):
        batched_question=[]
        batched_embedding=[]

        for sample in batch:
            question,question_embedding=sample
            batched_question.append(question)
            batched_embedding.append(question_embedding)

        return batched_question,torch.tensor(batched_embedding)

    ds=MyDataset(questions, questions_embeddings)
    dl=DataLoader(ds,batch_size=120,collate_fn=Collate_fn,shuffle=False)

    sim_score_sum=0
    num=0
    for batched_question, batched_embedding in tqdm(dl):
        batched_scores = util.dot_score(batched_embedding, questions_embeddings).squeeze(0).tolist()

        for question, scores in zip(batched_question, batched_scores):

            if len(question.strip()) < 2:
                continue

            sim_score_sum+=(sum(scores)-1)
            num+=(len(scores)-1)

    print(sim_score_sum/num)


def main_sim_score():
    args = set_args()

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device
    with open(args.data_path, "r") as input_file:
        datas = [json.loads(line) for line in input_file.readlines()]

    questions = []
    for data in datas:
        context=""
        if "context" in data.keys():
            context=data["context"].strip()

        if "question" not in data.keys():
            continue
        if isinstance(data["question"],str):
            stem=data["question"].strip()
        else:
            stem=data["question"]["stem"]

        if len(context)!=0:
            stem=context+" "+stem

        questions.append(stem)

    avg_score=count_sim_score(args.sentence_embedding_model_name,device,questions)
    print(avg_score)


def avg_rouge_score(params):
    question,questions=params
    rouge = evaluate.load('rouge')

    q_rouge1_sum=0
    q_rouge2_sum = 0
    q_num=0
    ref_sampled_questions=random.sample(questions,500)
    for ref in tqdm(ref_sampled_questions):
        if question==ref:
            continue
        results = rouge.compute(predictions=[question],references = [ref])
        q_rouge1_sum+=results["rouge1"]
        q_rouge2_sum+=results["rouge2"]
        q_num+=1

    avg_rouge1=q_rouge1_sum/q_num
    avg_rouge2=q_rouge2_sum/q_num

    return avg_rouge1,avg_rouge2


def main_rouge():
    args = set_args()

    with open(args.data_path, "r") as input_file:
        datas = [json.loads(line) for line in input_file.readlines()]

    questions = []
    for data in datas:
        context = ""
        if "context" in data.keys():
            context = data["context"].strip()

        if "question" not in data.keys():
            continue
        if isinstance(data["question"], str):
            stem = data["question"].strip()
        else:
            stem = data["question"]["stem"]

        if len(context) != 0:
            stem = context + " " + stem

        questions.append(stem)

    rouge1_sum=0
    rouge2_sum=0
    num = 0
    random.shuffle(questions)
    sample_questions=random.sample(questions,500)

    with Pool(args.num_workers) as p:
        results = p.map(avg_rouge_score,[(question,questions) for question in sample_questions])
        for result in tqdm(results):
            avg_rouge1,avg_rouge2=result
            rouge1_sum+=avg_rouge1
            rouge2_sum+=avg_rouge2
            num+=1

    print("rouge1: "+str(rouge1_sum/num))
    print("rouge2: " + str(rouge2_sum / num))



if __name__ == "__main__":
    main_sim_score()
    main_rouge()


