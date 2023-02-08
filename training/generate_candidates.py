from sentence_transformers import SentenceTransformer, util
import random
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import Dataset,DataLoader


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_embedding_model_name",default='all-mpnet-base-v2')
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)
    #parser.add_argument("--batch_size", default=120, type=int, required=False)

    parser.add_argument("--candidates_num", default=3, type=int, required=False)

    args = parser.parse_args()

    return args


class MyDataset(Dataset):
    def __init__(self,origin_sents, questions, answers, keywords_list, questions_embeddings):
        self.origin_sents=origin_sents
        self.questions = questions
        self.answers = answers
        self.keywords_list = keywords_list
        self.questions_embeddings = questions_embeddings

    def __getitem__(self, item):
        return self.origin_sents[item], self.questions[item],\
               self.answers[item], self.keywords_list[item],\
               self.questions_embeddings[item]

    def __len__(self):
        return len(self.questions)


def generate_hard_candidates(sentence_embedding_model_name,device,candidates_num,
                             origin_sents,questions,answers,answer_types,keywords_list=None,
                             Q_prefix="Q: ",A_prefix=" A: "):
    sentence_embedding_model = SentenceTransformer(sentence_embedding_model_name, device=device)
    questions_embeddings = sentence_embedding_model.encode(questions)
    #questions_embeddings=torch.tensor(questions_embeddings).to(device)

    hard_candidates_datas=[]
    if keywords_list==None:
        keywords_list=[None for i in range(len(questions))]

    def Collate_fn(batch):
        batched_origin_sent=[]
        batched_question=[]
        batched_answer=[]
        batched_keywords=[]
        batched_embedding=[]

        for sample in batch:
            origin_sent,question,answer,keywords_list,question_embedding=sample
            batched_origin_sent.append(origin_sent)
            batched_question.append(question)
            batched_answer.append(answer)
            batched_keywords.append(keywords_list)
            batched_embedding.append(question_embedding)

        return batched_origin_sent,batched_question,batched_answer,batched_keywords,torch.tensor(batched_embedding)

    ds=MyDataset(origin_sents, questions, answers, keywords_list, questions_embeddings)
    dl=DataLoader(ds,batch_size=120,collate_fn=Collate_fn)

    for batched_origin_sent, batched_question, batched_answer, batched_keywords, batched_embedding in tqdm(dl):
        batched_scores = util.dot_score(batched_embedding, questions_embeddings).squeeze(0).tolist()

        for origin_sent, question, answer, keywords, scores in\
                zip(batched_origin_sent, batched_question, batched_answer, batched_keywords, batched_scores):

            if len(answer.strip()) == 0 or len(question.strip()) < 2:
                continue

            key_scores = list(zip(origin_sents, questions, answers, answer_types, scores))
            key_scores.sort(key=lambda x: x[4], reverse=True)

            candidates = [answer]
            candidates_num = candidates_num

            index = 0
            while len(candidates) < candidates_num and index < len(key_scores):
                candidate_origin_sent, candidate_question, candidate, candidate_type, score = key_scores[index]
                if candidate_origin_sent == origin_sent or candidate == answer \
                        or len(candidate.strip()) == 0 or len(candidate_question.strip()) < 2:
                    index += 1
                    continue
                if candidate_origin_sent != origin_sent and score < 0.95 and len(candidate.strip()) > 0 and \
                        candidate != answer and \
                        answer not in candidate and \
                        candidate not in answer:
                    candidates.append(candidate)

                index += 1

            if len(candidates) < candidates_num:
                continue

            random.shuffle(candidates)
            answer_key = candidates.index(answer)
            candidates = [c[0].lower() + c[1:] for c in candidates]
            candidates = [c.strip() if c.strip().endswith('.') else c.strip() + '.' for c in candidates]
            stem = question
            hard_candidates_datas.append([stem, candidates, answer_key, keywords, Q_prefix, A_prefix])

    return hard_candidates_datas