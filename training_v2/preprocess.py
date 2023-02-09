from datasets import load_from_disk
from tqdm import tqdm
import json
import os
import penman
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data_path",default='pretrained_data/synthesize_QA_full',required=False)
    parser.add_argument("--output_dir", default='pretrained_data/synthesize_QA_full_files', required=False)
    parser.add_argument("--output_file_template", default='synthesize_QA_full_{}.jsonl', required=False)

    return parser.parse_args()

def convert_synthesize_dataset_to_jsonl(load_data_path="pretrained_data/synthesize_QA",
                                        output_dir="pretrained_data/synthesize_QA_files",
                                        output_file_template="synthesize_QA_{}.jsonl"):
    answer_types=["do","named_entity","other"]
    named_entities = [
        "person", "family", "animal", "language", "nationality", "ethnic-group", "regional-group", "religious-group",
        "political-movement",
        "organization", "company", "government-organization", "military", "criminal-organization", "political-party",
        "market-sector", "school", "university", "research-institute", "team", "league",
        "location", "city", "city-district", "county", "state", "province", "territory", "country", "local-region",
        "country-region", "world-region", "continent",
        "ocean", "sea", "lake", "river", "gulf", "bay", "strait", "canal",
        "peninsula", "mountain", "volcano", "valley", "canyon", "island", "desert", "forest moon", "planet", "star",
        "constellation",

        "facility", "airport", "station", "port", "tunnel", "bridge", "road", "railway-line", "canal", "building",
        "theater", "museum", "palace",
        "hotel", "worship-place", "market", "sports-facility", "park", "zoo", "amusement-park",

        "event", "incident", "natural-disaster", "earthquake", "war", "conference", "game", "festival",
        "product", "vehicle", "ship", "aircraft", "aircraft-type", "spaceship", "car-make", "work-of-art", "picture",
        "music", "show", "broadcast-program",
        "publication", "book", "newspaper", "magazine", "journal",
        "natural-object",
        "award", "law", "court-decision", "treaty", "music-key", "musical-note", "food-dish", "writing-script",
        "variable", "program",
        "molecular-physical-entity", "small-molecule", "protein", "protein-family", "protein-segment", "amino-acid",
        "macro-molecular-complex", "enzyme", "nucleic-acid",
        "pathway", "gene", "dna-sequence", "cell", "cell-line", "species", "taxon", "disease", "medical-condition"
    ]

    amr_based_synthesize_QA_datasets=load_from_disk(load_data_path)
    for key in amr_based_synthesize_QA_datasets.keys():
        if not os.path.exists(output_dir):
           os.makedirs(output_dir)
        with open(os.path.join(output_dir,output_file_template.format(key)),"w") as output_file:
            for example in tqdm(amr_based_synthesize_QA_datasets[key]):
                context=None
                if 'QA_contexts' in example.keys():
                    context=example['QA_contexts']
                if 'origin_sents' in example.keys():
                    origin_snt = example['origin_sents']
                else:
                    origin_snt=penman.decode(example['sent_origin_amrs']).metadata['snt']

                #sub_Q_texts=None
                for question_graph,question,answer_graph,answer in zip(example['multi_Q_amr_graphs_for_sents'],example['multi_Q_texts_for_sents'],
                                           example['multi_A_amr_graphs_for_sents'],example['multi_A_texts_for_sents']):
                    try:
                        question_graph = penman.decode(question_graph)
                        answer_graph=penman.decode(answer_graph)
                    except Exception:
                        print("error")
                        continue

                    answer_root=answer_graph.triples[0]
                    if answer_root[1]!=':instance':
                        print('error')
                        continue
                    if '-' in answer_root[2] and answer_root[2].rindex('-')+1<len(answer_root[2]) and answer_root[2][answer_root[2].rindex('-')+1:].isdigit():
                        answer_type='do'
                    elif answer_root[2] in named_entities:
                        answer_type=answer_root[2]
                    else:
                        answer_type='other'

                    relation_type=None
                    for triple_index,triple in enumerate(question_graph.triples):
                        if triple[2]=='amr-unknown' and triple_index>0:
                            if question_graph.triples[triple_index - 1][2] == triple[0]:
                                relation_type=question_graph.triples[triple_index-1][1]
                            elif question_graph.triples[triple_index - 1][0] == triple[0]:
                                relation_type = question_graph.triples[triple_index - 1][1]+'-of'
                    if relation_type==None:
                        print('error')
                        continue

                    data = {'origin_snt': origin_snt, 'question': question, 'answer': answer,
                            'answer_type': answer_type, "relation_type": relation_type}
                    if context!=None:
                        data['context']=context
                    output_file.write(json.dumps(data)+'\n')


def convert_dataset_to_jsonl(dataset_load_path='pretrained_data/wiki_sample_for_synthesize',
                             output_dir='pretrained_data/wiki_sample_for_synthesize_files',
                             output_file_template="wiki_sample_for_synthesize_{}.jsonl",
                             skip_keys=[]):
    datasets = load_from_disk(dataset_load_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for key in datasets.keys():
        with open(os.path.join(output_dir, output_file_template.format(key)), "w") as output_file:
            for example in tqdm(datasets[key]):
                for key in skip_keys:
                    if key in example.keys():
                        example.pop(key)
                #特殊操作
                if 'sent_origin_amrs' in example.keys():
                    example['origin_sents']=penman.decode(example['sent_origin_amrs']).metadata['snt']
                    example.pop('sent_origin_amrs')
                output_file.write(json.dumps(example) + '\n')


if __name__ == "__main__":
    args=set_args()
    convert_dataset_to_jsonl(args.load_data_path,args.output_dir,args.output_file_template)
    '''convert_synthesize_dataset_to_jsonl(load_data_path="pretrained_data/synthesize_QA_full",
                                        output_dir="pretrained_data/synthesize_QA_full_files",
                                        output_file_template="synthesize_QA_full_{}.jsonl", )'''

    '''convert_dataset_to_jsonl(dataset_load_path='pretrained_data/wiki_sample_for_synthesize',
                             output_dir='pretrained_data/wiki_sample_for_synthesize_files',
                             output_file_template="wiki_sample_for_synthesize_{}.jsonl",
                             skip_keys=[])'''
