import amrlib
import penman.surface
import penman
import argparse
import json
import os
import random
from datasets import load_from_disk,DatasetDict
import spacy

from amr_utils import (amr2text,
                             text2amr,
                             delete_undefined_keys_in_tree_triples,
                             complete_undefined_keys_in_tree_triples,
                             get_tree_all_defined_undefined_keys,
                             get_triple_parents_dict)


def set_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--preprocessing_num_workers', default=1, type=int, required=False)
    parser.add_argument('--overwrite_cache', default=False, type=bool, required=False)

    parser.add_argument('--wiki_sentences_dataset_path', default='pretrained_data/wiki_sample_for_synthesize',
                        type=str, required=False)
    parser.add_argument('--synthesize_dataset_save_path', default='pretrained_data/synthesize_QA_question',
                        type=str, required=False)

    parser.add_argument('--Q_node_percent', default=0.3, type=float, required=False)
    parser.add_argument('--Q_sub_keys_deleted', default=0.3, type=float, required=False)

    parser.add_argument('--is_cuda', default=True, type=bool, required=False)
    parser.add_argument('--gpu_id', default='1', type=str, required=False)
    parser.add_argument('--batch_size', default=32, type=int, required=False)

    return parser.parse_args()


def get_defined_keys(graph_triples):
    triple_keys = []
    defined_keys = {}
    for triple in graph_triples:
        if triple[0] not in triple_keys:
            triple_keys.append(triple[0])
        if triple[1] == ':instance':
            defined_keys[triple[0]] = triple

    return triple_keys, defined_keys


def create_new_key(prefix,existed_keys):
    new_key = prefix
    new_key_number = 1
    while new_key in existed_keys:
        new_key_number += 1
        new_key = prefix + str(new_key_number)

    return new_key


def get_key2token_indice_dict_and_aligned_token_indices(graph,tokens):
    key2token_indice_dict = {
        triple: list(graph.epidata[triple][-1].indices) if len(
            graph.epidata[triple]) > 0 and isinstance(
            graph.epidata[triple][-1], penman.layout.Alignment) else None for
        triple in graph.triples}
    all_alignment_token_indices = sum(list(filter(lambda x: x != None, key2token_indice_dict.values())), [])
    all_alignment_token_indices = list(
        filter(lambda x: x in all_alignment_token_indices, list(range(len(tokens)))))

    return key2token_indice_dict,all_alignment_token_indices


def get_phrases_and_evolved_indices2phrases_dict(nlp,origin_sent,all_alignment_token_indices,tokens):
    doc = nlp(origin_sent)
    all_phrases = []
    for sent in doc.sents:
        phrases = [[token.text_with_ws[0].lower() + token.text_with_ws[1:]
                    if token.is_sent_start and token.ent_type == 0 else
                    token.text_with_ws
                    for token in P]
                   for P in sent._.constituents]
        all_phrases.extend(phrases)

    token_indices2phrases_dict = {}
    for phrase in all_phrases:
        contrained_token_indices = []
        phrases_tokens = [token.lower().strip() for token in phrase]
        for indice in all_alignment_token_indices:
            if tokens[indice].lower() in phrases_tokens:
                contrained_token_indices.append(indice)
        # phrases_tokens_contrained_dict[phrase]=contrained_token_indices
        if len(contrained_token_indices) > 0:
            if token_indices2phrases_dict.get(tuple(contrained_token_indices)) == None:
                token_indices2phrases_dict[tuple(contrained_token_indices)] = [phrase]
            else:
                token_indices2phrases_dict[tuple(contrained_token_indices)].append(phrase)

    return all_phrases,token_indices2phrases_dict


def extract_answer_from_origin_sent(A_amr_graph, tokens, key2token_indice_dict, token_indices2phrases_dict):
    answer_contained_token_indices = []
    A_triples = penman.decode(A_amr_graph).triples
    for triple in A_triples:
        if key2token_indice_dict.get(triple) != None:
            answer_contained_token_indices.append(key2token_indice_dict[triple][0])
    answer_contained_token_indices.sort()

    answer_tokens = [tokens[indice].lower() for indice in answer_contained_token_indices]
    answer_contained_token_indices = []
    for indice in range(len(tokens)):
        if tokens[indice].lower() in answer_tokens:
            answer_contained_token_indices.append(indice)

    answer=None
    if len(answer_contained_token_indices) > 0:
        if token_indices2phrases_dict.get(tuple(answer_contained_token_indices)) != None:
            phrase = random.choice(token_indices2phrases_dict[tuple(answer_contained_token_indices)])
            answer = "".join(phrase).strip()

    return answer


def synthesize_self_train_data(args):
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

    sentences_datasets = load_from_disk(args.wiki_sentences_dataset_path)

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    if args.is_cuda:
        stog = amrlib.load_stog_model(device="cuda:" + str(args.gpu_id), batch_size=args.batch_size)
        gtos = amrlib.load_gtos_model(device="cuda:" + str(args.gpu_id), batch_size=args.batch_size, use_tense=False)
        # gtos = amrlib.load_gtos_model(device="cuda:" + str(args.gpu_id), batch_size=args.batch_size)
    else:
        stog = amrlib.load_stog_model(device="cpu")
        gtos = amrlib.load_gtos_model(device="cpu")


    def amr_based_synthesize(examples):
        sentences=[" ".join(" ".join([context,prefix_sent,last_sent]).strip().split())
                   for context,prefix_sent,last_sent in zip(examples['contexts'], examples['prefix_sents'], examples['last_sents'])]
        graphs_texts, graphs = text2amr(stog, sentences)

        Q_node_percent = args.Q_node_percent
        multi_Q_amr_graphs_for_sents = []
        multi_A_amr_graphs_for_sents = []
        multi_extracted_A_for_sents=[]
        sent_origin_amrs = []

        for graph_text, graph in zip(graphs_texts, graphs):
            if len(graph.triples) != len(graph.epidata.keys()) or len(graph.triples) <= 1:
                continue

            triples = graph.triples
            epidata = graph.epidata
            triple_keys, defined_keys = get_defined_keys(triples)
            triple_parents_dict=get_triple_parents_dict(triples,epidata)

            tokens = json.loads(graph.metadata['tokens'])
            for key in graph.epidata.keys():
                if key[2] in tokens and key[2] not in triple_keys and (
                        len(graph.epidata[key]) == 0 or isinstance(graph.epidata[key][-1],
                                                                   penman.layout.Alignment) == False):
                    graph.epidata[key].append(penman.layout.Alignment(indices=(tokens.index(key[2]),)))

            key2token_indice_dict, all_alignment_token_indices = get_key2token_indice_dict_and_aligned_token_indices(
                graph, tokens)
            origin_sent = graph.metadata['snt']
            all_phrases, token_indices2phrases_dict = get_phrases_and_evolved_indices2phrases_dict(nlp, origin_sent,
                                                                                                   all_alignment_token_indices,
                                                                                                   tokens)
            #

            amr_unknown_key=create_new_key('a',triple_keys)

            candidate_triples_indexes = []
            for triple_index, triple in enumerate(triples):
                if triple[1] == ':instance':
                    if triple_index == 0:
                        if '-' in triple[2] and triple[2].rindex('-') + 1 < len(triple[2]) and triple[2][
                                                                                               triple[2].rindex(
                                                                                                       '-') + 1:].isdigit():
                            candidate_triples_indexes.append(triple_index)
                    else:
                        candidate_triples_indexes.append(triple_index)

            if len(candidate_triples_indexes) == 0:
                continue

            Q_num = max(int(len(candidate_triples_indexes) * Q_node_percent), 1)
            Q_node_indexes = list(random.sample(candidate_triples_indexes, Q_num))

            Q_amr_graphs = []
            A_amr_graphs = []
            extracted_As=[]
            for Q_triple_index in Q_node_indexes:
                try:
                    # new_Q_amr_triples = triples
                    new_Q_amr_triples = []
                    new_A_amr_triples = []
                    # 考虑下面两种情况：
                    # 1.提问点为动词
                    #   1.提问点为动词，则动词处改为do-02，arg0保留，arg1改为amr-unknown，原arg1删除，更高数字的arg删除。
                    #   2.非arg关系概率概率保留，删除的所有concept节点将用于组建答案（包括点本身和被删除的子节点）
                    # 2.提问点为非动词，包括English Words，special keywords（special entity types，quantities and logical conjunctions）
                    #   1.其本身及其子节点全部删除作为答案
                    #   2.若点为root节点，则不将其放入候选问题节点中，防止问题长度为0

                    Q_triple = triples[Q_triple_index]
                    Q_key = Q_triple[0]

                    is_verb = False
                    if '-' in Q_triple[2] and Q_triple[2].rindex('-') + 1 < len(Q_triple[2]) and Q_triple[2][
                                                                                                 Q_triple[2].rindex(
                                                                                                         '-') + 1:].isdigit():
                        is_verb = True

                    if is_verb:
                        # 1.提问点为动词
                        #   1.提问点为动词，则动词处改为do-02，arg0保留，arg1改为amr-unknown，原arg1删除，更高数字的arg删除。
                        #   2.非arg关系概率概率保留，删除的所有concept节点将用于组建答案（包括点本身和被删除的子节点）

                        amr_do_key=create_new_key('d',triple_keys)

                        triples_for_Q=[]
                        arg0_exist=False
                        for triple in triples:
                            if triple==Q_triple:
                                triples_for_Q.append((amr_do_key,':instance','do-02'))
                                new_A_amr_triples.append(Q_triple)
                            elif triple[0]==Q_key and triple[1]==':ARG0':
                                arg0_exist=True
                                triples_for_Q.append(triple)
                            elif triple[0]==Q_key and\
                                    ':ARG' in triple[1] and triple[1][4:].isdigit() and int(triple[1][4:]) >= 2:
                                rel_root_key=triple[2]
                                rel_children_triples = list(
                                    filter(lambda x: x not in new_A_amr_triples and
                                                     rel_root_key in triple_parents_dict[x] and
                                                     (Q_triple_index == 0 or
                                                      (Q_key in triple_parents_dict[x] and
                                                       triple_parents_dict[x].index(Q_key) < triple_parents_dict[x].index(
                                                                  rel_root_key))),
                                           triples))
                                new_A_amr_triples.append(triple)
                                new_A_amr_triples.extend(rel_children_triples)
                            elif triple[0]==Q_key and (triple[1]==':ARG1' or ':ARG' not in triple[1]) and\
                                    random.random() < args.Q_sub_keys_deleted:
                                rel_root_key = triple[2]
                                rel_children_triples = list(
                                    filter(lambda x: x not in new_A_amr_triples and
                                                     rel_root_key in triple_parents_dict[x] and
                                                     (Q_triple_index == 0 or
                                                      (Q_key in triple_parents_dict[x] and
                                                       triple_parents_dict[x].index(Q_key) < triple_parents_dict[x].index(
                                                                  rel_root_key))),
                                           triples))
                                new_A_amr_triples.append(triple)
                                new_A_amr_triples.extend(rel_children_triples)
                            elif triple not in new_A_amr_triples:
                                triples_for_Q.append(triple)

                        for triple in triples_for_Q:
                            if triple==(amr_do_key,':instance','do-02') and arg0_exist==False:
                                new_Q_amr_triples.append((amr_do_key,':instance','do-02'))
                                new_Q_amr_triples.append((amr_do_key,':ARG1',amr_unknown_key))
                                new_Q_amr_triples.append((amr_unknown_key,':instance','amr-unknown'))
                            elif triple[0]==Q_key and triple[1]==':ARG0':
                                rel_root_key = triple[2]
                                rel_children_triples = list(
                                    filter(lambda x: x != (amr_do_key, ':instance', 'do-02') and
                                                     x not in new_Q_amr_triples and
                                                     rel_root_key in triple_parents_dict[x] and
                                                     (Q_triple_index == 0 or
                                                      (Q_key in triple_parents_dict[x] and
                                                       triple_parents_dict[x].index(Q_key) < triple_parents_dict[x].index(rel_root_key))),
                                           triples_for_Q))

                                new_Q_amr_triples.append(triple)
                                new_Q_amr_triples.extend(rel_children_triples)
                                new_Q_amr_triples.append((amr_do_key, ':ARG1', amr_unknown_key))
                                new_Q_amr_triples.append((amr_unknown_key, ':instance', 'amr-unknown'))
                            elif triple[0]==Q_key and triple[1]==':ARG1':
                                new_Q_amr_triples.append((amr_do_key,':ARG2',triple[2]))
                            elif triple not in new_Q_amr_triples:
                                new_Q_amr_triples.append(triple)

                        new_Q_amr_triples=list(map(lambda x:(amr_do_key,x[1],x[2]) if x[0]==Q_key else x,new_Q_amr_triples))
                        new_Q_amr_triples = list(
                            map(lambda x: (x[0], x[1], amr_do_key) if x[2] == Q_key else x, new_Q_amr_triples))

                    else:
                        # 2.提问点为非动词，包括English Words，special keywords（special entity types，quantities and logical conjunctions）
                        #   1.其本身及其子节点全部删除作为答案
                        #   2.若点为root节点，则不将其放入候选问题节点中，防止问题长度为0
                        new_A_amr_triples=list(filter(lambda x: Q_key in triple_parents_dict[x], triples))
                        is_named_entity = False
                        if Q_triple[2] in named_entities:
                            is_named_entity=True
                        for triple in triples:
                            if triple==Q_triple:
                                if is_named_entity:
                                    new_Q_amr_triples.append(triple)
                                    new_Q_amr_triples.append((triple[0], ':mod', amr_unknown_key))
                                new_Q_amr_triples.append((amr_unknown_key,':instance','amr-unknown'))
                            elif not is_named_entity and triple[0]==Q_key and triple not in new_A_amr_triples:
                                new_Q_amr_triples.append((amr_unknown_key,triple[1],triple[2]))
                            elif not is_named_entity and triple[2]==Q_key and triple not in new_A_amr_triples:
                                new_Q_amr_triples.append((triple[0],triple[1],amr_unknown_key))
                            elif triple not in new_A_amr_triples:
                                new_Q_amr_triples.append(triple)

                    # 删去Q中不连通的点
                    Q_keys, Q_defined_keys_list, Q_undefined_keys_list = get_tree_all_defined_undefined_keys(
                        new_Q_amr_triples, defined_keys)
                    deleted_connected_new_Q_amr_triples=delete_undefined_keys_in_tree_triples(new_Q_amr_triples,Q_undefined_keys_list)
                    new_Q_amr_graph = penman.graph.Graph(deleted_connected_new_Q_amr_triples)
                    # 暂时
                    uni_triples = []
                    for triple in deleted_connected_new_Q_amr_triples:
                        if triple not in uni_triples:
                            uni_triples.append(triple)
                    if len(uni_triples) != len(deleted_connected_new_Q_amr_triples):
                        print('repeated')
                        continue

                    A_keys, A_defined_keys_list, A_undefined_keys_list = get_tree_all_defined_undefined_keys(
                        new_A_amr_triples, defined_keys)
                    filled_connected_new_A_amr_triples = complete_undefined_keys_in_tree_triples(new_A_amr_triples,
                                                                                                 defined_keys, A_keys,
                                                                                                 A_defined_keys_list,
                                                                                                 A_undefined_keys_list)
                    new_A_amr_graph = penman.graph.Graph(filled_connected_new_A_amr_triples)

                    uni_triples=[]
                    for triple in filled_connected_new_A_amr_triples:
                        if triple not in uni_triples:
                            uni_triples.append(triple)
                    if len(uni_triples)!=len(filled_connected_new_A_amr_triples):
                        print('repeated')
                        continue

                    new_Q_amr_graph = penman.encode(new_Q_amr_graph)
                    new_A_amr_graph = penman.encode(new_A_amr_graph)
                    Q_amr_graphs.append(new_Q_amr_graph)
                    A_amr_graphs.append(new_A_amr_graph)
                    extracted_answer = extract_answer_from_origin_sent(new_A_amr_graph, tokens, key2token_indice_dict,
                                                                       token_indices2phrases_dict)
                    extracted_As.append(extracted_answer)

                except Exception:
                    continue

            if len(Q_amr_graphs)>0:
                multi_Q_amr_graphs_for_sents.append(Q_amr_graphs)
                multi_A_amr_graphs_for_sents.append(A_amr_graphs)
                multi_extracted_A_for_sents.append(extracted_As)
                sent_origin_amrs.append(graph)

        try:
            all_Q_amr_graphs_merged = sum(multi_Q_amr_graphs_for_sents, [])
            all_A_amr_graphs_merged = sum(multi_A_amr_graphs_for_sents, [])
            all_extracted_As_merged = sum(multi_extracted_A_for_sents, [])
            all_Q_texts_for_sents_merged = amr2text(gtos, all_Q_amr_graphs_merged)

            all_generated_needed_Amr_graphs_merged=list(zip(*list(filter(lambda x:x[1]==None,list(zip(all_A_amr_graphs_merged,all_extracted_As_merged))))))[0]
            all_generated_As_merged = amr2text(gtos, list(all_generated_needed_Amr_graphs_merged))
            all_A_texts_for_sents_merged=[]
            p=0
            for extracted_answer in all_extracted_As_merged:
                if extracted_answer==None:
                    all_A_texts_for_sents_merged.append(all_generated_As_merged[p])
                    p+=1
                else:
                    all_A_texts_for_sents_merged.append(extracted_answer)

            assert len(all_A_texts_for_sents_merged)==len(all_Q_texts_for_sents_merged)

            multi_Q_amr_graphs_saved_for_sents = []
            multi_A_amr_graphs_saved_for_sents = []
            sent_origin_amrs_saved=[]
            multi_Q_texts_for_sents = []
            multi_A_texts_for_sents = []
            origin_sents=[]

            p = 0
            for multi_Q_amr_graphs, multi_A_amr_graphs,sent_origin_amr in zip(multi_Q_amr_graphs_for_sents,
                                                              multi_A_amr_graphs_for_sents,sent_origin_amrs):
                multi_Q_amr_graphs_saved=[]
                multi_A_amr_graphs_saved = []

                multi_Q_texts = []
                multi_A_texts = []

                origin_sent = sent_origin_amr.metadata['snt']

                for Q_amr_graph, A_amr_graph in zip(multi_Q_amr_graphs, multi_A_amr_graphs):
                    #暂时
                    if all_Q_texts_for_sents_merged[p].strip().endswith('?'):
                        question=all_Q_texts_for_sents_merged[p]
                        answer=all_A_texts_for_sents_merged[p]

                        multi_Q_amr_graphs_saved.append(Q_amr_graph)
                        multi_A_amr_graphs_saved.append(A_amr_graph)

                        multi_Q_texts.append(question)
                        multi_A_texts.append(answer)


                    p += 1

                if len(multi_Q_texts)>0:
                    multi_Q_amr_graphs_saved_for_sents.append(multi_Q_amr_graphs_saved)
                    multi_A_amr_graphs_saved_for_sents.append(multi_A_amr_graphs_saved)
                    sent_origin_amrs_saved.append(penman.encode(sent_origin_amr))

                    multi_Q_texts_for_sents.append(multi_Q_texts)
                    multi_A_texts_for_sents.append(multi_A_texts)
                    origin_sents.append(origin_sent)

            result={#暂时
                    'multi_Q_amr_graphs_for_sents': multi_Q_amr_graphs_saved_for_sents,
                    'multi_A_amr_graphs_for_sents': multi_A_amr_graphs_saved_for_sents,
                    'multi_Q_texts_for_sents': multi_Q_texts_for_sents,
                    'multi_A_texts_for_sents': multi_A_texts_for_sents,
                    'sent_origin_amrs': sent_origin_amrs_saved,
                    'origin_sents': origin_sents
                    }
            return result
        except Exception:
            result={
                'multi_Q_amr_graphs_for_sents': [],
                'multi_A_amr_graphs_for_sents': [],
                'multi_Q_texts_for_sents': [],
                'multi_A_texts_for_sents': [],
                'sent_origin_amrs': [],
                'origin_sents': []
            }
            if args.decomposition_Q:
                result['multi_sub_Q_texts_for_sents'] = []

    amr_based_synthesize_QA_datasets = sentences_datasets.map(amr_based_synthesize,
                                                              batched=True,
                                                              # num_proc=args.preprocessing_num_workers,
                                                              remove_columns=['contexts', 'prefix_sents', 'last_sents'],
                                                              load_from_cache_file=not args.overwrite_cache,
                                                              #暂时
                                                              batch_size=50
                                                              #batch_size=10
                                                              )
    if not os.path.exists(args.synthesize_dataset_save_path):
        os.makedirs(args.synthesize_dataset_save_path)
    amr_based_synthesize_QA_datasets.save_to_disk(args.synthesize_dataset_save_path)
    return


if __name__ == "__main__":
    args = set_args()
    synthesize_self_train_data(args)