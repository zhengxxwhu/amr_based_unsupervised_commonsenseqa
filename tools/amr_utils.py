from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.graph_processing.annotator import add_lemmas
import penman.surface
import penman
from copy import deepcopy


def text2amr(stog, texts):
    graphs_texts = stog.parse_sents(texts)
    graphs = []
    aligned_graphs_texts = []
    for g_text in graphs_texts:
        try:
            penman_graph = add_lemmas(g_text, snt_key='snt')
            aligner = RBWAligner.from_penman_w_json(
                penman_graph)  # use this with a graph string that is properly annotated
            penman_graph = aligner.get_penman_graph()
            graphs.append(penman_graph)
            aligned_graphs_texts.append(g_text)

        except Exception:
            continue

    return aligned_graphs_texts, graphs


def amr2text(gtos, graphs):
    sents, _ = gtos.generate(graphs, disable_progress=True, use_tense=False)

    return sents


def get_tree_all_defined_undefined_keys(tree_triples,defined_keys):
    """
    子树内所有出现了的key，子树上定义了的key，子树上未定义的key（实际为子图）
    :param tree_triples: 子树的所有triples
    :param defined_keys: 完整图上定义的所有key和定义它们的triple
    :return:
    """
    tree_keys = []
    tree_defined_keys_list = []
    for triple in tree_triples:
        if triple[0] not in tree_keys:
            tree_keys.append(triple[0])
        if triple[2] not in tree_keys and triple[2] in defined_keys.keys():
            tree_keys.append(triple[2])
        if triple[1] == ':instance' and triple[0] not in tree_defined_keys_list:
            tree_defined_keys_list.append(triple[0])
    undefined_keys_list = list(filter(lambda x: x not in tree_defined_keys_list, tree_keys))
    return tree_keys,tree_defined_keys_list,undefined_keys_list


def delete_undefined_keys_in_tree_triples(tree_triples, undefined_keys_list):
    deleted_no_defined_simple_Q_amr_triples = []
    # 删除无定义的triple
    for triple in tree_triples:
        if triple[0] not in undefined_keys_list and triple[2] not in undefined_keys_list:
            deleted_no_defined_simple_Q_amr_triples.append(triple)

    return deleted_no_defined_simple_Q_amr_triples


#补全子树上出现的未定义的key的triple
def complete_undefined_keys_in_tree_triples(tree_triples, defined_keys, tree_keys=None, tree_defined_keys_list=None,
                                                undefined_keys_list=None):

    if tree_keys==None or tree_defined_keys_list==None or undefined_keys_list==None:
        tree_keys, tree_defined_keys_list, undefined_keys_list = get_tree_all_defined_undefined_keys(tree_triples,
                                                                                                     defined_keys)
    completed_tree_triples = []
    for triple in tree_triples:
        completed_tree_triples.append(triple)
        if triple[2] in undefined_keys_list:
            completed_tree_triples.append(defined_keys[triple[2]])
            undefined_keys_list.remove(triple[2])

    return completed_tree_triples


def get_key_basic_triples(root_key,full_graph_triples,defined_keys):
    """
    代表子图的基本triple，一般为子图root_key的instance。
    但若子图为named_entity，该named_entity的名称（name）也包含在内（name下的树不包含）
    :param root_key:
    :param full_graph_triples:
    :param defined_keys:
    :return:
    """
    root_tree_basic_triples = []
    name_key = None
    for triple in full_graph_triples:
        if triple[0] == root_key and triple[1] == ':instance':
            root_tree_basic_triples.append(triple)
        elif triple[0] == root_key and triple[1] == ':name':
            name_key = triple[2]
            root_tree_basic_triples.append(triple)
        elif name_key and triple[0] == name_key and (triple[1] == ':instance' or
                                                     (':op' in triple[1] and
                                                      triple[
                                                          2] not in defined_keys.keys())):
            root_tree_basic_triples.append(triple)

    return root_tree_basic_triples


def get_arg_rel_children_triples_dict(parent_key,full_graph_triples,with_other_rel=False):
    """
    返回点parent_key的所有属于arg关系的children点（即直接连接parent_key的keys，且关系为arg，即（parent_key,arg,key）。
    其中（key,arg-of,parent_key）的也算）
    :param parent_key:
    :param full_graph_triples:
    :return:
    """
    rel_children_keys = {}
    for triple in full_graph_triples:
        # if temp_triple[0] == Q_triple[0] and temp_triple[1] in [':ARG0', ':ARG1']:
        if triple[0] == parent_key and ':ARG' in triple[1]:
            rel_children_keys[triple[1]] = triple
        elif triple[2] == parent_key and ':ARG' in triple[1] and '-of' in triple[1]:
            reverse_rel = triple[1][:-3]
            rel_children_keys[reverse_rel] = (triple[2], reverse_rel, triple[0])
        elif with_other_rel and triple[0] == parent_key:
            rel_children_keys[triple[1]]=triple

    return rel_children_keys


def get_triple_parents_dict(graph_triples, graph_epidata):
    track = []
    triple_parents_dict = {}
    for triple in graph_triples:
        triple_parents_dict[triple] = deepcopy(track)

        actions = graph_epidata[triple]
        for action in actions:
            if isinstance(action, penman.layout.Push):
                track.append(action.variable)
            elif isinstance(action, penman.layout.Pop):
                track = track[:-1]

    return triple_parents_dict