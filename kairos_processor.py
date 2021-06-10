from lib_parser import PretrainedModel, AllenSRL, TimeStruct
from lib_control import get_story, get_verb_index, get_skeleton_phrase
from tracie_model.start_predictor import RelationOnlyPredictor
import random
from gurobi_graph import *
from lib_control import Graph
import torch


def read_tokens_file_source(file_id):
    lines = [x.strip() for x in open("kairos_data/input/ltf/{}.ltf.xml".format(file_id)).readlines()]
    all_tokens = []
    cur_tokens = []
    start_char_to_token = {}
    for line in lines:
        if line.startswith("</SEG>"):
            all_tokens.append(cur_tokens)
            cur_tokens = []
        if line.startswith("<TOKEN"):
            start_char = int(line.split()[4].split('"')[1])
            start_char_to_token[start_char] = (len(all_tokens), len(cur_tokens))
            token = line.split()[-1].split(">")[1].split("<")[0]
            cur_tokens.append(token)

    return all_tokens, start_char_to_token


def read_tokens_content_source(data_map, file_id):
    if file_id not in data_map:
        return None, None
    data = data_map[file_id]
    all_tokens = []
    start_char_to_token = {}
    for sent_id, sentence in enumerate(data):
        cur_tokens = []
        for i in range(0, len(sentence["tokens"])):
            start_char_to_token[int(sentence["token_ids"][i].split(":")[1].split("-")[0])] = (sent_id, i)
            cur_tokens.append(sentence["tokens"][i])
        all_tokens.append(cur_tokens)

    return all_tokens, start_char_to_token


def format_model_phrase(srl, verb_id, surface):
    phrase = ""
    for verb in srl['verbs']:
        if get_verb_index(verb['tags']) == verb_id:
            phrase = get_skeleton_phrase(verb['tags'], srl['words'])
    if phrase == "":
        phrase = surface
    return phrase


def process_kairos(data_map, lines):
    srl_model = PretrainedModel(
        'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
        'semantic-role-labeling'
    ).predictor()
    if torch.cuda.is_available():
        srl_model._model = srl_model._model.cuda()
    event_id_to_token_ids = {}
    added_story_ids = set()
    all_sentences = []
    predictor = RelationOnlyPredictor()
    out_lines = []
    for line in lines:
        groups = line.split("\t")
        if len(groups) < 2:
            continue
        event_id = groups[0]
        if groups[1] == "mention.actual":
            file_id = groups[3].split(":")[0]
            start_char = int(groups[3].split(":")[1].split("-")[0])
            doc_tokens, start_char_to_token = read_tokens_content_source(data_map, file_id)
            if doc_tokens is None:
                continue
            if event_id not in event_id_to_token_ids:
                event_id_to_token_ids[event_id] = []
            if file_id not in added_story_ids:
                added_story_ids.add(file_id)
                for tokens in doc_tokens:
                    all_sentences.append(tokens)
            token_id = start_char_to_token[start_char]
            event_id_to_token_ids[event_id].append(
                [doc_tokens, token_id]
            )
    all_srl_map = {}
    for i, tokens in enumerate(all_sentences):
        if i % 10 == 0:
            print("SRL Processed {}".format(str(float(i) / len(all_sentences))))
        all_srl_map[" ".join(tokens)] = srl_model.predict_tokenized(tokens)
    all_event_ids = list(event_id_to_token_ids.keys())
    for i in range(0, len(all_event_ids)):
        for j in range(i+1, len(all_event_ids)):
            i_events = event_id_to_token_ids[all_event_ids[i]]
            j_events = event_id_to_token_ids[all_event_ids[j]]
            random.shuffle(i_events)
            random.shuffle(j_events)
            if len(i_events) > 5:
                i_events = i_events[:10]
            if len(j_events) > 5:
                j_events = j_events[:10]

            to_process = []
            for i_t in i_events:
                for j_t in j_events:
                    story_raw_i = i_t[0]
                    story_raw_j = j_t[0]
                    i_srls = []
                    for t in story_raw_i:
                        i_srls.append(all_srl_map[" ".join(t)])
                    j_srls = []
                    for t in story_raw_j:
                        j_srls.append(all_srl_map[" ".join(t)])
                    story_i = get_story(i_srls, max_len=150)
                    story_j = get_story(j_srls, max_len=150)

                    phrase_i = format_model_phrase(i_srls[i_t[1][0]], i_t[1][1], story_raw_i[i_t[1][0]][i_t[1][1]])
                    phrase_j = format_model_phrase(j_srls[j_t[1][0]], j_t[1][1], story_raw_j[j_t[1][0]][j_t[1][1]])
                    instance = "event: {} starts before {} story: {} \t nothing".format(phrase_i, phrase_j, story_i + " " + story_j)

                    to_process.append(instance)
            results = predictor.predict(to_process)
            total_before = 0.0
            total_after = 0.0
            for r in results:
                total_before += r[0]
                total_after += r[1]
            prob_before = total_before / (total_before + total_after)
            prob_after = total_after / (total_before + total_after)
            if prob_before > prob_after:
                label = "TEMPORAL_BEFORE"
                prob = prob_before
            else:
                label = "TEMPORAL_AFTER"
                prob = prob_after
            out_lines.append("{}\t{}\t{}\t{}\n".format(all_event_ids[i], label, all_event_ids[j], str(prob)))
    del srl_model._model
    del predictor.model
    torch.cuda.empty_cache()
    return "".join(out_lines)


def ilp_sort(edges):
    output = gurobi_opt(edges).gurobi_output()
    g = Graph(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(i+1, output.shape[0]):
            if output[i][j][0] == 1.0:
                g.addEdge(i, j)
            else:
                g.addEdge(j, i)
    return g.topologicalSort()


def get_id_to_cluster():
    lines = [x.strip() for x in open("kairos_data/new/coref/event.cs").readlines()]
    id_map = {}
    k1 = ["K0C041O3D", "K0C047Z5A", "K0C041O37"]
    k2 = ["K0C041NHW", "K0C047Z57", "K0C041NHY", "K0C047Z59", "K0C041NHV"]
    for line in lines:
        group = line.split("\t")
        if len(group) > 3:
            if group[3].startswith("K0C"):
                gid = group[3].split(":")[0]
                if gid in k1:
                    id_map[group[0]] = 1
                if gid in k2:
                    id_map[group[0]] = 2
    return id_map


def close_constraint():
    lines = [x.strip() for x in open("results_new.txt").readlines()]
    directed_edge_map = {}
    f_out = open("result_constrained_1.txt", "w")
    id_to_event_id = {}
    id_map = get_id_to_cluster()
    for line in lines:
        group = line.split("\t")
        id_1 = int(group[0].split("_")[1])
        id_to_event_id[id_1] = group[0]
        id_2 = int(group[2].split("_")[1])
        id_to_event_id[id_2] = group[2]
        if id_map[group[0]] != 1 or id_map[group[2]] != 1:
            continue
        if group[1] == "TEMPORAL_BEFORE":
            key = "{},{}".format(str(id_1), str(id_2))
            directed_edge_map[key] = float(group[3])
        else:
            key = "{},{}".format(str(id_2), str(id_1))
            directed_edge_map[key] = float(group[3])

    s = ilp_sort(directed_edge_map)
    for i in range(0, len(s) - 1):
        id_1 = i
        id_2 = i + 1
        f_out.write(id_to_event_id[id_1] + "\tTEMPORAL_BEFORE\t" + id_to_event_id[id_2] + "\t1.0\n")

