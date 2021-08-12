from tracie_model.start_predictor import Predictor
from lib_parser import PretrainedModel, AllenSRL, TimeStruct
import random
import spacy
import torch
from collections import defaultdict
from gurobi_graph import *

def get_verb_index(tags):
    for i, t in enumerate(tags):
        if t == "B-V":
            return i
    # There is no verb, abort
    return -1


def get_skeleton_phrase(tags, words, normalize_verb=False):
    ret = ""
    for i, tok in enumerate(words):
        if tags[i] != "O" and "ARGM-TMP" not in tags[i] and "B-V" not in tags[i]:
            ret += tok + " "
        if tags[i] == "B-V":
            # TODO: in the future
            if normalize_verb:
                ret += tok + " "
            else:
                ret += tok + " "
    return ret.strip()


def get_story(srl_objs, max_len=300):
    all_story = []
    all_story_length = 0
    for obj in srl_objs:
        all_story.append(" ".join(obj['words']))
        all_story_length += len(obj['words'])
    selected_set = set()
    while all_story_length > max_len:
        to_remove = random.choice(range(0, len(all_story)))
        if to_remove not in selected_set:
            selected_set.add(to_remove)
            all_story_length -= len(all_story[to_remove].split())

    final_story = ""
    for i, s in enumerate(all_story):
        if i not in selected_set:
            final_story += s + " "

    return final_story


def get_story_prev_next(srl_objs, obj_id, max_len=300):
    ret = []
    if obj_id >= 1:
        for w in srl_objs[obj_id - 1]['words']:
            ret.append(w)
    for w in srl_objs[obj_id]['words']:
        ret.append(w)
    if obj_id < len(srl_objs) - 1:
        for w in srl_objs[obj_id + 1]['words']:
            ret.append(w)
    if len(ret) > max_len:
        ret = ret[:max_len]
    return " ".join(ret)


class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def isCyclicUtil(self, v, visited, recStack):

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self.graph[v]:
            if visited[neighbour] == False:
                if self.isCyclicUtil(neighbour, visited, recStack) == True:
                    return True
            elif recStack[neighbour] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def isCyclic(self):
        visited = [False] * self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if visited[node] == False:
                if self.isCyclicUtil(node, visited, recStack) == True:
                    return True
        return False

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Push current vertex to stack which stores result
        stack.insert(0, v)

        # The function to do Topological Sort. It uses recursive

    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i, visited, stack)

                # Print contents of stack
        return stack


class CogCompTimeBackend:

    def __init__(self, config_lines=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = Predictor(config_lines)
        self.srl_model = PretrainedModel(
            'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
            'semantic-role-labeling'
        ).predictor()
        if self.device == 'cuda':
            self.srl_model._model = self.srl_model._model.cuda()
        self.spacy_model = spacy.load("en_core_web_sm", disable='ner')
        self.alex_srl = AllenSRL(server_mode=True)

    def parse_srl(self, text, pre_sentencized=False):
        if not pre_sentencized:
            doc = self.spacy_model(text)
            sentences = []
            for sent in doc.sents:
                toks = []
                for tok in sent:
                    toks.append(str(tok))
                sentences.append(toks)
        else:
            sentences = text

        srl_objs = []
        for sentence in sentences:
            srl_obj = self.srl_model.predict_tokenized(sentence)
            srl_objs.append(srl_obj)
        return sentences, srl_objs

    def extract_events(self, srl_objs):
        ret = {}
        cur_id = 0
        for i, obj in enumerate(srl_objs):
            for verb in obj['verbs']:
                verb_idx = get_verb_index(verb['tags'])
                if verb_idx > 0:
                    ret[cur_id] = [i, verb_idx, obj['words'][verb_idx]]
                    cur_id += 1
        return ret, cur_id

    def extract_events_given(self, given_events, sentences):
        ret = {}
        cur_id = 0
        for event in given_events:
            ret[cur_id] = [event[0], event[1], sentences[event[0]][event[1]]]
            cur_id += 1
        return ret

    def format_model_phrase(self, event, srl):
        phrase = ""
        for verb in srl['verbs']:
            if get_verb_index(verb['tags']) == event[1]:
                phrase = get_skeleton_phrase(verb['tags'], srl['words'])
        if phrase == "":
            phrase = event[2]
        return phrase

    def format_duration_phrase(self, event, srl):
        phrase = ""
        for verb in srl['verbs']:
            if get_verb_index(verb['tags']) == event[1]:
                phrase = get_skeleton_phrase(verb['tags'], srl['words'], normalize_verb=True)
        # TODO: normalize verb
        if phrase == "":
            phrase = event[2]
        return phrase

    def format_duration_phrase_marker(self, event, srl):
        phrase = ""
        for i, w in enumerate(srl['words']):
            if i == event[1]:
                phrase += "<extra_id_1> "
            phrase += w + " "
        return phrase.strip()

    '''
    input: edge map 
    {"0,1":0.1} means an edge from index 0 to index 1 with weight 0.1
    return: a list of sorted indices from ILP
    '''
    def ilp_sort(self, edges):
        output = gurobi_opt(edges).gurobi_output()
        g = Graph(output.shape[0])
        for i in range(0, output.shape[0]):
            for j in range(i+1, output.shape[0]):
                if output[i][j][0] == 1.0:
                    g.addEdge(i, j)
                else:
                    g.addEdge(j, i)
        return g.topologicalSort()

    def get_argmax_unit(self, probabilities):
        keys = {
            0: "minutes",
            1: "hours",
            2: "days",
            3: "weeks",
            4: "months",
            5: "years",
            6: "decades",
        }
        max_i = -1
        max_v = 0.0
        for i, v in enumerate(probabilities):
            if v > max_v:
                max_i = i
                max_v = v
        return keys[max_i]

    def get_averaged_val(self, probabilities):
        # in minutes
        values = {
            0: 1.0,
            1: 60.0,
            2: 24.0 * 60.0,
            3: 7.0 * 24.0 * 60.0,
            4: 30.0 * 24.0 * 60.0,
            5: 365.0 * 24.0 * 60.0,
            6: 10.0 * 365.0 * 24.0 * 60.0
        }
        for k in values:
            values[k] = math.log(values[k])
        s = 0.0
        for i, v in enumerate(probabilities):
            s += v * values[i]
        return math.exp(s)


    def get_simple_averaged_val(self, probabilities):
        counter = 0.0
        ret = 0.0
        for p in probabilities:
            ret += p * counter
            counter += 1.0
        return ret


    def get_averaged_val_simple_mean(self, probabilities):
        # in minutes
        values = {
            0: 1.0,
            1: 60.0,
            2: 24.0 * 60.0,
            3: 7.0 * 24.0 * 60.0,
            4: 30.0 * 24.0 * 60.0,
            5: 365.0 * 24.0 * 60.0,
            6: 10.0 * 365.0 * 24.0 * 60.0
        }
        s = 0.0
        for i, v in enumerate(probabilities):
            s += v * values[i]
        return s

    '''
    input:
    @sentences: a list of list of tokens. [['i', 'am', 'sentence', 'one'], ['i', 'am', 'sentence', 'two']]
    @indices: a list of verbs (sent_id, tok_id) [(0, 1), (1, 1)] for the two 'am's.
    return:
    @temporal_relation: a list of binary comparisons [(0, 1, distance), (1, 0, distance)]
    '''
    def build_graph_with_events(self, sentences, indices, dct=None):
        sentences, srl_objs = self.parse_srl(sentences, pre_sentencized=True)
        story = get_story(srl_objs)
        event_map = self.extract_events_given(indices, sentences)

        all_event_ids = list(event_map.keys())
        to_process_instances = []
        for event_id_i in all_event_ids:
            for event_id_j in all_event_ids:
                if event_id_i == event_id_j:
                    continue
                event_i = event_map[event_id_i]
                event_j = event_map[event_id_j]
                phrase_i = self.format_model_phrase(event_i, srl_objs[event_i[0]])
                phrase_j = self.format_model_phrase(event_j, srl_objs[event_j[0]])
                instance = "event: {} starts before {} story: {} \t nothing".format(phrase_i, phrase_j, story)
                to_process_instances.append(instance)

        to_process_duration = []
        for event_id_i in all_event_ids:
            event_i = event_map[event_id_i]
            phrase = self.format_duration_phrase_marker(event_i, srl_objs[event_i[0]])
            to_process_duration.append("event: {} story: {} \t nothing".format(phrase, story))

        results = self.predictor.predict(to_process_instances)
        results_distance = self.predictor.predict(to_process_instances, query_type="distance")
        results_duration = self.predictor.predict(to_process_duration, query_type="duration")

        duration_map = {}
        for i, event_id_i in enumerate(all_event_ids):
            # duration_map[event_id_i] = self.get_averaged_val(results_duration[i])
            duration_map[event_id_i] = self.get_simple_averaged_val(results_duration[i])

        edge_map = {}
        distance_map = {}
        it = 0
        tokens = []
        for obj in srl_objs:
            tokens.append(list(obj['words']))
        if dct is None:
            dct = TimeStruct(None, None, 1, 10, 2020)
        else:
            dct = TimeStruct(None, None, int(dct.split("-")[2]), int(dct.split("-")[1]), int(dct.split("-")[0]))
        self.alex_srl.get_graph(tokens, dct)
        for event_id_i in all_event_ids:
            for event_id_j in all_event_ids:
                if event_id_i == event_id_j:
                    continue
                prediction = results[it]
                prediction_distance = results_distance[it]
                distance_map[(event_id_i, event_id_j)] = prediction_distance
                it += 1
                event_i = event_map[event_id_i]
                event_j = event_map[event_id_j]
                timex_relation = self.alex_srl.compare_events(
                    (event_i[0], event_i[1]), (event_j[0], event_j[1])
                )

                if timex_relation is None:
                    if event_id_i < event_id_j:
                        key = "{},{}".format(str(event_id_i), str(event_id_j))
                        value = prediction[0]
                    else:
                        key = "{},{}".format(str(event_id_j), str(event_id_i))
                        value = prediction[1]
                else:
                    if timex_relation > 0:
                        timex_relation = 0.0
                    if timex_relation < 0:
                        timex_relation = 1.0
                    if event_id_i < event_id_j:
                        key = "{},{}".format(str(event_id_i), str(event_id_j))
                        value = float(timex_relation)
                    else:
                        key = "{},{}".format(str(event_id_j), str(event_id_i))
                        value = 1.0 - float(timex_relation)
                if key not in edge_map:
                    edge_map[key] = 0.0
                edge_map[key] += value
        directed_edge_map = {}
        for edge in edge_map:
            if edge_map[edge] < 1.0:
                key = "{},{}".format(edge.split(",")[1], edge.split(",")[0])
                directed_edge_map[key] = (2.0 - edge_map[edge]) / 2.0
            else:
                directed_edge_map[edge] = edge_map[edge] / 2.0

        sorted_edges = self.ilp_sort(directed_edge_map)
        print(sorted_edges)
        single_verb_map = {}
        relation_map = {}
        for i in range(0, len(sorted_edges)):
            input_arg = (event_map[sorted_edges[i]][0], event_map[sorted_edges[i]][1])
            timex = str(self.alex_srl.get_absolute_time(input_arg))
            duration = duration_map[sorted_edges[i]]
            single_verb_map[sorted_edges[i]] = [timex, duration]
            for j in range(i+1, len(sorted_edges)):
                # distance = self.get_averaged_val(distance_map[(sorted_edges[i], sorted_edges[j])])
                distance = self.get_simple_averaged_val(distance_map[(sorted_edges[i], sorted_edges[j])])
                relation_map[(sorted_edges[i], sorted_edges[j])] = ["before", distance]
                relation_map[(sorted_edges[j], sorted_edges[i])] = ["after", distance]
        return single_verb_map, relation_map

    '''
    A version of build_graph_with_events() without gurobi sorting for fast processing
    input:
    @sentences: a list of list of tokens. [['i', 'am', 'sentence', 'one'], ['i', 'am', 'sentence', 'two']]
    @indices: a list of verbs (sent_id, tok_id) [(0, 1), (1, 1)] for the two 'am's.
    return:
    @temporal_relation: a list of binary comparisons [(0, 1, distance), (1, 0, distance)]
    '''
    def build_graph_with_events_no_gurobi(self, sentences, indices, dct=None):
        sentences, srl_objs = self.parse_srl(sentences, pre_sentencized=True)
        story = get_story(srl_objs)
        event_map = self.extract_events_given(indices, sentences)

        all_event_ids = list(event_map.keys())
        to_process_instances = []
        for event_id_i in all_event_ids:
            for event_id_j in all_event_ids:
                if event_id_i == event_id_j:
                    continue
                event_i = event_map[event_id_i]
                event_j = event_map[event_id_j]
                phrase_i = self.format_model_phrase(event_i, srl_objs[event_i[0]])
                phrase_j = self.format_model_phrase(event_j, srl_objs[event_j[0]])
                instance = "event: {} starts before {} story: {} \t nothing".format(phrase_i, phrase_j, story)
                to_process_instances.append(instance)

        to_process_duration = []
        for event_id_i in all_event_ids:
            event_i = event_map[event_id_i]
            phrase = self.format_duration_phrase_marker(event_i, srl_objs[event_i[0]])
            prev_next_context = get_story_prev_next(srl_objs, event_i[0])
            to_process_duration.append("event: {} story: {} \t nothing".format(phrase, prev_next_context))

        results = self.predictor.predict(to_process_instances)
        results_distance = self.predictor.predict(to_process_instances, query_type="distance")
        results_duration = self.predictor.predict(to_process_duration, query_type="duration")

        duration_map = {}
        duration_map_probs = {}
        for i, event_id_i in enumerate(all_event_ids):
            duration_map[event_id_i] = self.get_averaged_val(results_duration[i])
            duration_map_probs[event_id_i] = results_duration[i]

        edge_map = {}
        distance_map = {}
        it = 0
        tokens = []
        for obj in srl_objs:
            tokens.append(list(obj['words']))
        if dct is None:
            dct = TimeStruct(None, None, 1, 10, 2020)
        else:
            dct = TimeStruct(None, None, int(dct.split("-")[2]), int(dct.split("-")[1]), int(dct.split("-")[0]))
        self.alex_srl.get_graph(tokens, dct)
        for event_id_i in all_event_ids:
            for event_id_j in all_event_ids:
                if event_id_i == event_id_j:
                    continue
                prediction = results[it]
                prediction_distance = results_distance[it]
                distance_map[(event_id_i, event_id_j)] = prediction_distance
                it += 1
                event_i = event_map[event_id_i]
                event_j = event_map[event_id_j]
                # Anchor: rule parser results below, please see comments at L590 in lib_parser.py
                timex_relation = self.alex_srl.compare_events(
                    (event_i[0], event_i[1]), (event_j[0], event_j[1])
                )

                if timex_relation is None:
                    if event_id_i < event_id_j:
                        key = "{},{}".format(str(event_id_i), str(event_id_j))
                        value = prediction[0]
                    else:
                        key = "{},{}".format(str(event_id_j), str(event_id_i))
                        value = prediction[1]
                else:
                    if timex_relation > 0:
                        timex_relation = 0.0
                    if timex_relation < 0:
                        timex_relation = 1.0
                    if event_id_i < event_id_j:
                        key = "{},{}".format(str(event_id_i), str(event_id_j))
                        value = float(timex_relation)
                    else:
                        key = "{},{}".format(str(event_id_j), str(event_id_i))
                        value = 1.0 - float(timex_relation)
                if key not in edge_map:
                    edge_map[key] = 0.0
                edge_map[key] += value
        directed_edge_map = {}
        for edge in edge_map:
            if edge_map[edge] < 1.0:
                key = "{},{}".format(edge.split(",")[1], edge.split(",")[0])
                directed_edge_map[key] = (2.0 - edge_map[edge]) / 2.0
            else:
                directed_edge_map[edge] = edge_map[edge] / 2.0

        single_verb_map = {}
        relation_map = {}
        for key in edge_map:
            e1 = int(key.split(",")[0])
            e2 = int(key.split(",")[1])
            input_arg = (event_map[e1][0], event_map[e1][1])
            t1 = str(self.alex_srl.get_absolute_time(input_arg))
            input_arg = (event_map[e2][0], event_map[e2][1])
            t2 = str(self.alex_srl.get_absolute_time(input_arg))
            d1 = duration_map[e1]
            d2 = duration_map[e2]
            distance = self.get_averaged_val(distance_map[(e1, e2)])
            relation_map[(e1, e2)] = ["before", distance]
            relation_map[(e2, e1)] = ["after", distance]
            if e1 not in single_verb_map:
                single_verb_map[e1] = [t1, d1] + duration_map_probs[e1]
            if e2 not in single_verb_map:
                single_verb_map[e2] = [t2, d2] + duration_map_probs[e2]

        return single_verb_map, relation_map

    def kairos_wrapper(self, json_obj):
        pass


if __name__ == "__main__":
    backend = CogCompTimeBackend()
    backend.build_graph_with_events(
        [
            "I went to the park on January 1 .".split(),
            "I was really tired .".split(),
            "But luckily , I purchased enough food 2 days before I went to the park .".split(),
            "I wrote a review for the park and I plan to go again tomorrow .".split(),
        ],
        [(0, 1), (1, 1), (2, 4), (2, 11), (3, 1), (3, 11)],
        dct="2020-10-28"
    )
