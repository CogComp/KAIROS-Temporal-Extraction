from flask import Flask
from flask import request
from flask import send_from_directory
from flask_cors import CORS

from lib_control import CogCompTimeBackend
from kairos_processor import process_kairos
import argparse
import sys
import json


class CogCompTimeDemoService:

    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.backend = CogCompTimeBackend()

    @staticmethod
    def handle_root(path):
        if path == "" or path is None:
            path = "index.html"
        return send_from_directory('./frontend', path)

    @staticmethod
    def info():
        return {"status":"online"}

    def halt(self):
        # exit(0)
        # self.app.shutdown()
        func = request.environ.get('werkzeug.server.shutdown')
        func()
        return {"status":"restarting"}

    @staticmethod
    def handle_index():
        return send_from_directory('./frontend', 'index.html')

    def tokenized_to_origin_span(self, text, token_list):
        text = text.replace("\n", "")
        token_span = []
        pointer = 0
        for token in token_list:
            while True:
                if token[0] == text[pointer]:
                    start = pointer
                    end = start + len(token) - 1
                    pointer = end + 1
                    break
                else:
                    pointer += 1
            token_span.append([start, end])
        return token_span

    def handle_request(self):
        args = request.get_json()
        text = args['text']
        order = self.backend.build_graph(text)
        return {
            "result": order,
        }

    def handle_uiuc_request(self):
        form = json.loads(request.data)
        stories = form['oneie']['en']['json']
        story_jsons = {}
        for s in stories:
            if len(stories[s]) == 0:
                continue
            if s not in story_jsons:
                story_jsons[s] = []
            ls = [x.strip() for x in stories[s].split("\n")]
            for lls in ls:
                try:
                    obj = json.loads(lls)
                    story_jsons[s].append(obj)
                except:
                    continue
        event_lines = [x.strip() for x in form['coref']['event.cs'].split("\n")]
        temporal_content = process_kairos(story_jsons, event_lines)
        form['temporal_relation']['en']['temporal_relation.cs'] = temporal_content
        return form

    def handle_json_request(self):
        args = request.get_json()
        tokens = args["tokens"]
        sent_ends = args["sentences"]["sentenceEndPositions"]
        sentences = []
        sentence = []
        for i, token in enumerate(tokens):
            sentence.append(token)
            if i + 1 in sent_ends:
                sentences.append(sentence)
                sentence = []
        views = args["views"]
        event_view = None
        event_view_id = None
        for i, view in enumerate(views):
            if view["viewName"] == "Event_extraction":
                event_view = view
                event_view_id = i
                break
        if event_view is None:
            return args
        event_triggers = []
        con_id_to_json_id = {}
        for i, constituent in enumerate(event_view["viewData"][0]['constituents']):
            start = constituent["start"]
            if "properties" in constituent:
                key = (constituent['properties']['sentence_id'], start)
                if key[0] > 0:
                    key = (key[0], start - sent_ends[key[0] - 1])
                event_triggers.append(key)
                if key not in con_id_to_json_id:
                    con_id_to_json_id[key] = []
                con_id_to_json_id[key].append(i)
        event_triggers = list(set(event_triggers))
        formatted_events = []
        for event in event_triggers:
            formatted_events.append(event)
        single_verb_map, relation_map = self.backend.build_graph_with_events(sentences, formatted_events, dct="2020-10-01")
        for back_id in single_verb_map:
            trigger_key = formatted_events[back_id]
            update_ids = con_id_to_json_id[trigger_key]
            for uid in update_ids:
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration"] = int(single_verb_map[back_id][1])
        for back_id_pair in relation_map:
            source = back_id_pair[0]
            dest = back_id_pair[1]
            for s_uid in con_id_to_json_id[formatted_events[source]]:
                for d_uid in con_id_to_json_id[formatted_events[dest]]:
                    args["views"][event_view_id]["viewData"][0]["relations"].append(
                        {
                            "relationName": relation_map[back_id_pair][0],
                            "srcConstituent": s_uid,
                            "targetConstituent": d_uid,
                            "properties": {"distance": int(relation_map[back_id_pair][1])}
                        }
                    )

        return args

    def handle_json_request_no_gurobi(self):
        args = request.get_json()
        token_to_span = self.tokenized_to_origin_span(args['text'], args['tokens'])
        tokens = args["tokens"]
        sent_ends = []
        for sent_end_char_id in args["sentences"]["sentenceEndPositions"]:
            for i, t in enumerate(token_to_span):
                if t[1] + 1 == sent_end_char_id:
                    sent_ends.append(i)
                    break
        sent_ends = list(set(sent_ends))
        accumulator = 0
        token_to_send_id = {}
        for i in range(0, len(tokens)):
            token_to_send_id[i] = accumulator
            if i in sent_ends:
                accumulator += 1
        sentences = []
        cur_sent_id = 0
        sentence = []
        for t in token_to_send_id:
            if token_to_send_id[t] == cur_sent_id:
                sentence.append(tokens[t])
            else:
                sentences.append(sentence)
                cur_sent_id = token_to_send_id[t]
                sentence = [tokens[t]]
        sentences.append(sentence)
        sentence_minus_val = {}
        accu = 0
        for i in range(0, len(sentences)):
            sentence_minus_val[i] = accu + len(sentences[i])
            accu = len(sentences[i])
        views = args["views"]
        event_view = None
        event_view_id = None
        for i, view in enumerate(views):
            if view["viewName"] == "Event_extraction":
                event_view = view
                event_view_id = i
                break
        if event_view is None:
            return args
        event_triggers = []
        con_id_to_json_id = {}
        for i, constituent in enumerate(event_view["viewData"][0]['constituents']):
            start = constituent["start"]
            if "properties" in constituent:
                key = (constituent['properties']['sentence_id'], start)
                if key[0] > 0:
                    key = (key[0], start - sent_ends[key[0] - 1])
                event_triggers.append(key)
                if key not in con_id_to_json_id:
                    con_id_to_json_id[key] = []
                con_id_to_json_id[key].append(i)
        event_triggers = list(set(event_triggers))
        formatted_events = []
        for event in event_triggers:
            formatted_events.append(event)
        single_verb_map, relation_map = self.backend.build_graph_with_events_no_gurobi(sentences, formatted_events, dct="2020-10-01")
        for back_id in single_verb_map:
            trigger_key = formatted_events[back_id]
            update_ids = con_id_to_json_id[trigger_key]
            for uid in update_ids:
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration"] = int(single_verb_map[back_id][1])
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_minute_prob"] = single_verb_map[back_id][2]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_hour_prob"] = single_verb_map[back_id][3]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_day_prob"] = single_verb_map[back_id][4]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_week_prob"] = single_verb_map[back_id][5]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_month_prob"] = single_verb_map[back_id][6]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_year_prob"] = single_verb_map[back_id][7]
                args["views"][event_view_id]["viewData"][0]["constituents"][uid]["properties"]["duration_decade_prob"] = single_verb_map[back_id][8]

        for back_id_pair in relation_map:
            source = back_id_pair[0]
            dest = back_id_pair[1]
            for s_uid in con_id_to_json_id[formatted_events[source]]:
                for d_uid in con_id_to_json_id[formatted_events[dest]]:
                    args["views"][event_view_id]["viewData"][0]["relations"].append(
                        {
                            "relationName": relation_map[back_id_pair][0],
                            "srcConstituent": s_uid,
                            "targetConstituent": d_uid,
                            "properties": {"distance": int(relation_map[back_id_pair][1])}
                        }
                    )

        return args

    def start(self, localhost=False, port=80, ssl=False):
        self.app.add_url_rule("/", "", self.handle_index)
        self.app.add_url_rule("/<path:path>", "<path:path>", self.handle_root)
        self.app.add_url_rule("/request", "request", self.handle_request, methods=['POST', 'GET'])
        self.app.add_url_rule("/request_temporal_json", "request_temporal_json", self.handle_json_request, methods=['POST', 'GET'])
        self.app.add_url_rule("/annotate", "annotate", self.handle_json_request, methods=['POST', 'GET'])
        self.app.add_url_rule("/annotate_no_gurobi", "annotate_no_gurobi", self.handle_json_request_no_gurobi, methods=['POST', 'GET'])
        self.app.add_url_rule("/request_uiuc_temporal", "request_uiuc_temporal", self.handle_uiuc_request, methods=['POST', 'GET'])
        self.app.add_url_rule("/info", "info", self.info, methods=['POST', 'GET'])
        self.app.add_url_rule("/halt", "halt", self.halt, methods=['POST', 'GET'])
        if ssl:
            if localhost:
                if port == 0:
                    self.app.run(ssl_context='adhoc')
                else:
                    self.app.run(ssl_context='adhoc', port=port)
            else:
                self.app.run(host='0.0.0.0', port=port, ssl_context='adhoc')
        else:
            if localhost:
                if port == 0:
                    self.app.run()
                else:
                    self.app.run(port=port)
            else:
                self.app.run(host='0.0.0.0', port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('host_mode', metavar='N', type=int)
    parser.add_argument('port', metavar='N', type=int)
    args = parser.parse_args()
    if args.host_mode == 0:
        local_host = True
        print("Initializing localhost")
    elif args.host_mode == 1:
        local_host = False
        print("Initializing non-localhost")
    else:
        print("Argument 1 out of parameter. Please use 0 for localhost and 1 for non-localhost.")
        sys.exit()
    print("on port {}".format(str(args.port)))

    service = CogCompTimeDemoService()
    service.start(localhost=local_host, port=args.port)
