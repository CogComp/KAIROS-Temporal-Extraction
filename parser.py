import json
import random
import copy
import os
import spacy
import re
from word2number import w2n


def get_no_tmp_phrase(srl_obj):
    skip_list = []
    for verbs in srl_obj['verbs']:
        for i, t in enumerate(verbs['tags']):
            if "ARGM-TMP" in t:
                skip_list.append(i)
    ret = ""
    for i, w in enumerate(srl_obj['words']):
        if i not in skip_list:
            ret += w + " "
    return ret.strip()


def get_nagate_label(l):
    if l == "before":
        return "after"
    return "before"


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def format_train_roberta():
    #all_stories = [x.strip() for x in open("tmp_output.txt").readlines()]
    #all_srl = [x.strip() for x in open("wikipedia_srl_parsed.txt").readlines()]
    all_stories = []
    all_srl = []
    srl_map = {}
    f_out = open("t5_train_rep.txt", "w")
    for srl in all_srl:
        obj = json.loads(srl)
        for sent in obj:
            key = "".join(sent['words']).lower().replace(" ", "")
            srl_map[key] = sent

    for story in all_stories:
        all_sentences = []
        sentences = story.split("\t")
        if sentences[0].startswith("-----------"):
            continue
        use_no_tmp_sentence = True
        if random.random() < 0.333:
            use_no_tmp_sentence = False
        for m, sentence in enumerate(sentences):
            sent_key = sentence.replace(" ", "").lower()
            if sent_key not in srl_map:
                continue
            srl_obj = srl_map[sent_key]
            if use_no_tmp_sentence:
                new_sent = get_no_tmp_phrase(srl_obj)
            else:
                new_sent = sentence
            all_sentences.append(new_sent)

        order = list(range(0, len(all_sentences)))
        for i in range(0, 3):
            random.shuffle(order)
            if len(order) <= 3:
                continue
            story = [all_sentences[x] for x in order[0:-3]]
            if order[-3] < order[-2]:
                label_1 = "before"
            else:
                label_1 = "after"
            if order[-3] < order[-1]:
                label_2 = "before"
            else:
                label_2 = "after"
            final_label = "PRETRAIN-{}".format(len(story))
            story.append(all_sentences[order[-3]])
            story.append("starts " + label_1 + " " + all_sentences[order[-2]])
            story.append("ends " + label_2 + " " + all_sentences[order[-1]])
            story.append(final_label)
            f_out.write("\t".join(story) + "\n")


def format_train_pairwise_roberta():
    all_stories = [x.strip() for x in open("train_before_after_wiki.txt").readlines()] + [x.strip() for x in open("train_before_after_book.txt").readlines()]
    f_out = open("t5_train_pairwise.txt", "w")
    for line in all_stories:
        group = line.split("\t")
        if group[-1] == "before":
            first = group[0]
            second = group[1]
        else:
            first = group[1]
            second = group[0]
        if random.random() < 0.5:
            f = first
            s = second
            l = "before"
        else:
            f = second
            s = first
            l = "after"
        if random.random() < 0.5:
            left = "event: " + f + " starts " + l + " " + s
            out_label = "answer: positive"
        else:
            l = get_nagate_label(l)
            left = "event: " + f + " starts " + l + " " + s
            out_label = "answer: negative"
        f_out.write(left + "\t" + out_label + "\n")


def format_train_t5_paragraph():
    all_stories = [x.strip() for x in open("tmp_output.txt").readlines()]
    all_srl = [x.strip() for x in open("wikipedia_srl_parsed.txt").readlines()]
    srl_map = {}
    f_out = open("t5_wikiparagraph_rep_with_end.txt", "w")
    for srl in all_srl:
        obj = json.loads(srl)
        for sent in obj:
            key = "".join(sent['words']).lower().replace(" ", "")
            srl_map[key] = sent

    for story in all_stories:
        all_sentences = []
        sentences = story.split("\t")
        if sentences[0].startswith("-----------"):
            continue
        use_no_tmp_sentence = True
        if random.random() < 0.333:
            use_no_tmp_sentence = False
        for m, sentence in enumerate(sentences):
            sent_key = sentence.replace(" ", "").lower()
            if sent_key not in srl_map:
                continue
            srl_obj = srl_map[sent_key]
            if use_no_tmp_sentence:
                new_sent = get_no_tmp_phrase(srl_obj)
            else:
                new_sent = sentence
            all_sentences.append(new_sent)

        order = list(range(0, len(all_sentences)))
        if len(order) <= 3:
            continue
        for i in range(0, 5):
            random.shuffle(order)
            story = ["<s> " + all_sentences[x] + " </s>" for x in order[0:-2]]
            storyline = "story: " + " ".join(story)
            if order[-2] < order[-1]:
                label = "before"
            else:
                label = "after"
            answer = "answer: positive"
            if random.random() < 0.5:
                if label == "before":
                    label = "after"
                else:
                    label = "before"
                answer = "answer: negative"
            se_label = "starts"
            if random.random() < 0.3:
                se_label = "ends"
            event = "event: " + all_sentences[order[-2]] + " {} ".format(se_label) + label + " " + all_sentences[order[-1]]
            left = event + " " + storyline
            if len(left.split()) < 128:
                f_out.write(left + "\t" + answer + "\n")


def get_relevant_phrase(words, tags):
    ret = ""
    for i, t in enumerate(tags):
        if t != "O" and "ARGM-TMP" not in t:
            ret += words[i] + " "
    return ret.strip()


def get_verb_idx(tags):
    for i, t in enumerate(tags):
        if t == "B-V":
            return i
    return None


months = {'januray': 1, 'jan':1, 'feburary':2, 'feb':2, 'march':3,'mar':3,'april':4,'apr':4,'may':5,'june':6,'jun':6,'july':7,'jul':7,'august':8,'aug':8,'september':9,'sep':9,'october':10,'oct':10,'november':11,'nov':11,'december':12,'dec':12}


def get_int_val(tok):
    year = None
    try:
        year = int(tok)
    except:
        pass
    return year


class TimeStruct:
    def __init__(self, minute, hour, day, month, year):
        self.minute = minute
        self.hour = hour
        self.day = day
        self.month = month
        self.year = year

    def __str__(self):
        return "{} {} {} {}:{}".format(str(self.year), str(self.month), str(self.day), str(self.hour), str(self.minute))


def extract_in(toks):
    year = None
    month = None
    for i, t in enumerate(toks):
        if t == "in" and i < len(toks) - 1:
            if year is not None or month is not None:
                break
            if toks[i+1] in months:
                month = months[toks[i+1]]
                if i+2 < len(toks):
                    year = get_int_val(toks[i+2])
            else:
                year = get_int_val(toks[i+1])
            
    return TimeStruct(None, None, None, month, year)


def extract_on(toks):
    month = None
    year = None
    date = None
    for i, t in enumerate(toks):
        if t == "on":
            for j in range(i+1, min(i+5, len(toks))):
                if toks[j] in months:
                    month = months[toks[j]]
                else:
                    cur_tok = toks[j]
                    if cur_tok.endswith("th") or cur_tok.endswith("rd") or cur_tok.endswith("st"):
                        cur_tok = cur_tok[:-2]
                    intval = get_int_val(cur_tok)
                    if intval is not None:
                        if 1000 < intval < 3000:
                            year = intval
                        elif 0 < intval < 32:
                            date = intval
            if date != None or year != None or month != None:
                break
    return TimeStruct(None, None, date, month, year)


def extract_at(toks):
    hour = None
    minute = None
    for i, t in enumerate(toks):
        if t == "at" and i < len(toks) - 1:
            cr_tok = toks[i+1]
            pm_override = False
            found_unit = False
            if cr_tok.endswith("pm"):
                cr_tok = cr_tok[:-2]
                pm_override = True
                found_unit = True
            if cr_tok.endswith("am"):
                cr_tok = cr_tok[:-2]
                found_unit = True
            if ":" in cr_tok:
                hour = get_int_val(cr_tok.split(":")[0])
                if hour is not None and hour > 24:
                    hour = None
                minute = get_int_val(cr_tok.split(":")[1])
                if minute is not None and minute > 59:
                    minute = None
            else:
                hour = get_int_val(cr_tok)
                if hour is not None and hour > 24:
                    hour = None
                for j in range(i+1, min(i+6, len(toks))):
                    if toks[j] in ["am", "a.m", "a.m." "pm", "p.m", "p.m.", "afternoon", "morning", "day"]:
                        found_unit = True
                if not found_unit:
                    hour = None
            for j in range(i+1, min(i+6, len(toks))):
                if toks[j] in ["p.m", "p.m.", "pm", "afternoon"] or pm_override:
                    if hour is not None and hour < 12:
                        hour += 12
                    if hour is not None and hour > 24:
                        hour = None
    return TimeStruct(minute, hour, None, None, None)


def combine_timex(l):
    ret = TimeStruct(None, None, None, None, None)
    for t in l:
        if t.minute is not None:
            ret.minute = t.minute
        if t.hour is not None:
            ret.hour = t.hour
        if t.day is not None:
            ret.day = t.day
        if t.month is not None:
            ret.month = t.month
        if t.year is not None:
            ret.year = t.year
    return ret


def get_useful_count(timex):
    ret = 0
    if timex.minute is not None:
        ret += 1
    if timex.hour is not None:
        ret += 1
    if timex.day is not None:
        ret += 1
    if timex.month is not None:
        ret += 1
    if timex.year is not None:
        ret += 1
    return ret


def default_timex(timex):
    ret_cpy = copy.deepcopy(timex)
    month_mapping = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    if timex.year is None:
        ret_cpy.year = 2000
    if timex.month is None:
        ret_cpy.month = 1
    else:
        ret_cpy.month = month_mapping[timex.month]
    if timex.day is None:
        ret_cpy.day = 1
    if timex.hour is None:
        ret_cpy.hour = 0
    if timex.minute is None:
        ret_cpy.minute = 0
    return ret_cpy


def get_label(diff_in_hours):
    if 0 < diff_in_hours < 0.5:
        return "<extra_id_99>"
    if 0.5 <= diff_in_hours < 12.0:
        return "<extra_id_98>"
    if 12.0 <= diff_in_hours < 84.0:
        return "<extra_id_97>"
    if 84.0 <= diff_in_hours < 336.0:
        return "<extra_id_96>"
    if 336.0 <= diff_in_hours < 4320.0:
        return "<extra_id_95>"
    if 4320 <= diff_in_hours < 43800.0:
        return "<extra_id_94>"
    return "<extra_id_93>"


def calc_label(timex_1, timex_2):
    timex_1 = default_timex(timex_1)
    timex_2 = default_timex(timex_2)

    timex_1_val = timex_1.year * 8760.0 + (timex_1.month - 1) * 720.0 + (timex_1.day - 1) * 24.0 + timex_1.hour * 1.0 + (timex_1.minute / float(60.0))
    timex_2_val = timex_2.year * 8760.0 + (timex_2.month - 1) * 720.0 + (timex_2.day - 1) * 24.0 + timex_2.hour * 1.0 + (timex_2.minute / float(60.0))
    if timex_1_val == timex_2_val:
        return None, None
    if timex_1_val < timex_2_val:
        return "before", get_label(abs(timex_2_val - timex_1_val))
    else:
        return "after", get_label(abs(timex_2_val - timex_1_val))

#srl_objs is the prediction
def extract_timex(srl_objs):
    idx_accum = 0
    verb_phrase_to_tmp_map = {}
    paragraph = ""
    for srl_obj in srl_objs:
        for verb in srl_obj['verbs']:
            verb_phrase = get_relevant_phrase(srl_obj['words'], verb['tags'])
            tok_group = []
            for i, t in enumerate(verb['tags']):
                if "ARGM-TMP" in t:
                    tok_group.append(srl_obj['words'][i].lower())
            t_1 = extract_on(tok_group)
            t_2 = extract_in(tok_group)
            t_3 = extract_at(tok_group)
            timex = combine_timex([t_1, t_2, t_3])
            if get_verb_idx(verb['tags']) is None:
                continue
            map_idx = get_verb_idx(verb['tags']) + idx_accum
            verb_phrase_to_tmp_map[map_idx] = [verb_phrase, timex]
        idx_accum += len(srl_obj['words'])
        paragraph += " ".join(srl_obj['words']) + " "

    minute_record = None
    hour_record = None
    day_record = None
    month_record = None
    year_record = None
    final_list = []
    for i in range(0, idx_accum + 200):
        if i in verb_phrase_to_tmp_map:
            phrase, timex = verb_phrase_to_tmp_map[i]
            if timex.minute is not None:
                minute_record = timex.minute
            else:
                timex.minute = minute_record
            if timex.hour is not None:
                hour_record = timex.hour
            else:
                timex.hour = hour_record
            if timex.day is not None:
                day_record = timex.day
            else:
                timex.day = day_record
            if timex.month is not None:
                month_record = timex.month
            else:
                timex.month = month_record
            if timex.year is not None:
                year_record = timex.year
            else:
                timex.year = year_record
            if get_useful_count(timex) != 0 and len(phrase.split()) > 3:
                final_list.append([phrase, timex])

    counter_map = {
    }
    all_sentences = []
    for srl_obj in srl_objs:
        all_sentences.append(get_no_tmp_phrase(srl_obj))
    random.shuffle(all_sentences)
    concat = ""
    concat_counter = 0
    # concat = paragraph
    while len(concat.split()) < 128:
        if concat_counter >= len(all_sentences):
            break
        concat += all_sentences[concat_counter] + " "
        concat_counter += 1
    ret = []
    for i, (phrase_1, timex_1) in enumerate(final_list):
        for j in range(i+1, len(final_list)):
            phrase_2, timex_2 = final_list[j]
            tmp_label, dist_label = calc_label(timex_1, timex_2)
            if tmp_label is None:
                continue
            ret.append([concat, phrase_1, phrase_2, tmp_label, dist_label])
            if dist_label not in counter_map:
                counter_map[dist_label] = 0
            counter_map[dist_label] += 1
    return ret

def flip_label(l):
    if l == "before":
        return "after"
    return "before"


def format_train_t5_paragraph_with_distance():
    all_stories = [x.strip() for x in open("tmp_output.txt").readlines()]
    all_srl = [x.strip() for x in open("wikipedia_srl_parsed.txt").readlines()]
    srl_map = {}
    f_out = open("t5_wikiparagraph_with_distance.txt", "w")
    for srl in all_srl:
        obj = json.loads(srl)
        for sent in obj:
            key = "".join(sent['words']).lower().replace(" ", "")
            srl_map[key] = sent
    all_results = []
    for story in all_stories:
        sentences = story.split("\t")
        if sentences[0].startswith("-----------"):
            continue
        srl_objs = []
        for m, sentence in enumerate(sentences):
            sent_key = sentence.replace(" ", "").lower()
            if sent_key not in srl_map:
                continue
            srl_objs.append(srl_map[sent_key])
        all_results += extract_timex(srl_objs)
    key_limit = {}
    random.shuffle(all_results)
    for story, phrase_1, phrase_2, tmp_label, dist_label in all_results:
        if dist_label not in key_limit:
            key_limit[dist_label] = 0
        key_limit[dist_label] += 1
        if key_limit[dist_label] > 100000:
            continue
        right = "story: {}".format(story)
        if random.random() < 0.5:
            phrase_first = phrase_2
            phrase_second = phrase_1
            gold_label = flip_label(tmp_label)
            if random.random() < 0.5:
                display_label = gold_label
                answer_label = "positive"
            else:
                display_label = flip_label(gold_label)
                answer_label = "negative"
            left = "event: {} starts {} {}".format(phrase_first, display_label, phrase_second)
            answer = "answer: {} {}".format(answer_label, dist_label)
        else:
            phrase_first = phrase_1
            phrase_second = phrase_2
            gold_label = tmp_label
            if random.random() < 0.5:
                display_label = gold_label
                answer_label = "positive"
            else:
                display_label = flip_label(gold_label)
                answer_label = "negative"
            left = "event: {} starts {} {}".format(phrase_first, display_label, phrase_second)
            answer = "answer: {} {}".format(answer_label, dist_label)
        f_out.write(left + " " + right + "\t" + answer + "\n")


def stater():
    lines = [x.strip() for x in open("t5_train_combined_distance.txt").readlines()]
    max_len = 0
    all_len = 0.0
    for l in lines:
        if len(l.split()) > 200:
            max_len += 1
    print(max_len)


def recognize_num(s):
    try:
        _ = int(s)
        return True
    except:
        if s.lower() in ["a", "an", "several", "many", "some", "few", "couple", "of"]:
            return True
        else:
            try:
                a = w2n.word_to_num(s)
                if a is not None:
                    return True
                else:
                    return False
            except:
                return False


def match_for_pattern(path):
    file_paths = []
    for dirName, subdirList, fileList in os.walk(path):
        for subdir in subdirList:
            p = os.path.join(path, subdir)
            for d, s, f in os.walk(p):
                for ff in f:
                    file_path = os.path.join(p, ff)
                    file_paths.append(file_path)
    f_out = open("wikipedia_duration_paragraphs.txt", "w")
    for ii, f in enumerate(file_paths):
        if ii % 10 == 0:
            print(100.0 * float(ii) / float(len(file_paths)))
        documents = []
        cur_doc = []
        lines = [x.strip() for x in open(f).readlines()]
        for i, line in enumerate(lines):
            if line.startswith("<doc"):
                if len(cur_doc) > 0:
                    documents.append(cur_doc)
                    cur_doc = []
            else:
                cur_doc.append(line)
            if i == len(lines) - 1:
                cur_doc.append(line)
                documents.append(cur_doc)
                cur_doc = []
        for document in documents:
            for doc in document[1:-1]:
                doc = cleanhtml(doc)
                tokens = doc.split()
                first_non_num = -1
                valid = False
                for i, t in enumerate(tokens):
                    if t.lower() == "for":
                        for j in range(i+1, len(tokens)):
                            if not recognize_num(tokens[j]):
                                first_non_num = j
                                break

                if first_non_num > -1 and tokens[first_non_num].lower() in [
                    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month",
                    "months", "year", "years", "decade", "decades", "century", "centuries"
                ]:
                    valid = True
                if valid:
                    f_out.write(doc + "\n")


def gen_duration_srl():
    lines = [x.strip() for x in open("wikipedia_duration_paragraphs.txt").readlines()]
    nlp = spacy.load("en_core_web_sm", disable='ner')
    f_out = open("wikipedia_duration_to_srl.txt", "w")
    counter = {}
    for line in lines:
        doc = nlp(line)
        for sent in doc.sents:
            tokens = []
            for tt in sent:
                tokens.append(str(tt))
            first_non_num = -1
            for i, t in enumerate(tokens):
                if t.lower() == "for":
                    for j in range(i + 1, len(tokens)):
                        if not recognize_num(tokens[j]):
                            first_non_num = j
                            break

            if first_non_num > -1 and tokens[first_non_num].lower() in [
                "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month",
                "months", "year", "years", "decade", "decades", "century", "centuries"
            ]:
                if tokens[first_non_num].lower() not in counter:
                    counter[tokens[first_non_num].lower()] = 0
                counter[tokens[first_non_num].lower()] += 1
                f_out.write(" ".join(tokens) + "\n")
    print(counter)


def gen_filter_srl():
    #lines = [x.strip() for x in open("wikipedia_duration_to_srl.txt").readlines()]
    #f_out = open("wikipedia_duration_to_srl_real.txt", "w")
    lines = []
    f_out = []
    random.shuffle(lines)
    counter = {}
    for line in lines:
        tokens = line.split()
        first_non_num = -1
        for i, t in enumerate(tokens):
            if t.lower() == "for":
                for j in range(i + 1, len(tokens)):
                    if not recognize_num(tokens[j]):
                        first_non_num = j
                        break

        if first_non_num > -1 and tokens[first_non_num].lower() in [
            "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month",
            "months", "year", "years", "decade", "decades", "century", "centuries"
        ]:
            if tokens[first_non_num].lower() not in counter:
                counter[tokens[first_non_num].lower()] = 0
            counter[tokens[first_non_num].lower()] += 1
            if counter[tokens[first_non_num].lower()] > 10000:
                continue
            f_out.write(line + "\n")


gen_filter_srl()