import json

from lib_parser import Parser

'''
Get the verb index from the tags with respect to a specific verb
'''
def get_verb_index(tags):
    for i, t in enumerate(tags):
        if t == "B-V":
            return i
    # There is no verb, abort
    return -1

'''
Get the temporal argument as a list of list of tokens
e.g., "Before I joined the army, I graduated in 2002" will return 
[
    ["Before", "I", "joined", "the", "army"],
    ["in", "2002"],
]
'''
def get_temporal_arguments(words, tags):
    ret = []
    for i, t in enumerate(tags):
        if t == "B-ARGM-TMP":
            end = -1
            for j in range(i+1, len(tags)):
                if tags[j] != "I-ARGM-TMP":
                    end = j
                    break
            ret.append(words[i:end])
    return ret


'''
Use a preprocessed SRL result to quickly iterate the parser
So that there is no need to run SRL every single time
'''
def run_over_srl_preprocessed_files(file_name, limit=1000):
    parser = Parser()
    all_srl_lines = [x.strip() for x in open(file_name).readlines()][:limit]
    for srl_line in all_srl_lines:
        srl_objects_in_current_line = json.loads(srl_line)
        for obj in srl_objects_in_current_line:
            # Here obj is a SRL model output
            words = obj['words']
            # iterate through all verbs (triggers) in the object
            for verb in obj['verbs']:
                verb_index = get_verb_index(verb['tags'])
                if verb_index < 0:
                    continue

                temporal_arguments = get_temporal_arguments(words, verb['tags'])
                # Use this next line to see what's inside, comment it out when done
                #print(temporal_arguments)

                for tmp_arg in temporal_arguments:
                    result = parser.parse_phrase(tmp_arg)

'''
take the token list and 2 verb indicies and 
'''

parser = Parser()
print(parser.parse_reference_date("I met john, 2002/02/02".split(" ")))
#run_over_srl_preprocessed_files("samples/wikipedia_temporal_related_srl_parsed_small.txt")