from lib_control import CogCompTimeBackend


"""
An example main function
This first executes Alex's implementation of rule-based parsing (e.g., "2 days before")
Then runs probabilistic models from Zhou et al. 2020
The final results is an aggregation of the two, see line 28 for separate requests
"""
def run_example(sentences, indices, dct=None):
    backend = CogCompTimeBackend(config_lines=[
        "order_model	/shared/public/ben/final_matres",
        "distance_model	/shared/public/ben/final_distance",
        "duration_model	/shared/public/ben/final_duration",
    ])
    single_verb_map, relation_map = backend.build_graph_with_events_no_gurobi(sentences, indices, dct)
    """
    single_verb_map: 
        {
            event_id (index in the given indices): 
            [absolute_time (from rule-parser), duration_avg, a list of duration estimations of incremental units]
        }
    relation_map:
        {
            (event_id_1, event_id_2):
            ["before/after", distance_avg] 
        }
    if you only want rule-parser results (e.g., 2 days before), see L469 in lib_control.py
    """


run_example(
    [
        "I went to the park on January 1 .".split(),
        "I was really tired .".split(),
        "But luckily , I purchased enough food 2 days before I went to the park .".split(),
        "I wrote a review for the park and I plan to go again tomorrow .".split(),
    ],
    [(0, 1), (1, 1), (2, 4), (2, 11), (3, 1), (3, 11)],
    dct="2020-10-28"
)
