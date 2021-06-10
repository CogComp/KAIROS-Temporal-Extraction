'''
Deprecated
'''


from tracie_model.start_predictor import Predictor


'''
Gets the binary prediction of two verbs and returns a binary relation without probability
'''
def get_start_relation_prediction_no_prob(story, srl_obj_verb_1, srl_obj_verb_2, verb_1_idx, verb_2_idx, predictor=None):
    if predictor is None:
        predictor = Predictor()
