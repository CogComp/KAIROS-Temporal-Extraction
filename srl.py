
#import jsonlines
import time
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import sys
import os
import argparse
import torch
import parser

torch.set_num_threads(1)

class PretrainedModel:
    """
    A pretrained model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor.
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)

class AllenSRL:

    def __init__(self):
        model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                'semantic-role-labeling')
        self.predictor = model.predictor()
        #self.predictor._model = self.predictor._model.cuda()
        #self.output_path = output_path

    def predict(self,tokens):
        prediction = self.predictor.predict_tokenized(tokens)
        #predictionarray = [prediction]
        #result = parser.extract_timex(predictionarray)
        print(prediction)
        tags = prediction['verbs'][0]['tags']
        words = prediction['words']
        tempargs = ""
        for x in range(len(tags)):
            if tags[x] == 'I-ARGM-TMP':
                tempargs = tempargs + " " + words[x]
            if tags[x] == 'B-ARGM-TMP':
                tempargs = words[x]

srl = AllenSRL()


srl.predict("I ate dinner, 02/04/2002".split(" "))
