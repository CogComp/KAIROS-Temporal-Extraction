import parser 
import re
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from datetime import date
from word2number import w2n
from datetime import datetime
'''
An example of the data structure, it's up to you to use it or not
'''
class TimeStruct:
    def __init__(self, minute, hour, day, month, year):
        self.minute = minute
        self.hour = hour
        self.day = day
        self.month = month
        self.year = year
        self.second = 0
    
    def __str__(self):
        return "({} {} {} {}:{}, {})".format(str(self.year), str(self.month), str(self.day), str(self.hour), str(self.minute),str(self.second))
    def get_date(self):
        convert_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 30.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "year": 365.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        }
        array = []
        if self.year == None:
            array.append(0)
        else:
            array.append(self.year)
        if self.month == None:
            array.append(0)
        else:
            array.append(self.month)
        if self.day == None:
            array.append(0)
        else:
            array.append(self.day)
        if self.hour == None:
            array.append(0)
        else:
            array.append(self.hour)
        if self.minute == None:
            array.append(0)
        else:
            array.append(self.minute)
        try:
            if self.second == None:
                array.append(0)
            else:
                array.append(self.second)
        except AttributeError:
            array.append(0)

        return array[0] * convert_map['years'] + (array[1]) * convert_map['months']+ array[2] * convert_map['days']  + array[3] * convert_map['hours'] + array[4] * convert_map['minutes'] + array[5]

    def is_empty(self):
        if self == None:
            return True
        return (self.year or self.day or self.month or self.hour or self.minute) == None

    def subtract(self,compare):
        # TODO we only want to subtract the things that they share, october 26 2002 and october 27 is most likeley just 1 day apart. 
        return TimeStruct.get_date(compare) - TimeStruct.get_date(self)

        return compare.get_date - self.get_date

    def copy(self):
        x = TimeStruct(self.minute, self.hour,self.day,self.month,self.year)
        return x


class EventObject:
    def __init__(self,location,verb,absolute_time = None, comparison_time = None,related_events = None):
        self.location = location 
        self.verb = verb
        self.absolute_time = absolute_time
        self.comparison_time = comparison_time
        self.related_events = related_events
    def __str__(self):
        return "{} {} {} {} {}".format(str(self.location), str(self.verb), str(self.absolute_time), str(self.comparison_time),str(self.related_events))
class ComparativeStruct:
    def __init__(self):
        # DIY
        pass

class Parser:
    def __init__(self):
        pass

    '''
    Input: a list of tokens.
    e.g., ["in", "2002"]
    '''
    def parse_phrase(self, tmp_arg):
        date_obj = self.parse_reference_date(tmp_arg)
        fuzzy_date_obj = self.parse_fuzzy_date(tmp_arg)
        duration_obj = self.parse_duration(tmp_arg)
        comparative_timepoint = self.parse_comparative_timepoint(tmp_arg)
        comparative_event = self.parse_comparative_event(tmp_arg)
        # Then do something
        # Eventually we want to return something meaningful to the caller
    months = {'januray': 1, 'jan':1, 'feburary':2, 'feb':2, 'march':3,'mar':3,'april':4,'apr':4,'may':5,'june':6,'jun':6,'july':7,'jul':7,'august':8,'aug':8,'september':9,'sep':9,'october':10,'oct':10,'november':11,'nov':11,'december':12,'dec':12}
    '''
    Parse explicit timepoint in the timex
    e.g., 2002, February, morning, 8 AM
    '''
            
    def parse_reference_date(self, tmp_arg):
        months = {'january': 1, 'jan':1, 'feburary':2, 'feb':2, 'march':3,'mar':3,'april':4,'apr':4,'may':5,'june':6,'jun':6,'july':7,'jul':7,'august':8,'aug':8,'september':9,'sep':9,'october':10,'oct':10,'november':11,'nov':11,'december':12,'dec':12}
        result = TimeStruct(None,None,None,None,None)
        t_1 = parser.extract_on(tmp_arg)
        t_2 = parser.extract_in(tmp_arg)
        t_3 = parser.extract_at(tmp_arg)
        result = parser.combine_timex([t_1, t_2, t_3])
        # check if it doenst have the signpost words 
        #problem: cant differentiate between day month and year if there is a number, is 9 september 2009 or the 9th day. 

        year = None
        month = None
        day = None
        timeh = None
        timem = None
        dateameric = re.compile("^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$")
        for i, x in enumerate(tmp_arg):
            try:
                float(x)
                #problem here
                # if result.year is None and int(x) > 1000 and int(x) < 3000:
                #     year = x

                # if result.day is None and int(x) < 32:
                #     day = x
            except ValueError:
                if result.month is None and x in months:
                    month = months[x]
                if (result.day is None) and (x.endswith('th') or x.endswith('st') or x.endswith('nd')):
                    day = x[:-2]
                    try:
                        float(day)
                    except ValueError:
                        day = None

                if dateameric.match(x):
                    if len(x.split('/')) > 1:
                        if(len(x.split('/'))== 2):
                            # TODO create regex that will accept 04/02
                            month = x[0]
                            year = x[1]
                        elif(len(x.split('/')) == 3):
                            day = x[0]
                            month = x[1]
                            year = x[2]
        if year is not None:
            result.year = year 
        if day is not None:
            result.day = day
        if month is not None:
            result.month = month
        
        
        if(TimeStruct.is_empty(result)):
            return None
        return result
            
    '''
    Parse implicit/fuzzy timepoint in the timex
    e.g., Friday, Thanksgiving
    For you to think: how to handle things like ~last Monday~?
    '''
    def parse_fuzzy_date(self, tmp_arg):
        return None

    '''
    Parse durations in the timex, you may do it later
    e.g., for 2 hours, for 3 weeks
    '''
    def parse_duration(self, tmp_arg):
        return None

    '''
    Parse comparative timepoints
    e.g., yesterday, tomorrow, next week, 2 days ago
    '''
    def parse_comparative_timepoint(self, tmp_arg):
        convert_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 30.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "year": 365.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        } 
        parity_map = {
            "later": -1,
            "after": -1,
            "next": -1,
            "since": -1,
            "past":-1,
            "then": -1,
            "prior": 1,
            "ago":1,
            "earlier": 1,
            "before": 1,
            "while": 0,
            "and": 0,
        }
        
        # assumption: that the word before days/years/etc. will be the number 
        #modifier = None represents that the modifier is plural but we have no idea. "i ate food days before i went to the doctor"
        time = 0
        for x in convert_map.keys():
            context = 0
            index_of_time = None
            modifier = 1
            if x in tmp_arg:
                if x[-1] == 's':
                    modifier = None
                index_of_time = tmp_arg.index(x)
                context = convert_map[x]
            if index_of_time != None:
                number = tmp_arg[index_of_time - 1]
                if modifier == None:
                    try:
                        modifier = w2n.word_to_num(number)
                    except ValueError:
                        # default for days, ( like maybe 3 days is good enough)
                        modifier = 3
            time += context * modifier
        #TODO check for but eg. I ate food before I played piano but after I killed someone. 
        # only want first one becasue we can just infer the other one. 
        parity = None
        for x in tmp_arg:
            if x in parity_map.keys():
                parity = parity_map[x]
                break

        try:
            return (parity,time)
        except TypeError:
            return None

    def parse_relative_timepoint(self,temp_arg):
        convert_map = {
            "second": 1.0,
            "seconds": 1.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hour": 60.0 * 60.0,
            "hours": 60.0 * 60.0,
            "day": 24.0 * 60.0 * 60.0,
            "days": 24.0 * 60.0 * 60.0,
            "week": 7.0 * 24.0 * 60.0 * 60.0,
            "weeks": 7.0 * 24.0 * 60.0 * 60.0,
            "month": 30.0 * 24.0 * 60.0 * 60.0,
            "months": 30.0 * 24.0 * 60.0 * 60.0,
            "year": 365.0 * 24.0 * 60.0 * 60.0,
            "years": 365.0 * 24.0 * 60.0 * 60.0,
            "decade": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "century": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
            "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
        } 
        convert_map = {
            "sunday": 24.0 * 60.0 * 60.0,
            "monday": 24.0 * 60.0 * 60.0,
            "tuesday": 24.0 * 60.0 * 60.0,
            "wensday" : 24.0 * 60.0 * 60.0,

        }
        parity_map = {
            "last":-1,
            "next":1,
            "this":0,
            "on":0,
            "later": -1,
            "after": -1,
            "next": -1,
            "since": -1,
            "past":-1,
            "then": -1,
            "prior": 1,
            "ago":1,
            "earlier": 1,
            "before": 1,
            "while": 0,
            "and": 0,
        }

    '''
    Parse comparative events, you may do it later
    e.g., before I graduated
    '''
    def parse_comparative_event(self, tmp_arg):
        return None

    '''
    Add other functions / cases 
    '''
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
    def __init__(self, server_mode=False):
        if server_mode:
            model = PretrainedModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                'semantic-role-labeling'
            )
        else:
            model = PretrainedModel('model.tar.gz',
                                    'semantic-role-labeling')

        #model = PretrainedModel('https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz',
                                #'semantic-role-labeling')
        self.predictor = model.predictor()
        self.graph = None
        self.doc_time = None
        #self.predictor._model = self.predictor._model.cuda()
        #self.output_path = output_path

    '''
    takes the words and tags to 
    return the temporal arguments if it exists
    '''
    def get_temporal_arguments(self, words, tags):
        ret = []
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP" or t == "I-ARGM-TMP":
                ret.append(words[i])
        return ret

    '''
    takes the words and tags
    returns the index (beggining,end) of the temporal argument
    '''
    def get_temporal_index(self, words, tags):
        ret = []
        temp = None
        for i, t in enumerate(tags):
            if t == "B-ARGM-TMP":
                end = i + 1
                for j in range(i+1, len(tags)):
                    if tags[j] == "I-ARGM-TMP":
                        end = j + 1
                    else:
                        break
                temp = (i,end)
                ret.append(temp)
        return ret

    '''
    takes the words and tags 
    returns the index of the verb
    '''
    def get_verb_index(self, words, tags):
        ret = None
        for i, t in enumerate(tags):
            if t == "B-V":
                ret = i
        return ret

    '''
    takes a sentence as a array(tokens), and the index of the target verb. replaces the 
    returns the timestructure of the predicted time that the verb happened
    '''
    def predict_absolute(self, words, verb_index, prediction,doctime):
        #TODO lets add more 
        anchorwords ={
            "today":0,
            "tommorow": 24.0 * 60.0 * 60.0,
            "yesterday": -1 * 24.0 * 60.0 * 60.0,
        }
        parser = Parser()
        # figure out which duplicate the verb is
        verb = words[verb_index]
        number = 0
        tempargs = []
        for i, x in enumerate(words):
            if x == verb and i != verb_index:
                number += 1
            elif i == verb_index:
                number += 1
                break
        
        # gets the temparg of the verb
        counter = 0
        for i in prediction['verbs']:
            if i['verb'] == verb:
                counter += 1
            if counter == number:
                tempargs = self.get_temporal_arguments(words,i['tags'])
                if len(tempargs) == 0:
                    return None
                else:

                    absolute =  parser.parse_reference_date(tempargs)

                    if absolute == None:
                        for x in tempargs:
                            if x in anchorwords:
                                
                                absolute = doctime
                                absolute.second += anchorwords[x]
                    return absolute

    '''
    takes the target timestruct and today and replaces the args that are non in the timestruct with today
    returns the updated timestruct
    '''
    def replace(self, temargs, today):
        if temargs == None:
            return None
        if temargs.year == None:
            temargs.year = today.year
        if temargs.month == None:
            temargs.month = today.month
        if temargs.day == None:
            temargs.day = today.day
        if temargs.hour == None:
            temargs.hour = today.hour
        if temargs.minute == None:
            temargs.minute = today.minute
        return temargs

    '''
    takes the verb index and the sentence and the document time and prediction 
    returns the relative time difference between this verb and the verbs its related to
    '''
    def predict_comparison(self, sentence, verb_index, prediction):
        parser = Parser()
        verb = sentence[verb_index]
        number = 0
        tempargs = []
        for i, x in enumerate(sentence):
            if x == verb and i != verb_index:
                number += 1
            elif i == verb_index:
                number += 1
                break
        counter = 0
        for i in prediction['verbs']:
            if i['verb'] == verb:
                counter += 1
            if counter == number:
                tempargs = self.get_temporal_arguments(sentence,i['tags'])
                if len(tempargs) == 0:
                    return None
                else:
                    return parser.parse_comparative_timepoint(tempargs)

    def predict_relative(self,sentence,verb_index,prediction):
        parser = Parser()
        verb = sentence[verb_index]
        number = 0
        tempargs = []
        for i, x in enumerate(sentence):
            if x == verb and i != verb_index:
                number += 1
            elif i == verb_index:
                number += 1
                break
        counter = 0
        for i in prediction['verbs']:
            if i['verb'] == verb:
                counter += 1
            if counter == number:
                tempargs = self.get_temporal_arguments(sentence,i['tags'])
                if len(tempargs) == 0:
                    return None
                else:
                    return parser.parse_comparative_timepoint(tempargs)
    '''
    takes in a array of sentence arrays 
    returns a array of event objects
    '''
    def get_graph(self, tokens1, doc_time, debugmode= False):
        tokens = []
        self.doc_time = doc_time
        for x in tokens1:
            temp = []
            for y in x:
                temp.append(y.lower())
            tokens.append(temp)
        graph = {}
        assumed_year = doc_time.year

        for i, sentence in enumerate(tokens):
            prediction = self.predictor.predict_tokenized(sentence)
            words = prediction['words']
            verb_relation = self.get_verbs(prediction)
 


            absolute = {}
            hasNone = True

            for verb_index in verb_relation.keys():
                temp = self.predict_absolute(sentence,verb_index,prediction,doc_time)
                if temp != None:
                    if temp.year != None:
                        assumed_year = temp.year
                    else:
                        temp.year = assumed_year
                    if not TimeStruct.is_empty(temp):
                        hasNone = False
                absolute[verb_index] = temp

            # 
            for verb_index in verb_relation.keys():
                if len(verb_relation[verb_index]) > 0:
                    if not (TimeStruct.is_empty(absolute[verb_index])) and TimeStruct.is_empty(absolute[verb_relation[verb_index][0]]):
                        comparison = self.predict_comparison(sentence, verb_index,prediction)
                        x = TimeStruct.copy(absolute[verb_index])
                        try:
                            if comparison[1] == 0:
                                None * 1
                            x.second += (comparison[0] * comparison[1]) 
                            absolute[verb_relation[verb_index][0]] = x
                        except TypeError:
                            if comparison[0] != None:
                                x.second += comparison[0]
                                absolute[verb_relation[verb_index][0]]= x
                    if TimeStruct.is_empty(absolute[verb_index]) and not TimeStruct.is_empty(absolute[verb_relation[verb_index][0]]):
                        comparison = self.predict_comparison(sentence, verb_relation[verb_index][0],prediction)
                        x = TimeStruct.copy(absolute[verb_relation[verb_index][0]])
                        try: 
                            if comparison[1] == 0:
                                None * 1
                            x.second += - (comparison[0] * comparison[1]) 
                            absolute[verb_index] = x
                        except TypeError:
                            if comparison[0] != None:
                                x.second += -1 *  comparison[0]
                                absolute[verb_index] = x
                    #TODO add a case where both have absolute times, so you fill in the things that are missng. 

            for verb_index in verb_relation.keys():
                graph[(i,verb_index)] = EventObject((i,verb_index),words[verb_index],absolute[verb_index],self.predict_comparison(sentence,verb_index,prediction),verb_relation[verb_index])
        self.graph = graph
        if debugmode:
            counter = 0
            for x in graph.keys():
                if (x[0] == counter):
                    print(tokens1[counter])
                    counter = counter + 1
                print(graph[x])

    '''
    takes in two verb indexes and 
    return the distance between the two, positive if the second is later than the first or None if there isn't a defined differenece
    '''
    def compare_events(self, verbinx1, verbinx2):
        graph = self.graph
        if (verbinx1 not in list(graph.keys())) or (verbinx2 not in list(graph.keys())):
            return None
        else:
            #case 1 one of the verbs is inside the othe verb and thus we shall return the comparitive if it existsm 
            #TODO make the subtract better
            if graph[verbinx1].absolute_time != None and graph[verbinx2].absolute_time != None:
                return TimeStruct.subtract(graph[verbinx1].absolute_time,graph[verbinx2].absolute_time)
            if verbinx1[0] == verbinx2[0]:
                if verbinx2[1] in graph[verbinx1].related_events:
                    try:
                        if graph[verbinx1].comparison_time[1] == 0:
                            return graph[verbinx1].comparison_time[0]
                        return graph[verbinx1].comparison_time[0] * graph[verbinx1].comparison_time[1]
                    except TypeError:
                        return None
                elif verbinx1[1] in graph[verbinx2].related_events:
                    try:
                        if graph[verbinx2].comparison_time[1] == 0:
                            return -1 * graph[verbinx2].comparison_time[0]
                        return graph[verbinx2].comparison_time[0] * graph[verbinx2].comparison_time[1]
                    except TypeError:
                        return None
            
        return None

    '''
    takes in a verbinx
    returns None if the verb index doesn't have a absolute time, returns keyerror
    '''
    def get_absolute_time(self, verbinx):
        graph = self.graph
        try:
            ret = graph[verbinx].absolute_time
            year = None
            month = None
            day = None
            hour = None
            min = None
            second = None
        
            if ret == None:
                return None
            if ret.year != None:
                year = ret.year
            else:
                year = "xxxx"
            if ret.month != None:
                month = ret.month
            else:
                month = "xx"
            if ret.day != None:
                day = ret.day
            else:
                day = "xx"
            if ret.minute != None:
                min = ret.minute
            else:
                min = "xx"
            if ret.hour != None:
                hour = ret.hour
            else:
                hour = "xx"
            try:
                if self.second != None:
                    second = ret.second
                else:
                    second = "x"
            except AttributeError:
                second = "x"
            return "({}/{}/{} {}:{} + {})".format(str(month), str(day), str(year), str(hour), str(min),str(second))
             
        except KeyError:
            return None
            
    '''
    takes a predicition object from a single sentence and
    returns a dict of the verbs and an array of the verbs contained in 
    '''
    def get_verbs(self, prediction):
        words = prediction['words']
        ret = {}
        verb_map = {}
        for verb_obj in prediction['verbs']:
            tags = verb_obj['tags']
            verb_map[verb_obj['verb']] = (self.get_verb_index(words,tags),self.get_temporal_index(words,tags))
            ret[verb_map[verb_obj['verb']][0]] = []
        for verb in verb_map.items():
            if len(verb[1][1]) != 0:
                for testverb in verb_map.items():
                    verblocation = verb[1][0]
                    testverblocation = testverb[1][0]
                    if verblocation != testverblocation:
                        for x in verb[1][1]:
                            if testverb[1][0] >= x[0] and testverb[1][0] < x[1]:
                                ret[verb[1][0]].append(testverb[1][0])

        return ret


if __name__ == "__main__":
    srl = AllenSRL()
    doctime = TimeStruct(None,None,None,None,2002)
    srl.get_graph([['From', 'a', 'financial', 'standpoint', 'what', 'is', 'most', 'noteworthy', 'is', 'that', 'the', 'combined', 'debt', 'of', 'the', 'Cypriot', 'people', 'companies', 'and', 'government', 'is', '26', 'times', 'the', 'size', 'of', 'the', 'countrys', 'gross', 'domestic', 'product', 'Only', 'Ireland', 'still', 'struggling', 'to', 'recover', 'from', 'the', 'banking', 'collapse', 'that', 'required', 'an', 'international', 'bailout', 'in', '2010', 'has', 'a', 'higher', 'debttoGDP', 'ratio', 'among', 'euro', 'zone', 'countries']],doctime)
    #srl.get_graph(["I cheated on my girlfriend before we celebrated our anniversary".split(" ")],"hey")
    doctime = TimeStruct(None,None,None,None,2002)
    #srl.get_graph(["I ate food on october 5".split(), "I ran on october 10".split()], doctime)
    x = srl.compare_events((0,4),(0,11))
    print(x)
    #print(srl.comparison_predict(["I ate dinner on october 26 2002".split(" "),"I ran outside on october 25 2002".split(" ")],(0,1),(1,1)))
