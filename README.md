# KAIROS Temporal Extraction

## Start CogcompNLP Backend
- Make sure model paths are configured at configs/model_paths.txt
- Make sure Gurobi license and environment works
- Run `pip install -r requirements.txt`
- Run `python server.py 1 [PORT]` for non-localhost service or `python server.py 0 [PORT]` for localhost service

## Querying the service
- Example: `curl --request POST --data '{"corpusId":"","id":"","text":"Bob attacks Kevin.\nKevin buys a gun.","tokens":["Bob","attacks","Kevin",".","Kevin","buys","a","gun","."],"sentences":{"generator":"UnsupervisedEventExtraction","score":1.0,"sentenceEndPositions":[18,35]},"views":[{"viewName":"Event_extraction","viewData":[{"viewType":"edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView","viewName":"event_extraction","generator":"cogcomp_kairos_event_ie_v1.0","score":1.0,"constituents":[{"label":"Conflict:Attack:Unspecified","score":1.0,"start":1,"end":2,"properties":{"sentence_id": 0, "SenseNumber":"01","predicate":["attacks"], "sentence_id": 0}},{"label":"Attacker","score":1.0,"start":0,"end":1},{"label":"Target","score":1.0,"start":2,"end":3},{"label":"Transaction:ExchangeBuySell:Unspecified","score":1.0,"start":1,"end":2,"properties":{"SenseNumber":"01","predicate":["buys"], "sentence_id": 1}},{"label":"Beneficiary","score":1.0,"start":0,"end":1},{"label":"Place","score":1.0,"start":2,"end":4}],"relations":[{"relationName":"Attacker","srcConstituent":0,"targetConstituent":1},{"relationName":"Target","srcConstituent":0,"targetConstituent":2},{"relationName":"Beneficiary","srcConstituent":3,"targetConstituent":4},{"relationName":"Place","srcConstituent":3,"targetConstituent":5}]}]}]}' -H "Content-type: application/json" http://URL:PORT/annotate`
- The service requires the input TextAnnotation to have view `event_extraction` as shown in the example above
