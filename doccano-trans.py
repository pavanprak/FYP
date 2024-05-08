import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import logging
import json

#filtter data to convert in spacy format
def fillterDoccanoData(doccano_JSONL_FilePath):
    try:
        training_data = []
        lines=[]
        with open(doccano_JSONL_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['data']
            entities = data['label']
            if len(entities)>0:
                training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception as e:
        logging.exception("Unable to process " + doccano_JSONL_FilePath + "\n" + "error = " + str(e))
        return None

#read Doccano Annotation file .jsonl
TRAIN_DATA=fillterDoccanoData("/.admin.jsonl") #root.jsonl is annotation file name file name 

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object
for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
        db.add(doc)
    except:
        print(text, annot)
db.to_disk("./train.spacy") # save the docbin object
