"""
Adapted from 
- https://github.com/Maxlinn/CHAIR-metric-standalone/blob/master/chair.py
- https://github.com/shikiw/OPERA/blob/main/chair.py
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Set

import nltk
import tqdm
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def load_generated_captions(cap_file, image_id_key:str, caption_key:str):
    #Read in captions        
    # it should be list of dict
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unspported extension {ext} for cap_file: {cap_file}')

    # NOTE: list of str
    imids = [obj[image_id_key] for obj in caps]
    
    # list of str
    caps = [obj[caption_key] for obj in caps]
       
    return caps, imids


class CHAIR(object):

    def __init__(self, detections_path):

        self.imid_to_objects = defaultdict(list) # later become a dict of sets

        self.detections_path = detections_path

        #read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        #Some hard coded rules for implementing CHAIR metrics on MSCOCO
        
        #common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']
        
        #double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        self.get_annotations()


    def get_annotations(self):
        with open(self.detections_path, mode="r") as f:
            detections: Dict[str, List[str]] = json.load(f)

        # we expect the detections.json file to be a dict in which `sample_id: str` as key and `objects: List[str]` as value
        for objects in detections.values():
            assert set(objects) <= set(self.mscoco_objects)
            assert [self.inverse_synonym_dict[o] for o in objects] == objects
        self.imid_to_objects: Dict[str, Set[str]] = {sample_id: set(objects) for sample_id, objects in detections.items()}


    # def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):

    #     '''
    #     Meant to save time so imid_to_objects does not always need to be recomputed.
    #     '''
    #     #Read in captions        
    #     self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
    #     assert len(self.caps) == len(self.eval_imids)

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        # TODO: use a better format for answers file
        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        self.caps = []
        self.eval_imids = []
        with open(cap_file, mode="r") as f:
            list_of_dicts = json.load(f)
            for ans_dict in list_of_dicts[::10]: # FIXME: for mini-eval
                sample_id = list(ans_dict.keys())[0]
                answer = list(ans_dict.values())[0]
                self.caps.append(answer)
                self.eval_imids.append(sample_id)

        assert len(self.caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
    
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''
    
        #standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # words = [singularize(w) for w in words]
        words = lemmas_sent
    
        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
    
        #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']
    
        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words
    
    
    def compute_chair(self, cap_file, image_id_key, caption_key):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
        
        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        
        # :add:
        num_recall_gt_objects = 0.
        num_gt_objects = 0.

        output = {'sentences': []} 
        
        for i in tqdm.trange(len(caps)):
            cap :str = caps[i]
            imid :int = eval_imids[i]
    
            #get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid, 
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'words': raw_words 
                        }

            # :add:
            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0,
                                   'Recall': 0}
 
            #count hallucinated words
            coco_word_count += len(node_words) 
            hallucinated = False
            
            # add
            recall_gt_objects = set()
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)
    
            #count hallucinated caps
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1
            
            # add
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.
            
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
            
            # add
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
   
            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)
        # add
        recall = num_recall_gt_objects / num_gt_objects
    
        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall}
    
        return output 
    

    def compute_chairi_sample(self, caption_text: str, sample_id: str):
        """
        CHAIR-i for a single sample
        """

        #get all words in the caption, as well as corresponding node word
        words, node_words, idxs, raw_words = self.caption_to_words(caption_text)
        
        gt_objects = self.imid_to_objects[sample_id]
        
        hallucinated_word_count = 0
        hallucination_idxs = []
        mscoco_hallucinated_words = []
        recall_gt_objects = set()

        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                hallucinated_word_count += 1
                mscoco_hallucinated_words.append((word, node_word))
                hallucination_idxs.append(idx)
            else:
                recall_gt_objects.add(node_word)

        chair_i = 0.0
        recall = 0.0

        if len(words) > 0:
            chair_i = len(mscoco_hallucinated_words) / float(len(words))

        if len(gt_objects) > 0:
            recall = len(recall_gt_objects) / len(gt_objects)

        return {
            "chair_i": chair_i,
            "recall": recall,
            "mscoco_hallucinated_words": mscoco_hallucinated_words,
            "mscoco_gt_words": list(gt_objects),
            "mscoco_generated_words": list(node_words),
            "hallucination_idxs": hallucination_idxs,
            # needed for overall:
            "partial_hallucinated_word_count": hallucinated_word_count,
            "partial_coco_word_count": len(node_words),
            "partial_num_recall_gt_objects": len(recall_gt_objects),
            "partial_num_gt_objects": len(gt_objects)
        }


class ChairRewardModel:
    def __init__(self, detections_path):
        self.chair = CHAIR(detections_path)

    def compute_reward(self, caption_text, sample_id):
        result = self.chair.compute_chairi_sample(caption_text=caption_text, sample_id=sample_id)
        return dict(
            chair_i=result["chair_i"],
            recall=result["recall"]
        )

    def __call__(self, caption_texts: List[str], sample_ids: List[str]):
        rewards = []
        for caption_text, sample_id in zip(caption_texts, sample_ids, strict=True):
            reward = self.compute_reward(caption_text, sample_id)
            rewards.append(reward)

        return rewards