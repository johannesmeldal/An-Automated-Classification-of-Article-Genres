# Reuters is a dirty data set and requires sanitation. This script completes the necessary clean up and loads it into a DataFrame.
# The following clean up steps are performed:
# -prints the id for each article and whether it is a training/test set
# -creates an object for each article to store appropriate values
# -creates a dictionary with key (tag), value (contents) pairs using the article object
# -XML lists are surrounded by the "D" tag so the SAX parser interprets all tags with lists as the same value. Therefore, if a tag has a list,
# the overall tag for the section is distinguished and the value is placed with an overall tag in the dictionary.
# -cleans up the values of the BODY and TITLE tags since SAX parses the content according to the '\n' character
# -loads CSVs for the BODY and TITLE of each document ID, which will be used as features
# -creates a third CSV for teh most relevant TOPIC corresponding to each doc ID because multiple topics for each doc ID lead to a mismatch in
# labels and features. The most relevant TOPIC is isolated by determining the most frequent topics in the entire corpus. The most frequent of 
# the TOPICS in each document, according to the dictionary of most frequent topics of the corpus, is then used.
# -the csv also indicates whether the document was used as a training/test in the LEWISSPLIT separation.
# The path starts from the current root and accesses the 22 sgml files stored locally in reuters21578
#
# To run the script:
# 	$ python reuters_loader.py

#common packes for extracting and transforming Machine Learning data
from collections import defaultdict
import csv
import json
import numpy as np
import os
import pandas as pd # type: ignore
import xml.sax #for markup language transformation
from xml.sax.handler import ContentHandler
 
# object that holds all values for each individual article
class Article(object):
	#each value is initialized to an empty value because not every article has each tag/content present
	def __init__(self):
		self.body = ""
		self.date = ""
		self.title = ""
		self.topics = []
		self.places = []
		self.people = []
		self.author = ""
		self.dateline = ""
		self.exchanges = []
		self.companies = []
		self.orgs = []
		self.mknote = ""

class MyHandler(ContentHandler):
    def __init__(self):
        self.count = 1
        self.tag = ''
        self.defaultdict = defaultdict(list)
        self.topdict = defaultdict(list)
        self.boddict = defaultdict(list)
        self.titledict = defaultdict(list)
        self.lewissplit = list()
        self.article = Article()
        self.df = pd.DataFrame()
        self.in_d = False
        self.docID = 0
        self.lewis = "train"
        self.freqDict = defaultdict(int)
        # get frequency count of most popular topics
        file = open("all-topics-strings.lc.txt")
        for word in file.read().split():
            self.freqDict[word] += 1

    # booleans that keep track of the tags that SAX does not parse correctly and therefore need to be cleaned
    def _reset(self):
        self.in_places = False
        self.in_topics = False
        self.in_people = False
        self.in_exchanges = False
        self.in_companies = False
        self.in_orgs = False

    def startElement(self, name, attrs):
        self.tag = name

        # each REUTERS tag has attributes that describe the presence of topics, the id, and whether it's a training/test set
        if name == "REUTERS":
            self.docID = attrs.getValue("NEWID")
            if attrs.getValue("LEWISSPLIT") == "TRAIN":
                self.lewis = "lewis_train"
            elif attrs.getValue("LEWISSPLIT") == "TEST":
                self.lewis = "lewis_test"
        # the following statements denote the presence of the real tag associated with the "D" tag
        elif name == "D":
            self.in_d = True
        elif name == "TOPICS":
            self._reset()
            self.in_topics = True
        elif name == "PLACES":
            self._reset()
            self.in_places = True
        elif name == "PEOPLE":
            self._reset()
            self.in_people = True
        elif name == "EXCHANGES":
            self._reset()
            self.in_exchanges = True
        elif name == "COMPANIES":
            self._reset()
            self.in_companies = True
        elif name == "ORGS":
            self._reset()
            self.in_orgs = True

    # add the value associated with each tag to a dictionary
    def characters(self, content):
        # eliminates newline character because the parser does not
        if content != '\n':
            # groups lines of text into one BODY, TITLE, and MKNOTE per article
            if self.tag == "BODY":
                self.article.body += " " + content
            elif self.tag == "TITLE":
                self.article.title += " " + content
            elif self.tag == "MKNOTE":
                self.article.mknote += " " + content
        # labels each of the lists according to their real tag and eliminates the "D" tag
        elif self.in_d:
            if self.in_places:
                self.article.places.append(content)
            elif self.in_people:
                self.article.people.append(content)
            elif self.in_topics:
                self.article.topics.append(content)
            elif self.in_exchanges:
                self.article.exchanges.append(content)
            elif self.in_companies:
                self.article.companies.append(content)
            elif self.in_orgs:
                self.article.orgs.append(content)
        # handles the remaining standard parsing
        elif self.tag == "DATE":
            self.article.date = content
        elif self.tag == "AUTHOR":
            self.article.author = content
        elif self.tag == "DATELINE":
            self.article.dateline = content

    # resets content before moving to the next element
    def endElement(self, name):
        # Each REUTERS tag signals the start of a new article
        if name == "REUTERS":
            # Add each key, value pair to a dictionary only if it's a complete article
            if self.article.body != "" and self.article.topics and self.article.title:
                # Isolate most popular topic
                popularTopic = self.article.topics[0]
                if len(self.article.topics) > 1:
                    for topic in self.article.topics:
                        if topic in self.freqDict and self.freqDict[topic] > self.freqDict[popularTopic]:
                            popularTopic = topic
                # Dictionaries for individual features
                self.topdict[self.docID].append(popularTopic)
                self.boddict[self.docID].append(self.article.body)
                self.titledict[self.docID].append(self.article.title)
                self.lewissplit.append(self.lewis)

                # Store the article data in the defaultdict
                self.defaultdict["BODY"].append(self.article.body)
                self.defaultdict["DATE"].append(self.article.date)
                self.defaultdict["TITLE"].append(self.article.title)
                self.defaultdict["TOPICS"].append(self.article.topics)
                self.defaultdict["PLACES"].append(self.article.places)
                self.defaultdict["PEOPLE"].append(self.article.people)
                self.defaultdict["AUTHOR"].append(self.article.author)
                self.defaultdict["DATELINE"].append(self.article.dateline)
                self.defaultdict["EXCHANGES"].append(self.article.exchanges)
                self.defaultdict["COMPANIES"].append(self.article.companies)
                self.defaultdict["ORGS"].append(self.article.orgs)
                self.defaultdict["MKNOTE"].append(self.article.mknote)

                # Create a new Article object for the next article
                self.article = Article()
                self.in_d = False


            


# main function:
directory = './data'
files = os.listdir(directory)

# set up handler and parser
parser = xml.sax.make_parser()
handler = MyHandler()
parser.setContentHandler(handler)

# loop through 22 files
for f in files:
	cur_file = directory + "/" + f
	#only open sgm files
	if f.endswith('.sgm') and f.startswith('reut'):
		parser.parse(cur_file)

# csv, with doc id and "LEWISSPLIT" train/test information 
def create_csv(title, dictionary):
	values = [val for sublist in dictionary.values() for val in sublist]
	keys = dictionary.keys()
	newRow = [handler.lewissplit, keys, values]
	with open(title, 'w') as output:
		writer = csv.writer(output, lineterminator = '\n')
		writer.writerows(zip(*newRow))
	output.close()

create_csv("topics_popular.csv", handler.topdict)
create_csv("body_no_null.csv", handler.boddict)
create_csv("title_no_null.csv", handler.titledict)
