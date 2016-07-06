from guineapig import *
import math
import re

# supporting routines can go here
def tokens(line):
    parts = line.split("\t")
    labels = parts[1].split(",")
    words = re.split("\W",parts[2])
        #parts[2].split(" ")
    # 1. Y=*
    yield "Y=*"
    for label in labels:
        # 2. Y=y
        yield "Y*"+label.lower()+"=*"
        for word in words:
            if word != "" and word not in {"a","and","the","is","or","are","am"}:
                #word = word.replace("\\W","")
                # 3. X=x,Y=y
                yield "X="+word.lower()+"^Y="+label.lower()
                # 4. X=*,Y=y
                yield "X=*^Y="+label.lower()
                #yield (label.lower(),word.lower())

def tokens2(line):
    parts = line.split("\t")
    id = parts[0]
    labels = parts[1].split(",")
    words = re.split("\W",parts[2])
    # for each word, generate a request in the form of (word, docID)
    for word in words:
        if word != "" and word not in {"a","and","the","is","or","are","am"}:
            yield (word,id)

def prediction(view):
    qy = float(20)
    qx = float(1000)
    row = view
    #rows = GPig.rowsOf(view) # a list of tuples
    #for row in rows:
    maxP = - float("inf")
    maxL = ""
    id = row[0][0] # test doc id
    words = row[0][1] # words found in document
    counts = row[1][1] # found from training
    for count in counts:
        if count[0].startswith("Y=*"):
            C_Ystar = count[1]                                  # Y*
    for count in counts:
        if count[0].startswith("Y*"):
            C_Yy = count[1]                                     # Yy
            label = count[0].split("=")[0].split("*")[1]
            prob = math.log((1.0*(C_Yy+1))/(C_Ystar+qy))
            for count in counts:
               if count[0].startswith("X=*") and count[0].endswith(label):
                    C_YyWstar = count[1]                        # Yy W*
            for word in words:
                # word[0] is useless
                wordCounts = word[1][1]
                for wordCount in wordCounts:
                    if wordCount[0].endswith(label):
                        C_YyWw = wordCount[1]                    # Yy Ww
                        prob = prob + math.log((1.0*(1+ C_YyWw))/(C_YyWstar+qx))
            if prob > maxP:
                maxP = prob
                maxL = label
    return (id,maxL,maxP)

#always subclass Planner
class NB(Planner):
    # params is a dictionary of params given on the command line.
	# e.g. trainFile = params['trainFile']
    params = GPig.getArgvParams()
    #trainFile = params['trainFile']
    #testFile = params['testFile']
    trainFile = 'train'
    testFile = 'test'

    # 1. training, obtain wordcounts, assuming correct (eventCount => ('x,y',7)...)
    # 2. transform to key value pairs (ec_tran => (key,[(x=y=,1),()()]),(key,[]),...) BY piping
    eventCount = ReadLines(trainFile) | Flatten(by=tokens) | Group(by=lambda x:x, reducingTo=ReduceToCount()) \
               |Group(by= lambda (exp,count):exp.split("^")[0].split("=")[1]) # key is w from X=w

    # ** count the number of instances required for prediction, use counts[1]
    counts = Filter(eventCount,by=lambda (key,list):key.endswith("*"))
    #ec_tran = Group(eventCount, by= lambda (exp,count):exp.split("^")[0].split("=")[1])

    # 3. flatten the test documents = get request  (request => (key, id)
    # 4. join the request with outcome of #2. (joined => (key, counts, IDs) BY piping
    request = ReadLines(testFile) | Flatten(by=tokens2) | JoinTo(Jin(eventCount,by=lambda (word,id):word),by=lambda (word,id):word)

    #joined = Join(Jin(eventCount,by= lambda tuple:tuple[0]), Jin(request,by=lambda (word,id):word)) \

    # 5. group the result to classify
    output = Group(request, by = lambda (id,counts):id[1],reducingTo=ReduceToList())
    output = Augment(output,sideview=counts, loadedBy= lambda v:GPig.onlyRowOf(v)) | ReplaceEach(by=prediction)
    #final = Group(joined, by = lambda (counts,id):id[1],reducingTo=ReduceToSum())


    #prob = ReplaceEach(final, by=lambda ((word,count),n): (word,count,n,float(count)/n))

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here

