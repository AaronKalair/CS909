## CSV File needs to be
# Topic if topic in earn, acquisitions, money-fx, grain, crude, trade, interest, ship, wheat, corn
# Train / Test ?
# From text
#   Title
#   Dateline
#   Body

# For XML Parsing and opening the files
from lxml import etree
import os
import codecs
from io import StringIO, BytesIO
import csv
import re
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

# Get all of the XML files in the directory
files = [f for f in os.listdir('.') if ".xml" in f]
# Headers for the top row of the CSV file
headers = ["topic", "trainortest", "title", "datleline", "body"]
counts = {}

# Stop Words for removing when generating the word clouds
stopwords = """a,able,about,across,after,all,almost,also,am
        ,among,an,and,any,are,as,at,be,because
        ,been,but,by,can,cannot,could,dear
        ,did,do,does,either,else,ever,every
        ,for,from,get,got,had,has,have,he,her
        ,hers,him,his,how,however,i,if,in,into
        ,is,it,its,just,least,let,like,likely,
        may,me,might,most,must,my,neither,no,nor,
        not,of,off,often,on,only,or,other,our,own,
        rather,said,say,says,she,should,since,so,
        some,than,that,the,their,them,then,there,
        these,they,this,tis,to,too,twas,us,wants,
        was,we,were,what,when,where,which,while,
        who,whom,why,will,with,would,yet,you,your""".split(",")
stopwords = [i.strip() for i in stopwords]

# Keep track of how many of each class label we have
counts = {'earn':0, "acq":0, "money-fx":0, "grain":0, "crude":0, "trade":0, "interest":0, "ship":0, "wheat":0, "corn":0 }

# Determines if the element under the heading contains data and if so normalises it by removing special characters, converting it to ASCII and lower casing it
def determineIfAttributeExists(attr):
    # If the length is 0 this attribute doesnt exist
    if (len(attr )== 0 ):
        return ""
    else:
        # If it does exist normalise it
        ret = attr[0]
        ret =  ret.text.replace("\n", "").lower()
        ret =  ''.join([i if ord(i) < 128 else ' ' for i in ret])
        ret = re.sub('[^a-zA-Z ]', '', ret)
        # return ' '.join(ret.split())
        # uncomment to stem
        return st.stem(ret)

# Determines if the class label is one of the ones we're using
def isTopicWanted(attr):
    wanted = False
    topics = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
    ret = ""
    for element in attr:
        for d in element:
            if (d.text in topics):
                wanted = True
                ret =  d.text
                counts[ret] = counts[ret] +1
    return (wanted, ret)

# Returns if this is a train or test instance
def getTrainOrTest(attr):
    return attr.attrib['LEWISSPLIT']

# Writes a 2D array to a CSV file
def write(rows):
    with open("data.csv", 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

# Counts the number of occurences of each word in the text, used for building word clouds
def computeBodyWordsCounts(body):
    for word in body.split():
        word = re.sub('[^a-zA-Z ]', '', word)
        if word in stopwords:
            continue
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

# Set some variables we need for later
rows = []
acq = 0
earn = 0

# For each XML file parse it
for f in files:
    # Initially set every column in the row to 0
    row = [0 for _ in xrange(5)]
    # Open the XML file
    with open (f, "r") as myfile:
        data=myfile.read()
        # Decode the strings as they contain special characters
        data = data.decode('iso-8859-1')
        # Build a parsable tree from the data
        tree = etree.fromstring(data)
        # Each news article is under a REUTERS element
        for reut in tree.xpath("REUTERS"):
            # Get each of the required elements
            row = [0 for _ in xrange(5)]
            title = reut.xpath("TEXT/TITLE")
            dateline = reut.xpath("TEXT/DATELINE")
            body = reut.xpath("TEXT/BODY")
            topics = reut.xpath("TOPICS")
            train_test = getTrainOrTest(reut)
            # Alter this to determine if Test or Training data is output
            # if train_test == "TEST":
            #     continue
            # Determine if this topic is one of the top 10 we want
            topic_wanted = isTopicWanted(topics)
            # If this is a topic we want
            if( topic_wanted[0] ):
                # If we want this topic get its name
                topic = topic_wanted[1]
                # ACQ and Earn are being undersampled so keep track of how many we have for each label and stop when we get enough
                if topic == "acq":
                    acq = acq+1
                    if acq > 500:
                        continue
                if topic == "earn":
                    earn = earn +1
                    if earn > 500:
                        continue
            else:
                continue
            # Ignore any of the articles not part of the train test set
            if(train_test == "NOT-USED"):
                continue

            # Set each of the elements we want
            row[0] = topic.decode('iso-8859-1')
            row[1] = train_test.decode('iso-8859-1')
            row[2] = determineIfAttributeExists(title).decode('iso-8859-1')
            row[3] = determineIfAttributeExists(dateline).decode('iso-8859-1')
            row[4] = determineIfAttributeExists(body).decode('iso-8859-1')
            # If were building the word clouds count the number of eacg word in the body
            # computeBodyWordsCounts(row[4])
            rows.append(row)

# Now write out all of the rows
write(rows)
# with open('mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
#     w = csv.DictWriter(f, counts.keys())
#     w.writeheader()
#     w.writerow(counts)

