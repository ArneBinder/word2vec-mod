# coding: utf-8

# source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb

# Deep Learning
# =============
#
# Assignment 5
# ------------
#
# The goal of this assignment is to train a Word2Vec skip-gram model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#get_ipython().magic(u'matplotlib inline')
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import json
import csv
from sys import exit
from spacy.en import English
import sys
import time
import datetime
from functools import wraps
import operator

csv.field_size_limit(sys.maxsize)

data_dir = '/media/arne/DATA/DEVELOPING/ML/data/'
article_count = 10000


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

# Download the data from the source website if necessary.

# In[4]:

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


#filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.

# In[5]:

@fn_timer
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

# initialize attributes to collect values for
ling_stats = {key: dict() for key in ['pos_', 'shape_', 'dep_', 'tag_', 'ent_type_', 'ent_iob_']}

#ling_stats['pos_'] = dict()
# ling_stats['asda'] = dict()
pos_blacklist = [u'SPACE', u'PUNCT', u'NUM']
def process_token(token, plain_tokens=list()):

    ## collect linguistic stats
    #for attr, types in ling_stats.items():
    #    if not hasattr(token, attr):
    #        print("ERROR. Attribute not found:", attr, 'Skip it.')
    #        continue
    #    value = getattr(token, attr)
    #    types[value] = types.get(value, [])
    #    #if(token.string not in types[value]):
    #        #types[value] = (types[value] + [token.string])[-5:]
    #    types[value].append(token.string)

    # remove not-a-word tokens
    #if token.pos_ in pos_blacklist:
    #    plain_tokens.append(token.pos_)
    #    return False

    #if token.ent_iob_ == 'B':
    #    plain_tokens.append(token.ent_type_)
    #    return True

    #if token.ent_iob_ == 'I':
    #    return False

    # remove entities
    #if(token.ent_type != 0):
    #    return False
    plain_tokens.append(token.lemma_)
    return True


# to read data presented here: https://blog.lateral.io/2015/06/the-unknown-perils-of-mining-wikipedia/
@fn_timer
def read_data_csv(filename, max_rows=100):
    print('parse', max_rows, 'articles')
    parser = English()
    plain_tokens = list()
    with open(filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['article-id', 'content'])
        count = 0
        for row in reader:
            if(count >= max_rows):
                break
            if((count * 100) % max_rows == 0):
                #sys.stdout.write("progress: %d%%   \r" % (count * 100 / max_rows))
                #sys.stdout.flush()
                print('parse article:', row['article-id'], '... ', count * 100 / max_rows, '%')
            content = row['content'].decode('utf-8')

            parsedData = parser(content)

            for i, token in enumerate(parsedData):
                process_token(token, plain_tokens)

            count +=1
    return plain_tokens

def read_data_example():
    #text = u' Research Design and Standards Organization  The Research Design and Standards Organisation (RDSO) is an ISO 9001 research and development organisation under the Ministry of Railways of India, which functions as a technical adviser and consultant to the Railway Board, the Zonal Railways, the Railway Production Units, RITES and IRCON International in respect of design and standardisation of railway equipment and problems related to railway construction, operation and maintenance. History. To enforce standardisation and co-ordination between various railway systems in British India, the Indian Railway Conference Association (IRCA) was set up in 1903. It was followed by the establishment of the Central Standards Office (CSO) in 1930, for preparation of designs, standards and specifications. However, till independence in 1947, most of the designs and manufacture of railway equipments was entrusted to foreign consultants. After independence, a new organisation called Railway Testing and Research Centre (RTRC) was set up in 1952 at Lucknow, for undertaking intensive investigation of railway problems, providing basic criteria and new concepts for design purposes, for testing prototypes and generally assisting in finding solutions for specific problems. In 1957, the Central Standards Office (CSO) and the Railway Testing and Research Centre (RTRC) were integrated into a single unit named Research Designs and Standards Organisation (RDSO) under the Ministry of Railways with its headquarters at Manaknagar, Lucknow. The status of RDSO was changed from an ""Attached Office"" to a ""Zonal Railway"" on April 1, 2003, to give it greater flexibility and a boost to the research and development activities. Organisation. The RDSO is headed by a Director General who ranks with a General Manager of a Zonal Railway. The Director General is assisted by an Additional Director General and 23 Sr. Executive Directors and Executive Directors, who are in charge of the 27 directorates: Bridges and Structures, the Centre for Advanced Maintenance Techlogy (AMTECH), Carriage, Geotechnical Engineering, Testing, Track Design, Medical, EMU & Power Supply, Engine Development, Finance & Accounts, Telecommunication, Quality Assurance, Personnel, Works, Psycho-Technical, Research, Signal, Wagon Design, Electric Locomotive, Stores, Track Machines & Monitoring, Traction Installation, Energy Management, Traffic, Metallurgical & Chemical, Motive Power and Library & Publications. All the directorates except Defence Research are located in Lucknow. Projects. Development of a new crashworthy design of  4500 HP WDG4 locomotive incorporating new  technology to improve dynamic braking and attain  significant fuel savings. Development of Drivers’ Vigilance Telemetric Control  System which directly measures and analyses variations  in biometric parameters to determine the state of alertness  of the driver. Development of Train Collision Avoidance System(TCAS). Development of Computer Aided Drivers Aptitude test  equipment for screening high speed train drivers for  Rajdhani/Shatabdi Express trains to evaluate their reaction  time, form perception, vigilance and speed anticipation. Assessment of residual fatigue life of critical railway  components like rail, rail weld, wheels, cylinder head, OHE  mast, catenary wire, contact wire, wagon components, low  components, etc. to formulate remedial actions. Modification of specification of Electric Lifting Barrier to improve its strength and reliability Design and development of modern fault tolerant, fail-safe, maintainer friendly Electronic Interlocking system Development of 4500 HP Hotel Load Locomotive to provide clean and noise free power supply to coaches from locomotive to eliminate the existing generator car of Garib Rath express trains. Field trials conducted for electric locomotive hauling Rajdhani/Shatabdi express trains with Head On Generation (HOG) system to provide clean and noise free power supply to end on coaches. Development of WiMAX technology to provide internet access to the passengers in running trains. Design and Development of Ballastless Track with indigenous fastening system (BLT-IFS). Major Achievements. Development of Pre-stressed concrete sleeper and allied components along with Source Development.  '
    text = u' The Research Design and Standards Organisation (RDSO) is an ISO 9001 research and development organisation under the Ministry of Railways of India, which functions as a technical adviser and consultant to the Railway Board, the Zonal Railways, the Railway Production Units, RITES and IRCON International in respect of design and standardisation of railway equipment and problems related to railway construction, operation and maintenance. History. To enforce standardisation and co-ordination between various railway systems in British India, the Indian Railway Conference Association (IRCA) was set up in 1903. It was followed by the establishment of the Central Standards Office (CSO) in 1930, for preparation of designs, standards and specifications. However, till independence in 1947, most of the designs and manufacture of railway equipments was entrusted to foreign consultants. After independence, a new organisation called Railway Testing and Research Centre (RTRC) was set up in 1952 at Lucknow, for undertaking intensive investigation of railway problems, providing basic criteria and new concepts for design purposes, for testing prototypes and generally assisting in finding solutions for specific problems. In 1957, the Central Standards Office (CSO) and the Railway Testing and Research Centre (RTRC) were integrated into a single unit named Research Designs and Standards Organisation (RDSO) under the Ministry of Railways with its headquarters at Manaknagar, Lucknow. The status of RDSO was changed from an ""Attached Office"" to a ""Zonal Railway"" on April 1, 2003, to give it greater flexibility and a boost to the research and development activities. Organisation. The RDSO is headed by a Director General who ranks with a General Manager of a Zonal Railway. The Director General is assisted by an Additional Director General and 23 Sr. Executive Directors and Executive Directors, who are in charge of the 27 directorates: Bridges and Structures, the Centre for Advanced Maintenance Techlogy (AMTECH), Carriage, Geotechnical Engineering, Testing, Track Design, Medical, EMU & Power Supply, Engine Development, Finance & Accounts, Telecommunication, Quality Assurance, Personnel, Works, Psycho-Technical, Research, Signal, Wagon Design, Electric Locomotive, Stores, Track Machines & Monitoring, Traction Installation, Energy Management, Traffic, Metallurgical & Chemical, Motive Power and Library & Publications. All the directorates except Defence Research are located in Lucknow. Projects. Development of a new crashworthy design of  4500 HP WDG4 locomotive incorporating new  technology to improve dynamic braking and attain  significant fuel savings. Development of Drivers’ Vigilance Telemetric Control  System which directly measures and analyses variations  in biometric parameters to determine the state of alertness  of the driver. Development of Train Collision Avoidance System(TCAS). Development of Computer Aided Drivers Aptitude test  equipment for screening high speed train drivers for  Rajdhani/Shatabdi Express trains to evaluate their reaction  time, form perception, vigilance and speed anticipation. Assessment of residual fatigue life of critical railway  components like rail, rail weld, wheels, cylinder head, OHE  mast, catenary wire, contact wire, wagon components, low  components, etc. to formulate remedial actions. Modification of specification of Electric Lifting Barrier to improve its strength and reliability Design and development of modern fault tolerant, fail-safe, maintainer friendly Electronic Interlocking system Development of 4500 HP Hotel Load Locomotive to provide clean and noise free power supply to coaches from locomotive to eliminate the existing generator car of Garib Rath express trains. Field trials conducted for electric locomotive hauling Rajdhani/Shatabdi express trains with Head On Generation (HOG) system to provide clean and noise free power supply to end on coaches. Development of WiMAX technology to provide internet access to the passengers in running trains. Design and Development of Ballastless Track with indigenous fastening system (BLT-IFS). Major Achievements. Development of Pre-stressed concrete sleeper and allied components along with Source Development.  '
    parser = English()
    plain_tokens = list()
    parsedData = parser(text)
    for i, token in enumerate(parsedData):
        process_token(token, plain_tokens)
    return plain_tokens

#print('token count:', len(words_mod))


#words = read_data(maybe_download('text8.zip', 31344016))
words = read_data_csv(data_dir+'corpora/documents_utf8_filtered_20pageviews.csv', article_count)
#words = read_data_example()

word_count = len(words)
print('data preprocessing finished')
print('Data size %d' % len(words))

ling_stats_counted = list()
for type, values in ling_stats.items():
    #ling_stats_counted[type] = dict()
    for value, samples in values.items():
        #ling_stats_counted[type][value] = dict()
        d = dict()
        for sample in samples:
            count = d.get(sample, 0)
            d[sample] = count + 1

        #sorted_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        l = [item for sublist in sorted(d.items(), key=operator.itemgetter(1), reverse=True) for item in sublist][:20]
        ling_stats_counted.append({'categorie': type, 'value': value, 'types': len(d), 'tokens': len(samples), 'l': l, 'q': len(d)/float(len(samples))})


#with open('ling_stats.txt', 'w') as outfile:
#    json.dump(ling_stats_counted, outfile, indent=4, separators=(',', ': '))


#exit()

# Build the dictionary and replace rare words with UNK token.

# In[5]:


@fn_timer
def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    #i = 1
    for word, c in count:
        # take only tokens which occure >1 time
        #if c > 1:
        dictionary[word] = len(dictionary)
        #    i+=1
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# initila vocabulary size
vocabulary_size = 500000
init_voc_size = vocabulary_size
data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)

# reset vocabulary size if it was not reached
if(vocabulary_size > len(reverse_dictionary)):
    print('reset vocabulary_size=', vocabulary_size, ' to ', len(reverse_dictionary))
    vocabulary_size = len(reverse_dictionary)

print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)

with open('reverse_dictionary.txt', 'w') as outfile:
    json.dump(reverse_dictionary, outfile)

with open('count.txt', 'w') as outfile:
    json.dump(count, outfile)
# Function to generate a training batch for the skip-gram model.

# In[6]:

data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

# Train a skip-gram model.

# In[7]:

logdir = data_dir+'summaries/train_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')
    softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)), name='weights')
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]), name='biases')

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset, name='embed')
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                                     train_labels, num_sampled, vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True), name='norm')
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset, name='valid_embeddings')
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings), name='similarity')


    #summary for tensorboard
    tf.summary.scalar('loss', loss)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir, graph, flush_secs=10)
    #test_writer = tf.test.SummaryWriter('summaries/test')

    saver = tf.train.Saver()

num_steps = 1000000
# In[9]:
@fn_timer
def train():
    #num_steps = 100001
    #num_steps = 1000000
    interval_avg = 50   # average the loss every num_steps/interval_avg steps
    interval_sav = 10   # save the model every num_steps/interval_sav steps

    with tf.Session(graph=graph) as session:
        #tf.initialize_all_variables().run() # for older versions of Tensorflow
        tf.global_variables_initializer().run()
        print('Initialized')
        average_loss = 0
        for step in range(1, num_steps+1):
            batch_data, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, summary, l = session.run([optimizer, merged, loss], feed_dict=feed_dict)
            average_loss += l
            if ((step * interval_avg) % num_steps) == 0 or step == 1:
            #if step % 2000 == 0:
                if step > 1:
                    average_loss = average_loss * interval_avg / num_steps
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
                train_writer.add_summary(summary, step)

            # note that this is expensive (~20% slowdown if computed every 500 steps)
            # do it after ever 10% of max_steps
            if ((step * interval_sav) % num_steps) == 0 or step == 1:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

                save_path = saver.save(session, logdir+"/model.ckpt", step)
                print("Model saved in file: %s" % save_path)

        return normalized_embeddings.eval()


final_embeddings = train()

@fn_timer
def embeddings_to_tsv(embeddings, path):
    i = 0
    with open(path,'w') as f:
        size, dims = embeddings.shape
        f.write('{} {}'.format(size, dims) + '\n')
        for vec in embeddings:
            f.write(reverse_dictionary[i].encode('utf8') + '\t' + '\t'.join(str(x) for x in vec) + '\n')
            i += 1
        print('vec count:', i)
        print('label:', 'mod_a'+str(article_count)+'-w'+str(word_count)+'-v'+str(init_voc_size)+'-'+str(vocabulary_size)+'-dim'+str(embedding_size)+'-s'+str(i))
    return i

vec_count = embeddings_to_tsv(final_embeddings, logdir+'/embeddings.tsv')

with open('embeddings_meta', 'w') as f:
    json.dump({'filename':logdir+'/embeddings.tsv', 'label': 'mod_a'+str(article_count)+'-w'+str(word_count)+'-v'+str(init_voc_size)+'-'+str(vocabulary_size)+'-dim'+str(embedding_size)+'-s'+str(num_steps)},f)

#with open('model.txt', 'w') as outfile:
#    json.dump(final_embeddings, outfile)
# In[10]:

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points + 1, :])


# In[11]:

def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points + 1)]
plot(two_d_embeddings, words)


# ---
#
# Problem
# -------
#
# An alternative to skip-gram is another Word2Vec model called [CBOW](http://arxiv.org/abs/1301.3781) (Continuous Bag of Words). In the CBOW model, instead of predicting a context word from a word vector, you predict a word from the sum of all the word vectors in its context. Implement and evaluate a CBOW model trained on the text8 dataset.
#
# ---

import csv, codecs, cStringIO

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self