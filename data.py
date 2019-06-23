# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval.sh/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)"""

  def __init__(self, vocab_file, max_size):
    """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
      max_size: integer. The maximum size of the resulting Vocabulary."""
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print 'Warning: incorrectly formatted line in vocabulary file: %s\n' % line
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count)
          break

    print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

  def word2id(self, word):
    """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer)."""
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    """Returns the total size of the vocabulary"""
    return self._count

  def write_metadata(self, fpath):
    """Writes metadata file for Tensorboard word embedding visualizer as described here:
      https://www.tensorflow.org/get_started/embedding_viz

    Args:
      fpath: place to write the metadata file
    """
    print "Writing word embedding metadata file to %s..." % (fpath)
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
  """Generates tf.Examples from data files.

    Binary data format: <length><blob>. <length> represents the byte size
    of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
    the tokenized article text and summary.

  Args:
    data_path:
      Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
    single_pass:
      Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

  Yields:
    Deserialized tf.Example.
  """
  while True:
    filelist = glob.glob(data_path) # get the list of datafiles
    # assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
    else:
      random.shuffle(filelist)
    for f in filelist:
      reader = open(f, 'rb')
      while True:
        # len_bytes = reader.read(8)
        # print("len_bytes", len_bytes)
        # if not len_bytes: break # finished reading this file
        # str_len = struct.unpack('q', len_bytes)[0]
        # print("str_len", str_len)
        # example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        # print("example_str", example_str)
        example_str = "\n\xcf<\n\xf0\x02\n\x08abstract\x12\xe3\x02\n\xe0\x02\n\xdd\x02<s> marseille prosecutor says `` so far no videos were used in the crash investigation '' despite media reports . </s> <s> journalists at bild and paris match are `` very confident '' the video clip is real , an editor says . </s> <s> andreas lubitz had informed his lufthansa training school of an episode of severe depression , airline says . </s>\n\xd99\n\x07article\x12\xcd9\n\xca9\n\xc79marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . '' robin 's comments follow claims by two magazines , german daily bild and french paris match , of a cell phone video showing the harrowing final seconds from on board germanwings flight 9525 as it crashed into the french alps . all 150 on board were killed . paris match and bild reported that the video was recovered from a phone at the wreckage site . the two publications described the supposed video , but did not post it on their websites . the publications said that they watched the video , which was found by a source close to the investigation . `` one can hear cries of ` my god ' in several languages , '' paris match reported . `` metallic banging can also be heard more than three times , perhaps of the pilot trying to open the cockpit door with a heavy object . towards the end , after a heavy shake , stronger than the others , the screaming intensifies . then nothing . '' `` it is a very disturbing scene , '' said julian reichelt , editor-in-chief of bild online . an official with france 's accident investigation agency , the bea , said the agency is not aware of any such video . lt. col. jean-marc menichini , a french gendarmerie spokesman in charge of communications on rescue efforts around the germanwings crash site , told cnn that the reports were `` completely wrong '' and `` unwarranted . '' cell phones have been collected at the site , he said , but that they `` had n't been exploited yet . '' menichini said he believed the cell phones would need to be sent to the criminal research institute in rosny sous-bois , near paris , in order to be analyzed by specialized technicians working hand-in-hand with investigators . but none of the cell phones found so far have been sent to the institute , menichini said . asked whether staff involved in the search could have leaked a memory card to the media , menichini answered with a categorical `` no . '' reichelt told `` erin burnett : outfront '' that he had watched the video and stood by the report , saying bild and paris match are `` very confident '' that the clip is real . he noted that investigators only revealed they 'd recovered cell phones from the crash site after bild and paris match published their reports . `` that is something we did not know before . ... overall we can say many things of the investigation were n't revealed by the investigation at the beginning , '' he said . what was mental state of germanwings co-pilot ? german airline lufthansa confirmed tuesday that co-pilot andreas lubitz had battled depression years before he took the controls of germanwings flight 9525 , which he 's accused of deliberately crashing last week in the french alps . lubitz told his lufthansa flight training school in 2009 that he had a `` previous episode of severe depression , '' the airline said tuesday . email correspondence between lubitz and the school discovered in an internal investigation , lufthansa said , included medical documents he submitted in connection with resuming his flight training . the announcement indicates that lufthansa , the parent company of germanwings , knew of lubitz 's battle with depression , allowed him to continue training and ultimately put him in the cockpit . lufthansa , whose ceo carsten spohr previously said lubitz was 100 % fit to fly , described its statement tuesday as a `` swift and seamless clarification '' and said it was sharing the information and documents -- including training and medical records -- with public prosecutors . spohr traveled to the crash site wednesday , where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside . he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . menichini told cnn late tuesday that no visible human remains were left at the site but recovery teams would keep searching . french president francois hollande , speaking tuesday , said that it should be possible to identify all the victims using dna analysis by the end of the week , sooner than authorities had previously suggested . in the meantime , the recovery of the victims ' personal belongings will start wednesday , menichini said . among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board . check out the latest from our correspondents . the details about lubitz 's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and lubitz 's possible motive for downing the jet . a lufthansa spokesperson told cnn on tuesday that lubitz had a valid medical certificate , had passed all his examinations and `` held all the licenses required . '' earlier , a spokesman for the prosecutor 's office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot 's license . kumpa emphasized there 's no evidence suggesting lubitz was suicidal or acting aggressively before the crash . investigators are looking into whether lubitz feared his medical condition would cause him to lose his pilot 's license , a european government official briefed on the investigation told cnn on tuesday . while flying was `` a big part of his life , '' the source said , it 's only one theory being considered . another source , a law enforcement official briefed on the investigation , also told cnn that authorities believe the primary motive for lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems . lubitz 's girlfriend told investigators he had seen an eye doctor and a neuropsychologist , both of whom deemed him unfit to work recently and concluded he had psychological issues , the european government official said . but no matter what details emerge about his previous mental health struggles , there 's more to the story , said brian russell , a forensic psychologist . `` psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they were n't going to keep doing their job and they 're upset about that and so they 're suicidal , '' he said . `` but there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person 's problems . '' germanwings crash compensation : what we know . who was the captain of germanwings flight 9525 ? cnn 's margot haddad reported from marseille and pamela brown from dusseldorf , while laura smith-spark wrote from london . cnn 's frederik pleitgen , pamela boykoff , antonia mortensen , sandrine amiel and anna-maja rappard contributed to this report ."
        yield example_pb2.Example.FromString(example_str)

        # Added by zachary #
        """To break right away to make it irrelevant to file input"""
        break
        ###################

    if single_pass:
      print "example_generator completed reading all datafiles. No more data."
      break


def article2ids(article_words, vocab):
  """Map the article words to their ids. Also return a list of OOVs in the article.

  Args:
    article_words: list of words (strings)
    vocab: Vocabulary object

  Returns:
    ids:
      A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
    oovs:
      A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
  """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

  Args:
    abstract_words: list of words (strings)
    vocab: Vocabulary object
    article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

  Returns:
    ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

  Args:
    id_list: list of ids (integers)
    vocab: Vocabulary object
    article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

  Returns:
    words: list of words (strings)
  """
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words


def abstract2sents(abstract):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(article, vocab):
  """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
  """Returns the abstract string, highlighting the article OOVs with __underscores__.

  If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

  Args:
    abstract: string
    vocab: Vocabulary object
    article_oovs: list of words (strings), or None (in baseline mode)
  """
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if article_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str
