#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/dev-test-train.de-en", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
t_ef = defaultdict(int)
ef_count = defaultdict(int)
count_ef = defaultdict(int)
total_f = defaultdict(int)
total_s = defaultdict(int)
t_ef = defaultdict(int)
all_ef = defaultdict(int)
sys.stderr.write("Training IBM Model 1")
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
        t_ef[e_j,f_i] = 1.0
        all_ef[e_j,f_i]=1 
  for e_j in set(e):
    e_count[e_j] += 1

for n in range(1,5):
    sys.stderr.write("\n Starting Iteration = %i " % (n))
    for ef in all_ef:
        count_ef[ef] = 0.0
    for f in f_count:
        total_f[f] = 0.0
    # E Step
    for (s, (f_s, e_s)) in enumerate(bitext):
        for e in set(e_s):
            total_s[e] = 0.0    
            for f in set(f_s): 
                total_s[e] = total_s[e] + t_ef[e,f]
        for e in set(e_s):        
            for f in set(f_s):
                count_ef[e,f] = count_ef[e,f] + ((t_ef[e,f]) / total_s[e])
                total_f[f] = total_f[f] + ((t_ef[e,f])  / total_s[e])
    # M Step
    for ef in all_ef:
         t_ef[ef] = count_ef[ef] / total_f[ef[1]]
         
sys.stderr.write("\n Starting Writing Alignments to File")
for (f, e) in bitext:
  for (i, f_i) in enumerate(f):
    max_prob=0
    max_j=0 
    for (j, e_j) in enumerate(e):
        if t_ef[e_j,f_i] > max_prob:
            max_prob = t_ef[e_j,f_i]
            max_j=j
    sys.stdout.write("%i-%i " % (i,max_j))
  sys.stdout.write("\n")

