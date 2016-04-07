""" Main class for phonological lazy-learner.
LIbPhon.py

Initializes learner or teacher, defines produce(), categorize(),
train(), test() and auxiliary methods.
"""
import copy
import cPickle as pickle
import numpy as np
from random import choice
__author__ = "Fred Mailhot (fred.mailhot@gmail.com)"


class LIbPhon(object):

    def __init__(self, teacher=False, lexicon=None, knn=5, coart=0.0):
        """ In:
                teacher     -- <bool> is this is the training or learning agent
                syllable    -- <str> "V" or "CV" indicating basic syllable type
                instances   -- <int> # of instances per word (base & inflected)
        """
        self.knn = knn
        if coart != 0.0:
            self.coart = coart
            self.coarticulation = True
        else:
            self.coarticulation = False

        if not teacher:
            if lexicon:
                fin = file(lexicon)
                self.lexicon = pickle.load(fin)
                fin.close()
            else:
                self.lexicon = {}
        else:
            # see lex_generator.py for info on lexicon.pck
            with open(lexicon, "rb") as f_in:
                self.lexicon = pickle.load(f_in)

    def quick_select(self, data, k):
        """ Find the kth rank ordered element (least value has rank 0).
                Adapted from: http://code.activestate.com/recipes/269554/
            N.B. to find kth largest item in data, call with k = len(data)-k
        """
        if data.shape[0] < k:
            return np.max(data)
        while True:
            pivot = data[np.random.randint(0, data.shape[0])]
            under = data[np.where(data < pivot)]
            over = data[np.where(data > pivot)]
            pcount = np.where(data == pivot)[0].shape[0]
            if k < under.shape[0]:
                data = under
            elif k < under.shape[0] + pcount:
                return pivot
            else:
                data = over
                k -= under.shape[0] + pcount

    ## This essentially does k-NN regression
    #TODO(fmailhot): say more here...?
    #TODO(fmailhot): this is too long...split it up!
    def produce(self, label):
        """ In:
                label       -- space-separated string with one lexical label
                                    and one or more infl labels.
            Out:
                output      -- output exemplar...(n,2) ndarray
        """
        (root, sep, suffixes) = label.partition(" ")
        all_keys_list = self.lexicon.keys()   # let's not call this too often
        all_keys_join = " ".join(all_keys_list)
        if label in all_keys_list:
            # query known; seed & cloud from lexical entry
            cloud = self.lexicon[label]
            seed = np.random.permutation(cloud)[0]
        elif root not in all_keys_join:
            # query root unknown; seed & cloud from RANDOM lexical entry
            lbl = choice(all_keys_list)
            seed = self.lexicon[lbl][np.random.randint(self.lexicon[lbl].shape[0])]
            cloud = self.lexicon[lbl]
        else:
            # query root known
            # (following list has len()<=3)
            candidates = [x for x in all_keys_list if
                          x.split()[0] in all_keys_join]
            if "PL" not in suffixes:
                # singular query; check for known singular LE
                # (i.e. opposite CASE)
                tmp_lbl = " ".join([root, "NOM" if suffixes == "ACC" else "ACC"])
                if tmp_lbl in candidates:
                    # found singular; seed from it, cloud from suffix(es)
                    seed = np.random.permutation(self.lexicon[tmp_lbl])[0]
                    try:
                        cloud = np.vstack([self.lexicon[x] for
                                           x in all_keys_list if suffixes in x])
                    except ValueError:
                        cloud = self.lexicon[tmp_lbl]
                else:
                    # no singular form known, ergo LE exists as plural;
                    # seed from it
                    tmp_pl_lbl = " ".join([root, "PL", suffixes])
                    if tmp_pl_lbl in candidates:
                        # same case
                        seed = np.random.permutation(self.lexicon[tmp_pl_lbl])[0]
                    else:
                        # opposite case
                        tmp_pl_lbl = tmp_pl_lbl.replace(suffixes, "NOM" if suffixes == "ACC" else "ACC")
                        seed = np.random.permutation(self.lexicon[tmp_pl_lbl])[0]
                    # cloud is all non-plurals with same case
                    try:
                        cloud = np.vstack([self.lexicon[x] for x in all_keys_list
                                           if ("PL" not in x and suffixes in x)])
                    except ValueError:
                        cloud = self.lexicon[tmp_pl_lbl]
            else:
                # plural query;
                suffixes = suffixes.split()
                tmp_lbl = label.replace(suffixes[1], "ACC" if "NOM" in label else "NOM")
                if tmp_lbl in all_keys_list:
                    # plural LE (opposite case) found;
                    # seed from it, same-case LEs as cloud
                    seed = np.random.permutation(self.lexicon[tmp_lbl])[0]
                    try:
                        cloud = np.vstack([self.lexicon[x] for x in
                                           all_keys_list if suffixes[1] in x])
                    except ValueError:
                        cloud = self.lexicon[tmp_lbl]
                else:
                    # singular LE exists
                    tmp_sg_lbl = " ".join([root, suffixes[1]])
                    if tmp_sg_lbl in all_keys_list:
                        # singular form (same case) known;
                        # seed from it, use all plurals as cloud
                        seed = np.random.permutation(self.lexicon[tmp_sg_lbl])[0]
                        try:
                            cloud = np.vstack([self.lexicon[x] for x in
                                               all_keys_list if "PL" in x])
                        except ValueError:
                            cloud = self.lexicon[tmp_sg_lbl]
                    else:
                        # singular form (opposite case) known
                        tmp_sg_lbl = tmp_sg_lbl.replace(suffixes[1], "ACC" if "NOM" in label else "NOM")
                        seed = np.random.permutation(self.lexicon[tmp_sg_lbl])[0]
                        try:
                            cloud = np.vstack([self.lexicon[x] for x in all_keys_list
                                               if " ".join(suffixes) in x])
                        except:
                            cloud = self.lexicon[tmp_sg_lbl]

        ## Need to noisify the ouputs here...isotropic...
        dists = np.sqrt(np.sum(np.sum((cloud - seed)**2, axis=1), axis=1))
        nn_idx = np.where(dists <= self.quick_select(dists, self.knn))
        nn = cloud[nn_idx]
        wts = dists[nn_idx]
        if np.sum(wts) == 0:
            output = seed
        else:
            # distance weighted average (this is basically knn-regression)
            output = np.average(nn, axis=0, weights=wts).astype("int32")

        if (self.coarticulation):
            return self.coarticulate(output)
        else:
            return output

    def coarticulate(self, trajectory):
        """ Shift values of formants according to spec'd degree of coarticulation.

            This is brain-dead anticipatory, I just blindly move all formants a fixed
            proportion of the distance to the next formant.
        """
        # each agent has its own articulatory variability
        #self.alpha = N.random.normal(15,2)
        #self.beta = N.random.normal(2,0.25)
        #if self.beta < 1.0:
        #    self.beta = 1.1
        trajectory_out = copy.copy(trajectory)
        # next_vowel = None
        # cur_vowel = None
        for i in range(1, trajectory_out.shape[0] - 1, 2):    # want the vowels, only
            trajectory_out[i] -= self.coart * (trajectory_out[i] - trajectory_out[i + 1])

        return trajectory_out

    def categorize(self, input):
        """
            Store labelled inputs as-is, tagged with 'meaning' label.
            Label-free input is classified according to nearest neighbour category
        """
        label, exemplar = input
        if label is not None:
            if label in self.lexicon:
                self.lexicon[label] = np.vstack((np.expand_dims(exemplar, axis=0), self.lexicon[label]))
            else:
                self.lexicon[label] = np.expand_dims(exemplar, axis=0)
        else:
            #TODO: make this k-NN categorization ?
            candidates = sorted([(np.mean(np.abs(np.mean(self.lexicon[x], axis=0).astype("int32") - exemplar), axis=0)[1], x)
                                 for x in self.lexicon.keys()])
            self.lexicon[candidates[0][1]] = exemplar


if __name__ == "__main__":
    teach = LIbPhon(teacher=True)
    teach.produce(teach.lexicon.keys()[0])
