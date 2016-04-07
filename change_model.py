#!/usr/bin/env python
###########################################################################
# change_model.py
# by: Fred Mailhot
###########################################################################
""" Module defining basic class for "world" and handling params for population-level
sims (# of agents, intra/inter-generational connectivity scheme, etc.)
"""
import sys
import copy
import numpy as N
from scipy import vectorize
from scipy.cluster import vq


class Agent(object):
    """
    Main class of simulation. Defines both speaker and learner agents.
    """
    vf = None
    letters = None
    vowel_map = None
    vowel_spread = None
    stems = None
    affixes = None
    memory = None
    alpha = None
    beta = None
    coarts = {"iu": 1, "io": 1, "eu": 1, "eo": 1,
              "ui": -1, "oi": -1, "ue": -1, "oe": -1}

    def __init__(self, type, slen=4, alen=1, lexsize=256):
        """
        Initialize an agent with an exhaustive lexicon made up of
        4-vowel stems (TODO: incorporate consonants,
            affixes & variable word-lengths)
        """
        # vowels {i,u,e,o} in articulatory features (hi, bk, rd) \in {-1,0,1}
        self.vowels = N.array(((1.0, 0.0, 0.0),
                               (1.0, 1.0, 0.0),
                               (0.0, 0.0, 0.0),
                               (0.0, 1.0, 0.0)))
        self.vf = {(1.0, 0.0, 0.0): "i",
                   (1.0, 1.0, 0.0): "u",
                   (0.0, 0.0, 0.0): "e",
                   (0.0, 1.0, 0.0): "o"}
        self.consonants = list("bcdfghjklmnpqrstvwxyz")
        # acoustic:articulatory mapping fxn for vowel prototypes
        # acoustic reps are F1,F2' pairs, articulatory reps are feature-based
        self.vowel_map = {}
        self.vowel_spread = 0
        self.memory = N.empty((lexsize, slen, 2))
        # each agent has its own articulatory variability
        #TODO: maybe this should be inferred by the learners
        #      on the basis of their data?
        self.alpha = N.random.normal(15, 2)
        self.beta = N.random.normal(2, 0.25)
        if self.beta < 1.0:
            self.beta = 1.1

        if type == "learner":
            self.stems = N.empty((lexsize, 4, 3), dtype=float)
            #self.affixes = N.empty((1,4))
        elif type == "speaker":
            tmp = [[x, y, 0.0] for x in [0.0, 1.0] for y in [0.0, 1.0]]
            self.stems = N.array([[a, b, c, d] for a in tmp for b in tmp
                                  for c in tmp for d in tmp])
        else:
            sys.exit("Undefined agent type. Aborting.")
        # vectorized versions of some fxns
        self.vec_perceive = vectorize(self.perceive)
        self.vec_articulate = vectorize(self.articulate)
        self.vec_acoustify = vectorize(self.acoustify)

    def articulate(self, ar_in):
        """
        Add gaussian noise and undershoot to articulation
        (beta distribution models undershoot)
        """
        #pdb.set_trace()
        ar_out = ar_in * 2 - 1
        ar_out[:, 0:2] *= N.random.beta(self.alpha, self.beta, (4, 2))
        #ar_out[:,0:2] += N.random.normal(0,0.001)
        ar_out = 0.5 * ar_out + 0.5
        return ar_out

    def acoustify(self, ar_in):
        """
        Calculate values of F1-F4 from the articulatory output rep.
        ar_in: numpy array (shape=(4,3)) of articulatory descriptions with
                feature values in [0,1]

        Return: Acoustic rep of articulatory description (in Bark units)
        """
        formants = N.ones((1, 4), dtype=int)
        for seg in ar_in:
            (hi, bk, rd) = tuple(seg)
            F1 = int(((-392+392*rd)*(hi*hi) + (596-668*rd)*hi + (-146+166*rd))*(bk*bk) + ((348-348*rd)*(hi*hi) + (-494+606*rd)*hi + (141-175*rd))*bk + ((340-72*rd)*(hi*hi) + (-796+108*rd)*hi + (708-38*rd)))
            F2 = int(((-1200+1208*rd)*(hi*hi) + (1320-1328*rd)*hi + (118-158*rd))*(bk*bk) + ((1864-1488*rd)*(hi*hi) + (-2644+1510*rd)*hi + (-561+221*rd))*bk + ((-670+490*rd)*(hi*hi) + (1355-697*rd)*hi + (1517-117*rd)))
            F3 = int(((604-604*rd)*(hi*hi) + (1038-1178*rd)*hi + (246+566*rd))*(bk*bk) + ((-1150+1262*rd)*(hi*hi) + (-1443+1313*rd)*hi + (-317-483*rd))*bk + ((1130-836*rd)*(hi*hi) + (-315+44*rd)*hi + (2427-127*rd)))
            F4 = int(((-1120+16*rd)*(hi*hi) + (1696-180*rd)*hi + (500+522*rd))*(bk*bk) + ((-140+240*rd)*(hi*hi) + (-578+214*rd)*hi + (-692-419*rd))*bk + ((1480-602*rd)*(hi*hi) + (-1220+289*rd)*hi + (3678-178*rd)))
            formants = N.vstack((formants, N.array((F1, F2, F3, F4))))
        return formants[1:, ]

    def coarticulate(self, ar_in, formants_in, antic, persev):
        """
        Shift values of formants in acoustic representation according to
        user-spec'd amount of "coarticulation"
        ar_in: numpy array (shape=(4,3)) of articulatory descriptions with
                feature values in [0,1]
        formants_in: numpy array of acoustic descriptions
        antic: amount of anticipatory coarticulation
               (i.e. amount by which to shift F2)
        persev: amount of perseverative coarticulation
                (i.e. amount by which to shift F2)

        Return: acoustic rep with shifted formants
        """
        formants_out = copy.copy(formants_in)
        if antic:              # anticipatory coarticulation
            # next_vowel = None
            # cur_vowel = None
            for i in range(3):
                cur_ar = ar_in[i]
                next_ar = ar_in[i + 1]
                ## DODGY ROUNDING HAPPENING HERE;
                ## NOT CERTAIN THAT IT'S JUSTIFIED...
                try:
                    self.vf[tuple(N.round(cur_ar))]
                except KeyError:
                    index = N.random.randint(len(self.letters))
                    self.vf[N.round(tuple(cur_ar))] = self.letters[index]
                cur_v = self.vf[tuple(N.round(cur_ar))]
                try:
                    self.vf[tuple(N.round(next_ar))]
                except KeyError:
                    index = N.random.randint(len(self.letters))
                    self.vf[tuple(N.round(next_ar))] = self.letters[index]
                next_v = self.vf[tuple(N.round(next_ar))]
                key = "".join([cur_v, next_v])
                if key in self.coarts:
                    # we're in a coarticulatory configuration
                    formants_out[i][1] -= self.coarts[key] * antic
        if persev:              # perseverative coarticulation
            # prev_vowel = None
            # cur_vowel = None
            for i in range(1, 4):
                cur_ar = ar_in[i]
                prev_ar = ar_in[i - 1]
                ## DODGY ROUNDING HAPPENING HERE, TOO
                try:
                    self.vf[tuple(N.round(cur_ar))]
                except KeyError:
                    index = N.random.randint(len(self.letters))
                    self.vf[N.round(tuple(cur_ar))] = self.letters[index]
                cur_v = self.vf[tuple(N.round(cur_ar))]
                try:
                    self.vf[tuple(N.round(prev_ar))]
                except KeyError:
                    index = N.random.randint(len(self.letters))
                    self.vf[tuple(N.round(prev_ar))] = self.letters[index]
                prev_v = self.vf[tuple(N.round(prev_ar))]
                key = "".join([prev_v, cur_v])
                if key in self.coarts:
                    # we're in a coarticulatory configuration
                    formants_out[i][1] += self.coarts[key] * persev
        return formants_out

    def perceive(self, word, comp=False):
        """
        Store 2-d (F1,F2') acoustic form of perceived word in memory.
        word : 2-d numpy array
        comp : perceptual compensation...if not False, should be a float in
               (0,1) for likelihood of misparse

        crit  :: critical distance between formants for calculating F2'
        w1, w2 :: weighting factors from de Boer (2000).
                  He says they're a bit ad hoc...check this!
        """
        if comp:
            pass
        crit = 347
        # TODO: probably lots below here
        new_word = N.ones((1, 2))
        for f in word:
            w1 = (crit - (f[2] - f[1])) / crit
            w2 = ((f[3] - f[2]) - (f[2] - f[1])) / (f[3] - f[1])
            if f[2] - f[1] > crit:
                new_word = N.vstack((new_word, (f[0], f[1])))
            elif f[3] - f[1] > crit:
                new_word = N.vstack((new_word, (f[0], (((2 - w1) * f[1] + w1 * f[2]) / 2))))
            elif f[2] - f[1] < f[3] - f[2]:
                new_word = N.vstack((new_word, (f[0], ((((w2 * f[1]) + (2 - w2) * f[2]) / 2) - 1))))
            elif f[2] - f[1] >= f[3] - f[2]:
                new_word = N.vstack((new_word, (f[0], ((((2 - w2) * f[2] + w2 * f[3]) / 2) - 1))))
        return new_word[1:, ]

    def encode(self, word, index):
        self.memory[index] = word

    def learn_vowels(self, data=None):
        """
        Infer articulatory feature reps of vowels from
        stored acoustically encoded words.
        Step 1: find acoustic prototypes for vowels using k-means clustering
        Step 2: convert acoustic prototypes to articulatory feature reps: analysis-by-synthesis.
                    - choose an articulatory rep, and see how closely its (noise-free)
                      articulation matches the acoustic output
        """
        #pdb.set_trace()
        if not data:
            data = self.memory
        # find acoustic prototypes by clustering over stored acoustic reps
        raw_data = data.reshape(4 * len(self.stems), 2)
        ac_vowels, ac_spread = vq.kmeans(raw_data, 4)
        # find articulatory reps by comparing synthesized output vowels to
        # acoustic prototypes
        # start with candidate list of "all possible" articulations
        tmp_ar = N.empty((1, 3))
        rd = 0.0
        for hi in [0.0, 1.0]:
            for bk in [0.0, 1.0]:
                tmp_ar = N.vstack((tmp_ar, N.array([hi, bk, rd])))
        tmp_ar = tmp_ar[1:]
        while len(self.vowel_map) < 4:
            # no noise (since this shouldn't be running through the "mouth")
            tmp_ac = self.perceive(self.acoustify(tmp_ar))
            for v in ac_vowels:
                dists = N.sqrt(N.sum((v - tmp_ac)**2, axis=1))
                d = 0
                while True:
                    if dists[d] < (2 * ac_spread):
                        # found an articulatory prototype
                        self.vowel_map[tuple(v)] = tmp_ar[d]
                        # remove it from the candidate list
                        tmp_ar = N.vstack((tmp_ar[:d], tmp_ar[d + 1:]))
                        tmp_ac = N.vstack((tmp_ac[:d], tmp_ac[d + 1:]))
                        break
                    d += 1
                    if d == len(dists):
                        # take the best of the bad ones
                        index = N.argmin(dists)
                        self.vowel_map[tuple(v)] = tmp_ar[index]
                        break
        self.vowel_spread = ac_spread
        return self.vowel_map

    def learn_words(self):
        """
        Convert encoded acoustic reps of words to articulatory representations.
        This assumes that vowel prototypes have already been learned, and you
        can just reverse the acoustic/articulatory mapping. (I was using a
        horrible guessing-based naive "analysis-by-synthesis" before, and got nowhere.)
        """
        codebook = N.array(self.vowel_map.keys())
        for i in range(len(self.memory)):
            codes, dist = vq.vq(self.memory[i], codebook)
            self.stems[i] = N.array([self.vowel_map[tuple(codebook[x])] for x
                                     in codes])
        return self.stems

    def harmony(self):
        """
        Calculate harmony of speaker's lexicon, in terms of number of words
        that are "harmonic" (i.e. harmonic across the three last segments).
        """
        back = 0
        for i in range(self.stems.shape[0]):
            tmp = N.sum(self.stems[i][1:, 1])
            if tmp == 0 or tmp == 3:
                back += 1
        ### FOR PROPORTIONAL HARMONY ###
        #return float(back)/self.stems.shape[0]
        return back

    def backcount(self, word):
        return sum([seg[1] for seg in word])


def usage():
    sys.exit("\n\tUsage: sys.argv[0] <popsize> <gens> "
             "<antic coart> <persev coart>\n")


def main(argv):
    # check & process command-line args
    if len(argv) != 4:
        usage()
    num_agents = int(argv[0])
    gens = int(argv[1])
    coart_a = int(argv[2])
    coart_p = int(argv[3])

    # setup population & output files
    adult_population = []
    for i in range(num_agents):
        adult_population.append(Agent("speaker"))     # initial population
    # write initial lexicon(s) to file
    for index in range(num_agents):
        bar = file("lex/%s_%d_initial.lex" % ("".join([str(gens), str(coart_a), str(coart_p)]), index), "w")
        bar.write("# Harmony: %f\n" % adult_population[index].harmony())
        for i in range(len(adult_population[index].stems)):
            for j in range(adult_population[index].stems.shape[1]):
                bar.write(adult_population[index].vf[tuple(adult_population[index].stems[i][j])])
            bar.write("\n")
        bar.close()
    # files to track the proportion of harmony
    fnames = ["data/harmony_%s_%i.dat" % ("".join([str(gens), str(coart_a), str(coart_p)]), i) for i in range(num_agents)]
    harm_files = [file(fname, "w") for fname in fnames]

    # main loop of simulation
    for gen in xrange(int(gens)):
        # create population of kids
        child_population = []
        for i in range(num_agents):
            child_population.append(Agent("learner", lexsize=len(adult_population[0].stems)))
        # iterate over all adults for each child...en(does that make sense?)
        for grownup in adult_population:
            # spit out lexicon in random order
            output_order = range(len(adult_population[0].stems))
            #N.random.shuffle(output_order)
            for kid in child_population:
                for word in output_order:
                    articulated = grownup.articulate(grownup.stems[word])
                    acoustified = grownup.acoustify(articulated)
                    coarticulated = grownup.coarticulate(articulated, acoustified, coart_a, coart_p)
                    percept = kid.perceive(coarticulated)
                    kid.encode(percept, word)
                kid.learn_vowels()
                kid.learn_words()

        # replace adults with kids & log harmony
        for i in range(num_agents):
            adult_population[i] = copy.copy(child_population[i])
            harm_files[i].write("%d %f\n" % (gen, adult_population[i].harmony()))
        # compute "harmony" of learned lexicon
        del(child_population)
        sys.stderr.write(".")
    for foo in harm_files:
        foo.close()

    # output final lexicon(s)
    for index in range(num_agents):
        bar = file("lex/%s_%d_final.lex" % ("".join([str(gens), str(coart_a), str(coart_p)]), index), "w")
        bar.write("# Harmony: %f\n" % adult_population[index].harmony())
        for i in range(len(adult_population[index].stems)):
            for j in range(adult_population[index].stems.shape[1]):
                bar.write(adult_population[index].vf[tuple(adult_population[index].stems[i][j])])
            bar.write("\n")
        bar.close()

    sys.stderr.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
