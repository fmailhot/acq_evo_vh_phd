#!/usr/bin/python
######################################################################
# lex_maker.py
# Fred Mailhot <fmailhot@connect.carleton.ca>
######################################################################
"""Generate lexicon for teacher agent. Options specify with or without
vowel harmony, and morphology & neutral vowels.
"""
import sys
import optparse as opt
import cPickle as pickle
import numpy as np
from itertools import product

vowels = {"i": np.array([1.0, 0.0, 0.0]),
          "u": np.array([1.0, 1.0, 1.0]),
          "e": np.array([0.5, 0.0, 0.0]),
          "o": np.array([0.5, 1.0, 1.0])}
ft_vowels = ["i", "e"]
bk_vowels = ["u", "o"]
ft_syllables = ["bi", "be", "gi", "ge", "di", "de"]
bk_syllables = ["bu", "bo", "gu", "go", "du", "do"]
# "be/bo" harmonize, not "gu"
suffix_morphs = {"NOM": "",
                 "ACC": {"ft": "be",
                         "bk": "bo"},
                 "PL": "gu"}

lexicon = {}


def build_lexicon(opts, syllable="CV", instances=50):
    #TODO: get the hard-coded numbers out of here
    if opts.vh == "true":
        front_nom_forms = np.array(list(product(ft_syllables, repeat=4)))
        back_nom_forms = np.array(list(product(bk_syllables, repeat=4)))

        if opts.case == "true":
            front_acc_forms = np.column_stack((front_nom_forms,
                                               np.array([suffix_morphs["ACC"]["ft"]] * 1296)))
            back_acc_forms = np.column_stack((back_nom_forms,
                                              np.array([suffix_morphs["ACC"]["bk"]] * 1296)))

            if opts.plural == "true":
                front_pl_nom_forms = np.column_stack((front_nom_forms,
                                                      np.array([suffix_morphs["PL"]] * 1296)))
                back_pl_nom_forms = np.column_stack((back_nom_forms,
                                                     np.array([suffix_morphs["PL"]] * 1296)))

                if opts.neutral == "opaq":
                    front_pl_acc_forms = np.column_stack((front_pl_nom_forms,
                                                          np.array([suffix_morphs["ACC"]["bk"]] * 1296)))
                    back_pl_acc_forms = np.column_stack((back_pl_nom_forms,
                                                         np.array([suffix_morphs["ACC"]["bk"]] * 1296)))
                else:
                    front_pl_acc_forms = np.column_stack((front_pl_nom_forms,
                                                          np.array([suffix_morphs["ACC"]["ft"]] * 1296)))
                    back_pl_acc_forms = np.column_stack((back_pl_nom_forms,
                                                         np.array([suffix_morphs["ACC"]["bk"]] * 1296)))
    else:
        nom_forms = np.array(list(product(ft_syllables + bk_syllables, repeat=4)))
        if opts.case == "true":
            # eventually enable front or back ACC suffix as user param
            acc_forms = np.column_stack((nom_forms,
                                         np.array([suffix_morphs["ACC"]["ft"]] * 20736)))

            if opts.plural == "true":
                pl_nom_forms = np.column_stack((nom_forms,
                                                np.array([suffix_morphs["PL"]] * 20736)))
                pl_acc_forms = np.column_stack((pl_nom_forms,
                                                np.array([suffix_morphs["ACC"]["ft"]] * 20736)))

    if opts.vh == "true":
        size = front_nom_forms.shape[0]
    else:
        size = nom_forms.shape[0]
    for i in range(size):
        sys.stderr.write(str(size - i) + " ")
        for j in range(instances):
            if opts.vh == "true":
                ft_nom_ex = np.append(np.vstack([make_syllable(x, syllable) for
                                                 x in front_nom_forms[i]]),
                                      [[0, 0], [0, 0], [0, 0], [0, 0]], axis=0)
                bk_nom_ex = np.append(np.vstack([make_syllable(x, syllable) for
                                                 x in back_nom_forms[i]]),
                                      [[0, 0], [0, 0], [0, 0], [0, 0]], axis=0)
                if opts.case == "true":
                    ft_acc_ex = np.append(np.vstack([make_syllable(x, syllable)
                                                     for x in front_acc_forms[i]]),
                                          [[0, 0], [0, 0]], axis=0)
                    bk_acc_ex = np.append(np.vstack([make_syllable(x, syllable)
                                                     for x in back_acc_forms[i]]),
                                          [[0, 0], [0, 0]], axis=0)
                    if opts.plural == "true":
                        ft_pl_nom_ex = np.append(np.vstack([make_syllable(x, syllable)
                                                            for x in front_pl_nom_forms[i]]),
                                                 [[0, 0], [0, 0]], axis=0)
                        bk_pl_nom_ex = np.append(np.vstack([make_syllable(x, syllable)
                                                            for x in back_pl_nom_forms[i]]),
                                                 [[0, 0], [0, 0]], axis=0)
                        ft_pl_acc_ex = np.vstack([make_syllable(x, syllable) for
                                                  x in front_pl_acc_forms[i]])
                        bk_pl_acc_ex = np.vstack([make_syllable(x, syllable) for
                                                  x in back_pl_acc_forms[i]])
                ft_nom_lbl = " ".join(["".join(front_nom_forms[i]), "NOM"])
                bk_nom_lbl = " ".join(["".join(back_nom_forms[i]), "NOM"])
                ft_acc_lbl = " ".join(["".join(front_nom_forms[i]), "ACC"])
                bk_acc_lbl = " ".join(["".join(back_nom_forms[i]), "ACC"])
                ft_pl_nom_lbl = " ".join(["".join(front_nom_forms[i]), "PL", "NOM"])
                bk_pl_nom_lbl = " ".join(["".join(back_nom_forms[i]), "PL", "NOM"])
                ft_pl_acc_lbl = " ".join(["".join(front_nom_forms[i]), "PL", "ACC"])
                bk_pl_acc_lbl = " ".join(["".join(back_nom_forms[i]), "PL", "ACC"])
                if not j:
                    lexicon[ft_nom_lbl] = np.expand_dims(ft_nom_ex, axis=0)
                    lexicon[bk_nom_lbl] = np.expand_dims(bk_nom_ex, axis=0)
                    if opts.case == "true":
                        lexicon[ft_acc_lbl] = np.expand_dims(ft_acc_ex, axis=0)
                        lexicon[bk_acc_lbl] = np.expand_dims(bk_acc_ex, axis=0)
                        if opts.plural == "true":
                            lexicon[ft_pl_nom_lbl] = np.expand_dims(ft_pl_nom_ex, axis=0)
                            lexicon[bk_pl_nom_lbl] = np.expand_dims(bk_pl_nom_ex, axis=0)
                            lexicon[ft_pl_acc_lbl] = np.expand_dims(ft_pl_acc_ex, axis=0)
                            lexicon[bk_pl_acc_lbl] = np.expand_dims(bk_pl_acc_ex, axis=0)
                else:
                    lexicon[ft_nom_lbl] = np.vstack((np.expand_dims(ft_nom_ex, axis=0), lexicon[ft_nom_lbl]))
                    lexicon[bk_nom_lbl] = np.vstack((np.expand_dims(bk_nom_ex, axis=0), lexicon[bk_nom_lbl]))
                    if opts.case == "true":
                        lexicon[ft_acc_lbl] = np.vstack((np.expand_dims(ft_acc_ex, axis=0), lexicon[ft_acc_lbl]))
                        lexicon[bk_acc_lbl] = np.vstack((np.expand_dims(bk_acc_ex, axis=0), lexicon[bk_acc_lbl]))
                        if opts.plural == "true":
                            lexicon[ft_pl_nom_lbl] = np.vstack((np.expand_dims(ft_pl_nom_ex, axis=0), lexicon[ft_pl_nom_lbl]))
                            lexicon[bk_pl_nom_lbl] = np.vstack((np.expand_dims(bk_pl_nom_ex, axis=0), lexicon[bk_pl_nom_lbl]))
                            lexicon[ft_pl_acc_lbl] = np.vstack((np.expand_dims(ft_pl_acc_ex, axis=0), lexicon[ft_pl_acc_lbl]))
                            lexicon[bk_pl_acc_lbl] = np.vstack((np.expand_dims(bk_pl_acc_ex, axis=0), lexicon[bk_pl_acc_lbl]))
            else:
                nom_ex = np.append(np.vstack([make_syllable(x, syllable) for
                                              x in nom_forms[i]]),
                                   [[0, 0], [0, 0], [0, 0], [0, 0]], axis=0)
                if opts.case == "true":
                    acc_ex = np.append(np.vstack([make_syllable(x, syllable) for
                                                  x in acc_forms[i]]),
                                       [[0, 0], [0, 0]], axis=0)
                    if opts.plural == "true":
                        pl_nom_ex = np.append(np.vstack([make_syllable(x, syllable)
                                                         for x in pl_nom_forms[i]]),
                                              [[0, 0], [0, 0]], axis=0)
                        pl_acc_ex = np.vstack([make_syllable(x, syllable) for
                                               x in pl_acc_forms[i]])
                nom_lbl = " ".join(["".join(nom_forms[i]), "NOM"])
                acc_lbl = " ".join(["".join(nom_forms[i]), "ACC"])
                pl_nom_lbl = " ".join(["".join(nom_forms[i]), "PL", "NOM"])
                pl_acc_lbl = " ".join(["".join(nom_forms[i]), "PL", "ACC"])
                if not j:
                    lexicon[nom_lbl] = np.expand_dims(nom_ex, axis=0)
                    if opts.case == "true":
                        lexicon[acc_lbl] = np.expand_dims(acc_ex, axis=0)
                        if opts.plural == "true":
                            lexicon[pl_nom_lbl] = np.expand_dims(pl_nom_ex,
                                                                 axis=0)
                            lexicon[pl_acc_lbl] = np.expand_dims(pl_acc_ex,
                                                                 axis=0)
                else:
                    lexicon[nom_lbl] = np.vstack((np.expand_dims(nom_ex,
                                                                 axis=0), lexicon[nom_lbl]))
                    if opts.case == "true":
                        lexicon[acc_lbl] = np.vstack((np.expand_dims(acc_ex,
                                                                     axis=0), lexicon[acc_lbl]))
                        if opts.plural == "true":
                            lexicon[pl_nom_lbl] = np.vstack((np.expand_dims(pl_nom_ex, axis=0), lexicon[pl_nom_lbl]))
                            lexicon[pl_acc_lbl] = np.vstack((np.expand_dims(pl_acc_ex, axis=0), lexicon[pl_acc_lbl]))


def make_vowel(art_in):
    """
        Calculate values of F1-F4 from an articulatory rep.
        Equations from de Boer (2001), who gets them from Maeda (1989)
        In:
            art_in   -- numpy array (shape=(4,3)) of articulatory descriptions
                        with feature values in [0,1]

        Return:
            Acoustic [F1, F2] rep of articulatory description
    """
    (hi, bk, rd) = tuple(art_in)
    formants = np.ones((1, 2), dtype="int32")
    F1 = int(((-392+392*rd)*(hi*hi) + (596-668*rd)*hi + (-146+166*rd))*(bk*bk) + ((348-348*rd)*(hi*hi) + (-494+606*rd)*hi + (141-175*rd))*bk + ((340-72*rd)*(hi*hi) + (-796+108*rd)*hi + (708-38*rd)))
    F2 = int(((-1200+1208*rd)*(hi*hi) + (1320-1328*rd)*hi + (118-158*rd))*(bk*bk) + ((1864-1488*rd)*(hi*hi) + (-2644+1510*rd)*hi + (-561+221*rd))*bk + ((-670+490*rd)*(hi*hi) + (1355-697*rd)*hi + (1517-117*rd)))
    # for estimates of formant noise (based on JNDs)
    #       Kewley-Port et al (1995) in JASA97
    #           !!! Guessing 5 JNDs is OK !!!
    F1 = int(np.random.normal(F1, F1 * 0.01 * 5))
    F2 = int(np.random.normal(F2, F2 * 0.015 * 5))
    formants = np.vstack((formants, np.array((F1, F2))))
    return np.array([F1, F2])


def make_consonant(cons, vow):
    """
        Locus equations for deriving "syllable"-initial F2 values.
        Eqn params taken from Sussman et al (1998), p.247

        In:
            cons    -- character representation of consonant to create
            vow     -- formant [F1,F2] rep of vowel midpoint to use
        Return:
            F2 value for input consonant in syllable initial position
            (plus Gaussian noise??)
    """
    if cons == "b":
        cons_f2 = 231 + (0.813 * vow[1])
    elif cons == "d":
        cons_f2 = 1217 + (0.394 * vow[1])
    elif cons == "g":
        # "g" has two clusters of values, depending on whether it is
        # produced before a front (F2 >~ 1600) or back vowel
        if vow[1] > 1600:
            cons_f2 = 1814 + (0.261 * vow[1])
        else:
            cons_f2 = 169 + (1.223 * vow[1])
    else:
        #TODO(fmailhot): PROPER ERROR HANDLING
        sys.stderr.write("\nThis should not happen.\n")
        sys.exit(2)

    cons_f1 = 120     # by fiat (see. Liberman et al (1955)
    # amount of noise is made-up -- need to check
    cons_f1 = int(np.random.normal(120, 120 * 0.015 * 5))
    cons_f2 = int(np.random.normal(cons_f2, cons_f2 * 0.015 * 5))
    return np.array([cons_f1, cons_f2])


def make_syllable(cv, shape):
    c = cv[0]
    v = cv[1]

    if shape not in ["V", "CV"]:
        sys.exit("Invalid syllable shape.")

    vowel = make_vowel(vowels[v])
    if shape == "CV":
        consonant = make_consonant(c, vowel)
    else:
        consonant = np.zeros((2, ))
    return np.vstack((consonant, vowel))

if __name__ == "__main__":
    parser = opt.OptionParser(description=__doc__, version="%prog v0.9")
    parser.add_option("-v", "--vowel-harmony",
                      help="<true>|<false> vowel harmony (mandatory)?",
                      dest="vh", action="store")
    parser.add_option("-c", "--case",
                      help="<true>|<false> case morphology (mandatory)?",
                      dest="case", action="store")
    parser.add_option("-p", "--plural",
                      help="<true>|<false> plural morphology (mandatory)?",
                      dest="plural", action="store")
    parser.add_option("-n", "--neutrality",
                      help="<opaq>|<trans> neutrality (mandatory with -p)?",
                      dest="neutral", action="store")
    (opts, args) = parser.parse_args()

    #need to add opts verification code here
    if opts.vh is None or opts.case is None or opts.plural is None:
        parser.print_help()
        sys.exit(2)

    sys.stderr.write("Building...")
    build_lexicon(opts)
    sys.stderr.write("Pickling & Dumping...")
    fout = file("teacher_lexicon_h%s_c%s_p%s_n%s.pck" % (opts.vh, opts.case,
                                                         opts.plural, opts.neutral), "w")
    pickle.dump(lexicon, fout, pickle.HIGHEST_PROTOCOL)
    fout.close()
    sys.stderr.write("Done.\n")
