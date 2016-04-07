#!/usr/bin/python
""" Set up/execute runs of exemplar-based harmony learning model. """
import sys
import os.path
import copy
import optparse as opt
import cPickle as pickle
import numpy as np
from random import choice, sample
from glob import glob
from LIbPhon import LIbPhon
__author__ = "Fred Mailhot (fred.mailhot@gmail.com)"


def train(opts):
    """Iterated trans/acq until gens exhausted, and 'big enough' lexicon."""
    lexfile = "teacher_lexicon_h%s_c%s_p%s_n%s.pck" % (opts.vharmony,
                                                       opts.case,
                                                       opts.plural,
                                                       opts.neutrality)
    # zeroth generation
    teachers = [LIbPhon(teacher=True, lex=lexfile,
                        knn=int(opts.knn), coart=float(opts.antic)) for x
                in range(int(opts.size))]
    # learners dump 100 lexicons, or 1 per generation

    dumps = np.linspace(0, int(opts.lexsize), num=100).astype("int32")
    print "TEACHER --> LABEL --> LEARNER\n============================="
    for g in xrange(int(opts.gens)):
        # setup (i+1)th gen of learners
        learners = [LIbPhon(knn=int(opts.knn), coart=float(opts.antic)) for
                    x in range(int(opts.size))]
        trainfiles = [open("trainfile_learner_" + str(x) +
                           "_gen%d_%s_%s_%s.log" % (g + 1, opts.lexsize,
                                                    opts.instances,
                                                    opts.neutrality), "w")
                      for x in range(int(opts.size))]
        sys.stderr.write("=GEN %d=" % (g + 1))
        print "==GEN %d==" % (g + 1)
        while(True):
            # training done?
            if np.median([len(learner.lexicon.keys()) for
                          learner in learners]) == int(opts.lexsize):
                break
            # select uniform random learner
            learner = choice(learners)
            # select uniform random teacher according to information flow
            if "v" in opts.flow:
                # vert: select parent
                teacher = teachers[learners.index(learner)]
            else:
                # oblique: select random teacher
                teacher = choice(teachers)
            if "p" in opts.flow:
                # peers allowed: switch teacher to random peer
                # with prob opts.xfactor
                if np.random.random() < float(opts.xfactor):
                    peers = range(int(opts.size))
                    while(True):
                        if peers == []:
                            break
                        tmp_teach = choice(peers)
                        # make sure they're distinct and teacher
                        # knows >=1 labels
                        if ((learners[tmp_teach] is learner) or
                                (learners[tmp_teach].lexicon.keys() == [])):
                            peers.pop(peers.index(tmp_teach))
                            continue
                        else:
                            teacher = learners[tmp_teach]
                            break
            # select label at random
            label = choice(teacher.lexicon.keys())
            trainfiles[learners.index(learner)].write("%s\n" % label)
            print "%x --> %s --> %x" % (id(teacher), label, id(learner))
            # learner gets multiple tokens
            # (N.B. this is dumb, but avoids obligatory wait to generate)
            for j in range(int(opts.instances)):
                input = (label, teacher.produce(label))
                learner.categorize(input)
            sys.stderr.write(".")           # visual aid to track sim progress
            if int(opts.gens) == 1:
                # dump lexicon periodically for single-gen runs
                i = len(learner.lexicon.keys())
                if i in dumps:
                    if os.path.isfile("lexicon_learner_%d_gen%d_%s_%s_%s_%04d.pck" %
                                      (learners.index(learner), g + 1,
                                       opts.lexsize, opts.instances,
                                       opts.neutrality, i)):
                        pass    # no need to dump, since it's been done before
                    else:
                        lexfile = open("lexicon_learner_%d_gen%d_%s_%s_%s_%04d.pck" %
                                       (learners.index(learner), g + 1,
                                        opts.lexsize, opts.instances,
                                        opts.neutrality, i), "w")
                        print "!!!%x dump (lexsize: %d)!!!" % (id(learner), i)
                        sys.stderr.write("!")  # track lex dumping
                        pickle.dump(learner.lexicon, lexfile,
                                    pickle.HIGHEST_PROTOCOL)
                        lexfile.close()
        # dump all lexicons @ end-of-gen
        for learner in learners:
            if os.path.isfile("lexicon_learner_%d_gen%d_%s_%s_%s_final.pck" %
                              (learners.index(learner), g + 1, opts.lexsize,
                               opts.instances, opts.neutrality)):
                pass    # no need to dump, since it's been done before
            else:
                lexfile = open("lexicon_learner_%d_gen%d_%s_%s_%s_final.pck" %
                               (learners.index(learner), g + 1, opts.lexsize,
                                opts.instances, opts.neutrality), "w")
                sys.stderr.write("%d dump final!!!" % learners.index(learner))
                pickle.dump(learner.lexicon, lexfile, pickle.HIGHEST_PROTOCOL)
                lexfile.close()
        for f in trainfiles:
            f.close()
        # new gen becomes old gen
        teachers = copy.copy(learners)
        del learners
    sys.stderr.write("DONE.\n")


def test_prod(opts):
    """ Track RMSE vs lexicon size on production of held out labels """
    lexfile = "teacher_lexicon_h%s_c%s_p%s_n%s.pck" % (opts.vharmony,
                                                       opts.case,
                                                       opts.plural,
                                                       opts.neutrality)
    teacher = LIbPhon(teacher=True, lex=lexfile)

    all_train_dirs = glob("%s/*" % opts.neutrality)
    train_dirs = []
    for d in all_train_dirs:
        try:
            t_opts = file(d + "/options.log").read().split()
            if t_opts[2] == opts.flow and t_opts[3] == opts.lexsize:
                train_dirs.append(d)
        except IOError:
            print "No options.log file in " + d
            continue
    if train_dirs == []:
        sys.stderr.write("\nNo options.log files found.\n")
        sys.exit(2)
    d = sorted(train_dirs, reverse=True)[0]
    # d is most recent dir corresponding to the user-spec'd harmonic config
    train_files = glob(d + "/train*")
    train_data_raw = [x.strip() for f in train_files for
                      x in file(f).readlines()]
    train_data = set(train_data_raw)
    test_data_raw = set(teacher.lexicon.keys()) - train_data
    if len(test_data_raw) > 500:
        sample_size = 500
    else:
        sample_size = len(test_data_raw) - len(test_data_raw) % 100
    test_data = sample(test_data_raw, sample_size)
    print "%d test words" % len(test_data)
    # I now have a set of held-out data that none of my trained agents has seen

    learners = [LIbPhon(knn=int(opts.knn)) for i in range(int(opts.size))]
    for learner in learners:
        sys.stderr.write("%d " % learners.index(learner))
        total_f2_rmse = []
        for lexicon in sorted(glob(d + "/lexicon_learner_%d_*" %
                                   learners.index(learner))):
            lexicon_f2_rmse = 0
            lexfile = file(lexicon)
            learner.lexicon = pickle.load(lexfile)
            lexfile.close()
            for word in test_data:
                # target is average of teacher's cloud
                target = np.mean(teacher.lexicon[word], axis=0).astype("int32")
                output = learner.produce(word)
                # accumulate root-mean-squared-error
                # pylint: disable E225
                lexicon_f2_rmse += np.sqrt(np.sum(((target - output)[:, 1]**2), axis=0) /
                                           float(target.shape[0]))
            # root-mean-squared-error per word over the test data
            total_f2_rmse.append(lexicon_f2_rmse / len(test_data))
            sys.stderr.write(".")
            del learner.lexicon
        if total_f2_rmse == []:
            print "DOH: " + d
        sys.stderr.write("\n")
        # err = np.array(total_f2_rmse)
        np.savetxt(d + "/F2_sse_learner_%d.dat" % learners.index(learner),
                   total_f2_rmse, fmt="%f")
        fout = file(d + "/test_data.dat", "w")
        fout.write("\n".join(test_data))
        fout.close()


def test_harm(opts):
    """ Track harmony across lexicon across generations """
    learners = range(int(opts.size))

    all_train_dirs = glob("%s/*" % opts.neutrality)
    train_dirs = []
    for d in all_train_dirs:
        try:
            t_opts = file(d + "/options.log").read().split()
            if t_opts[2] == opts.flow and t_opts[3] == opts.lexsize:
                train_dirs.append(d)
        except IOError:
            print "No options.log file in " + d
            continue
    if train_dirs == []:
        sys.stderr.write("\nNo options.log files found anywhere.\n")
        sys.exit(2)
    d = sorted(train_dirs, reverse=True)[0]
    # d is most recent dir corresponding to the user-spec'd harmonic config

    for learner in learners:
        sys.stderr.write("Learner:\t%d\nGen:\t" % learner)
        harmfile = open("harmfile_learner_%d.log" % learner, "w")
        for g in range(int(opts.gens)):
            sys.stderr.write(str(g) + " ")
            # open biggest lexfile for given learner and generation
            lexfile = file(sorted(glob(d + "/lexicon_learner_%d_gen%d_*" %
                                       (learner, g + 1)))[-1])
            lexicon = pickle.load(lexfile)
            lexfile.close()
            lexicon_vars = []
            for k in lexicon.keys():
                lexicon_vars.append(np.var(np.mean(lexicon[k], 0)[:, 1][np.nonzero(np.mean(lexicon[k], 0)[:, 1])]))
            del lexicon
            harmfile.write(str(np.mean(lexicon_vars)) + "\n")
        sys.stderr.write("\n")
        harmfile.close()


def test_class(opts):
    """ Track F-score vs lexicon size on classification of held-out tokens """
    lexfile = "teacher_lexicon_h%s_c%s_p%s_n%s.pck" % (opts.vharmony,
                                                       opts.case,
                                                       opts.plural,
                                                       opts.neutrality)
    teacher = LIbPhon(teacher=True, lex=lexfile)

    all_train_dirs = glob("%s/*" % opts.neutrality)
    train_dirs = []
    for d in all_train_dirs:
        t_opts = file(d + "/options.log").read().split()
        if t_opts[2] == opts.flow and t_opts[3] == opts.lexsize:
            train_dirs.append(d)
    d = sorted(train_dirs, reverse=True)[0]
    # d is most recent dir corresponding to the user-spec'd harmonic config
    train_files = glob(d + "/train*")
    train_data_raw = [x.strip() for f in train_files for
                      x in file(f).readlines()]
    train_data = set(train_data_raw)
    test_data_raw = set(teacher.lexicon.keys()) - train_data
    if len(test_data_raw) > 500:
        sample_size = 500
    else:
        sample_size = len(test_data_raw) - len(test_data_raw) % 100
    test_data = sample(test_data_raw, sample_size)
    print "%d test words" % len(test_data)
    # I now have a set of held-out data that none of my trained agents has seen

    learners = [LIbPhon(knn=int(opts.knn)) for i in range(int(opts.size))]
    for learner in learners:
        sys.stderr.write("%d " % learners.index(learner))
        total_f2_rmse = []
        for lexicon in sorted(glob(d + "/lexicon_learner_%d_*" %
                                   learners.index(learner))):
            lexicon_f2_rmse = 0
            lexfile = file(lexicon)
            learner.lexicon = pickle.load(lexfile)
            lexfile.close()
            for word in test_data:
                # target is average of teacher's cloud
                target = np.mean(teacher.lexicon[word], axis=0).astype("int32")
                output = learner.produce(word)
                # root-mean-squared-error
                lexicon_f2_rmse += np.sqrt(np.sum(((target - output)[:, 1]**2),
                                           axis=0) / float(target.shape[0]))
            total_f2_rmse.append(lexicon_f2_rmse / len(test_data))
            sys.stderr.write(".")
            del learner.lexicon
        if total_f2_rmse == []:
            print "DOH: " + d
        sys.stderr.write("\n")
        # err = np.array(total_f2_rmse)
        np.savetxt(d + "/F2_sse_learner_%d.dat" % learners.index(learner),
                   total_f2_rmse, fmt="%f")
        fout = file(d + "/test_data.dat", "w")
        fout.write("\n".join(test_data))
        fout.close()


def check_args(opts):
    """Verify that command-line args are OK. opts is the dictionary
    created by optparse."""
    if ((opts.gens is None) or
        (opts.size is None) or
        (opts.flow not in ["v", "o", "vp", "op"]) or
        (opts.lexsize is None) or
        (opts.instances is None) or
            (opts.task is None)):
        sys.exit("\nOne or more mandatory options missing.\n\n")
    elif ((int(opts.gens) < 1) or
          (int(opts.size < 1))):
        sys.exit("\n>=one learner and one teacher for one generation.\n")
    elif (int(opts.size) < 2 and
            ("o" in opts.flow or "p" in opts.flow)):
        sys.exit("\nPopulation topology and flow parameter incompatible.\n\n")
    elif opts.task not in ["train", "test_prod", "test_class", "test_harm"]:
        sys.exit("\nTask must be one of 'train','test_prod', "
                 "'test_class', 'test_harm'\n")
    elif ((opts.vharmony not in ["True", "False"]) or
          (opts.case not in ["True", "False"]) or
          (opts.plural not in ["True", "False"])):
        sys.exit("\nvharmony, case, and plural must be "
                 "in ['True', 'False'].\n")
    else:
        return(0)


#### execution starts here for CLI script invocation ####
if __name__ == "__main__":
    parser = opt.OptionParser(description=__doc__, version="%prog v0.99")
    parser.add_option("-g", "--generations",
                      help="# trans/acq cycles (mandatory)",
                      dest="gens", action="store")
    parser.add_option("-s", "--population-size",
                      help="Number of individuals in a generation (mandatory)",
                      dest="size", action="store")
    parser.add_option("-f", "--information-flow",
                      help="Direction of information flow (mandatory)",
                      dest="flow", action="store")
    parser.add_option("-l", "--lexicon-size",
                      help="Size of lexicon to be learned (mandatory)",
                      dest="lexsize", action="store")
    parser.add_option("-i", "--instances",
                      help="# tokens of each lexical item (mandatory)",
                      dest="instances", action="store")
    parser.add_option("-t", "--task",
                      help=("<train|test_prod|test_class|test_harm> "
                            "(mandatory)"),
                      dest="task", action="store")
    parser.add_option("-v", "--vowel-harmony",
                      help="Specify <True|False> (default True)",
                      dest="vharmony", default="True", action="store")
    parser.add_option("-c", "--case-morph",
                      help="Specify <True|False> (default True)",
                      dest="case", default="True", action="store")
    parser.add_option("-p", "--plural-morph",
                      help="Specify <True|False> (default True)",
                      dest="plural", default="True", action="store")
    parser.add_option("-n", "--neutrality",
                      help=("Specify <trans|opaq|None> "
                            "neutrality (mandatory if -p True)"),
                      dest="neutrality", action="store")
    parser.add_option("-k", "--k-neighbours",
                      help="# neighbours used in deciding output (default 5)",
                      dest="knn", default="5", action="store")
    parser.add_option("-x", "--x-factor",
                      help="P(interact with peer) (default 0.5)",
                      dest="xfactor", default="0.5", action="store")
    parser.add_option("-a", "--antic-coart",
                      help=("Degree of (anticipatory) "
                            "coarticulation (default 0.0)"),
                      dest="antic", default="0.0", action="store")

    (opts, args) = parser.parse_args()

    if check_args(opts):
        parser.print_help()
        sys.exit(2)

    eval(opts.task + "(opts)")
