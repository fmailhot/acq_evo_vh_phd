#!/usr/bin/python
import sys
from LIbPhon import LIbPhon
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
import pylab as plt
params = {'backend': 'ps',
          'legend.fontsize': 14,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'text.usetex': True}
plt.rcParams.update(params)


if len(sys.argv) != 3:
    print '\nUsage: %s <LEX MEANING> <opaq|trans>\n' % sys.argv[0]
    sys.exit(2)

teach = LIbPhon(teacher=True,
                lex="teacher_lexicon_hTrue_cTrue_pTrue_n%s.pck" % sys.argv[2])
nom = teach.produce("%s NOM" % sys.argv[1])
acc = teach.produce("%s ACC" % sys.argv[1])
pl_nom = teach.produce("%s PL NOM" % sys.argv[1])
pl_acc = teach.produce("%s PL ACC" % sys.argv[1])

plt.subplot(211)
l1 = plt.plot(pl_nom[:10][:, 0], "k-", linewidth=3, label="F1")
l2 = plt.plot(pl_nom[:10][:, 1], "k--", linewidth=3, label="F2")
plt.legend()
plt.text(0.5, 2700, r"\textbf{\textsc{%s nom}}" % sys.argv[1], size="x-large")
plt.text(5.5, 2700, r"\textbf{\texttt{%sgu}}" % sys.argv[1], size="x-large")
plt.xlim(0, 12)
plt.ylim(0, 3250)

if sys.argv[2] == "opaq":
    suff = "bo"
else:
    suff = "be"

plt.subplot(212)
l1 = plt.plot(pl_acc[:12][:, 0], "k-", linewidth=3, label="F1")
l2 = plt.plot(pl_acc[:12][:, 1], "k--", linewidth=3, label="F2")
plt.legend()
plt.text(0.5, 2700, r"\textbf{\textsc{%s acc}}" % sys.argv[1],
         size="x-large")
plt.text(5.5, 2700, r"\textbf{\texttt{%sgu%s}}" % (sys.argv[1], suff),
         size="x-large")
plt.xlim(0, 12)
plt.ylim(0, 3250)

plt.show()
