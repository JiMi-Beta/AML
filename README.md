# AML

Code repository for our experiments comparing random search, SMAC, and TPEs (Tree Parzen Estimators).

The folders random-search/experimentScripts, smac, and tpe contain all scripts for the different datasets. To run those scripts (after installing dependencies which can be inferred from import statements), one can change the paths specified in there such that the reference to the corresponding file is correct. Bear in mind that one some scripts can only be run on Linux systems (due to dependencies).

SMAC has special install instructions: https://automl.github.io/SMAC3/master/installation.html

Random search and TPE can be installed through pip install hyperopt.

After having followed the above steps, one can run one of the scripts using 'python3 scriptname.py > resultsfilename'
