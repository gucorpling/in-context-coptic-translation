# Code for "Syntax as a Rosetta Stone: Universal Dependencies for In-Context Coptic Translation

ACL Anthology : [https://aclanthology.org/2026.findings-acl.1803/](https://aclanthology.org/2026.findings-acl.1803/)

The main entrypoints for the code are `local.py` and `gpt.py` which help run inferences.

Here is a summary of the code in the repository
* `lrmdata.py` combines the different data sources with components to provide the specific augmented prompt.
* `configs/` contains different configs and are loaded through `util.py`
  * `lexicon.py`, `ud_parsing.py`, `constructions/` are effectively components of the system.  
* `metrics.py` wraps implements of different metrics
* `data` contains  
  * lexicon from Coptic Dictionary: [https://github.com/KELLIA/dictionary](https://github.com/KELLIA/dictionary)
  * ud from UD Coptic Scriptorium [https://github.com/UniversalDependencies/UD_Coptic-Scriptorium/](https://github.com/UniversalDependencies/UD_Coptic-Scriptorium/)
  * additional evaluation data from [https://github.com/somiyagawa/GreekCopticMTEval/tree/main](https://github.com/somiyagawa/GreekCopticMTEval/)
* `err_ana` contains the error analysis, `results` contains our extensive results
* There are additional code related to specific utilities or analytical tools.
* `requirements.txt` contains explicit hard record of the experimental environment.

If you have any questions or queries about the code, please reach out to the authors.

Please cite using the `bib` from the ACL Anthology publication linked above.