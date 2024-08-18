## Human Value Detection in Speech Through LLM Ensemble

The `scripts` folder contains the scripts used in training the models, the models were fine tuned using PCAD infrastructure. `only_training.py` is the base that was adjusted for each model execution, `inference.py` created the output dataset with all the models predictions to be used in the ensemble.

The notebook in `notebooks/experiments.ipynb` describes the voting systems used, each voting model has its diagram available on the `images` folder.

Complete description of the work is available at [ValueEval24 task](https://touche.webis.de/clef24/touche24-web/human-value-detection.html#synopsis)

Datasets are available at https://zenodo.org/records/13283288




Some experiments in this work used the PCAD infrastructure, http://gppd-hpc.inf.ufrgs.br, at INF/UFRGS.

