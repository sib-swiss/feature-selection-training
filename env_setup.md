# Environment setup for the Feature Selection Course

> :exclamation: if you encounter any error with the instructions given on this page, please create a [github issue](https://github.com/sib-swiss/feature-selection-training/issues/new) to explain your problem and we will try to get back to you ASAP.


We detail in this page how to set up your environment with the different external modules you will need in order to be able to follow the course.

We recommend you create a new [uv project](https://docs.astral.sh/uv/getting-started/), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Nevertheless, we detail here several methods a trust you will choose the one most appropriate to your situations.

**important**: the course materials were developped and tested with **python >=3.11 and scikit-learn >=1.5**. Any anterior version will give errors and warnings aplenty!


## requirements:

Python : at least 3.12

 * [scikit-learn](https://scikit-learn.org/): `pip install scikit-learn` | `uv add scikit-learn` | `conda install -c conda-forge scikit-learn`
 * [seaborn](https://seaborn.pydata.org/): `pip install seaborn` | `uv add seaborn` | `conda install seaborn -c conda-forge`
 * [pandas](https://pandas.pydata.org/): `pip install pandas` | `uv add pandas` | `conda install -c conda-forge pandas`

 * [shap](https://shap.readthedocs.io/en/latest/): `pip install shap` | `uv add shap` | `conda install -c conda-forge shap`
 * [boruta](https://github.com/scikit-learn-contrib/boruta_py): `pip install Boruta` | `uv add Boruta` | `conda install -c conda-forge boruta_py`
 * [knockpy](https://amspector100.github.io/knockpy/) [in case of problem](https://amspector100.github.io/knockpy/installation.html#installation): `pip install knockpy[fast]` | `uv add knockpy[fast]` | no conda recipe

