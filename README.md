# Factorization techniques for the analysis and separation of fetal ECG from maternal ECG

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc] <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a> 

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Installation

Create a virtual environment with [venv](https://docs.python.org/3/library/venv.html) and install all the requirements from `requirements.txt` file.

```bash
python -m venv .venv
python -m .venv\Scripts\Activate
python -m pip install -r requirements.txt
```

## Usage

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ecg_svd and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ecg_svd   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ecg_svd a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Contributing

## Authors and acknowledgment

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

For any commercial use or license requested for market activities, interested parties are invited to contact the Technology Transfer Office (TTO) of the University of Bari Aldo Moro, copyright holder. 

For a copy of the license, please visit https://creativecommons.org/licenses/by-nc/4.0/

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]
