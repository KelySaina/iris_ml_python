@echo off

REM Create and activate a virtual environment
python -m venv iris_ml_env
iris_ml_env\Scripts\activate

REM Install required Python packages
pip install -r requirements.txt