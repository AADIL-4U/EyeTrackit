@echo off
echo ==============================================
echo Starting Reading Difficulty Pipeline
echo ==============================================

echo [1/4] Running Data Loader (Downloading & Processing CSV)
python src\data_loader.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/4] Running Feature Engineering (Imputing & Scaling)
python src\feature_engineer.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/4] Running Model Training (Ridge and LSTM)
python src\model_trainer.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [4/4] Running Evaluator (Metrics, SHAP, and Final Report)
python src\evaluator.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==============================================
echo Pipeline execution successfully finished!
echo ==============================================
pause
