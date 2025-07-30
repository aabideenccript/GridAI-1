@echo off
echo Setting up GridAI locally...

REM Create .env file
echo OPENAI_API_KEY=your_openai_api_key_here > .env

REM Create .streamlit directory and secrets
mkdir .streamlit 2>nul
echo OPENAI_API_KEY = "your_openai_api_key_here" > .streamlit\secrets.toml

REM Install dependencies
pip install -r requirements.txt

echo.
echo Setup complete! 
echo 1. Edit .env and add your OpenAI API key
echo 2. Edit .streamlit\secrets.toml and add your OpenAI API key  
echo 3. Run: streamlit run frontend.py
pause