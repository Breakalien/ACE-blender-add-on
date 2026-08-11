@echo off
cd /d "%~dp0"
python acevo_lite_editor.py %*
if errorlevel 1 pause
