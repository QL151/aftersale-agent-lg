@echo off
rem ---------------------------------------------------------------
rem  Do NOT put Chinese text in this file.
rem  cmd parses .bat with the system OEM codepage (GBK on zh-CN),
rem  so UTF-8 Chinese here would be mangled and break the commands.
rem  All Chinese UI lives in launcher.py instead.
rem ---------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"
title E-commerce After-sales Agent

rem Prefer the project-local virtualenv; fall back to python on PATH.
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

"%PY%" launcher.py
if errorlevel 1 pause
