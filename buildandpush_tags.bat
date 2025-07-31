@echo off
setlocal enabledelayedexpansion


:: ===== LOAD ENV FILE =====
set ENV_FILE=.env
if not exist %ENV_FILE% (
    echo ERROR: %ENV_FILE% not found.
    pause
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    set "%%A=%%B"
)

:: ==== CONFIGURATION ====
set IMAGE_NAME=tag-pop
set ARTIFACTORY_URL=art.t3.daimlertruck.com
set REPO_PATH=godap-docker-local
set FULL_IMAGE_NAME=%ARTIFACTORY_URL%/%REPO_PATH%/%IMAGE_NAME%:%VER%

:: ===== LOGIN =====
echo Logging in to Docker registry: %ARTIFACTORY_URL%
docker login %ARTIFACTORY_URL% -u %DOCKER_USERNAME% -p %DOCKER_PASSWORD%
if errorlevel 1 (
    echo Docker login failed.
    pause
    exit /b 1
)

:: ===== BUILD IMAGE =====
echo Building Docker image: %FULL_IMAGE_NAME%
docker build -f tags\tagdockerfile -t %FULL_IMAGE_NAME% .
if errorlevel 1 (
    echo Docker build failed.
    pause
    exit /b 1
)

:: ===== PUSH IMAGE =====
echo Pushing Docker image to Artifactory...
docker push %FULL_IMAGE_NAME%
if errorlevel 1 (
    echo Docker push failed.
    pause
    exit /b 1
)

echo Done!
echo Image pushed successfully: %FULL_IMAGE_NAME%
pause