@echo off
# assuming we have git installed 
# assuming we have conda installed 
#echo Downloading Miniforge installer for Windows (x86_64 assumed)
#powershell -Command "Invoke-WebRequest -Uri https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe -OutFile %temp%\miniforge.exe"

#echo Installing Miniforge to C:\Miniforge
#%temp%\miniforge.exe /S /D=C:\Miniforge
#setx PATH "C:\Miniforge\Library\bin;C:\Miniforge\Scripts;C:\Miniforge;%PATH%"

echo Creating conda environment imswitch311 and installing packages
call C:\Miniforge\Scripts\activate.bat
conda create -y --name imswitch311 python=3.11
conda install -y -n imswitch311 -c conda-forge h5py numcodecs scikit-image=0.19.3

echo Installing UV
call activate imswitch311
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

echo Cloning ImSwitchConfig if not present
if not exist "%USERPROFILE%\ImSwitchConfig" (
    git clone https://github.com/openUC2/ImSwitchConfig "%USERPROFILE%\ImSwitchConfig"
)

echo Cloning and installing ImSwitch
if not exist "%USERPROFILE%\ImSwitch" (
    git clone https://github.com/openUC2/imSwitch "%USERPROFILE%\ImSwitch"
)
call activate imswitch311
uv pip install -e "%USERPROFILE%\ImSwitch"

echo Installing UC2-REST
if not exist "%USERPROFILE%\UC2-REST" (
    git clone https://github.com/openUC2/UC2-REST "%USERPROFILE%\UC2-REST"
)
uv pip install -e "%USERPROFILE%\UC2-REST"

echo Installing specific versions of ome-zarr, numpy, and scikit-image
uv pip install ome-zarr==0.9.0
conda install -y -c conda-forge --strict-channel-priority numpy==1.26.4 scikit-image==0.19.3

echo Installation complete. 
echo To use ImSwitch, run:
echo call C:\Miniforge\Scripts\activate.bat imswitch311 && python %USERPROFILE%\ImSwitch\main.py --headless --http-port 8001