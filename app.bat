@echo OFF

::指定conda安装目录
set CONDAPATH=C:\Miniconda3
::指定conda虚拟环境名称
set ENVNAME=py3

::激活虚拟环境
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

::进入app.bat文件所在的工作空间
cd /d %~dp0

::调用脚本
python app.py %1

::取消激活conda环境
call conda deactivate