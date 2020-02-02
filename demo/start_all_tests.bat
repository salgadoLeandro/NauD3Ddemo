@echo off

FOR /F %%I in ('powershell -command "& {Get-ExecutionPolicy}"') do SET var=%%I

powershell -command "& {Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Force}"
FOR /F %%I in ('powershell -command "& {Get-ExecutionPolicy}"') do SET vat=%%I

echo BEGIN
powershell -command "%~dp0runall.ps1 -file .\runall.txt"
echo END

powershell -command "& {Set-ExecutionPolicy -ExecutionPolicy %var% -Force}"
FOR /F %%I in ('powershell -command "& {Get-ExecutionPolicy}"') do SET vat=%%I

pause
