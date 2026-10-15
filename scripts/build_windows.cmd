@echo off
rem Native Windows build and tests, from any terminal (cmd, PowerShell or
rem Git Bash: scripts/build_windows.cmd). Builds the CPU programs with
rem clang-cl and, with the CUDA toolkit installed, cxxgpt-gpu and test_gpu
rem (nvcc with cl.exe); then runs the kernel, engine and GPU tests.
rem   scripts\build_windows.cmd          (NO_GPU=1: CPU only)
rem Output: build\*.exe; build\configure.log and build\build.log.
setlocal
cd /d "%~dp0.." || exit /b 1

rem Visual Studio's C++ tools (MSVC, the Windows SDK, CMake's Ninja).
set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%vswhere%" (echo Visual Studio's C++ build tools are not installed: docs\LAPTOP.md & exit /b 1)
for /f "usebackq delims=" %%i in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vs=%%i"
if not defined vs (echo No Visual Studio with the C++ tools found: docs\LAPTOP.md & exit /b 1)
call "%vs%\VC\Auxiliary\Build\vcvars64.bat" > nul || exit /b 1

rem clang-cl: LLVM's installer, else Visual Studio's own, else the PATH.
set "clang=clang-cl"
if exist "%vs%\VC\Tools\Llvm\x64\bin\clang-cl.exe" set "clang=%vs%\VC\Tools\Llvm\x64\bin\clang-cl.exe"
if exist "%ProgramFiles%\LLVM\bin\clang-cl.exe" set "clang=%ProgramFiles%\LLVM\bin\clang-cl.exe"
set "clang=%clang:\=/%"

set "cuda=OFF"
if not "%NO_GPU%"=="1" where nvcc > nul 2>&1 && set "cuda=ON"
echo clang-cl: %clang%; CUDA build: %cuda%

if not exist build mkdir build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=%clang%" ^
  -DCXXGPT_MARCH=native -DCXXGPT_CUDA=%cuda% > build\configure.log 2>&1
if errorlevel 1 (type build\configure.log & exit /b 1)
cmake --build build --parallel > build\build.log 2>&1
if errorlevel 1 (findstr /i /c:"error" build\build.log & echo build failed: build\build.log & exit /b 1)
echo build ok

build\ternary_moe.exe --tests > build\ternary_moe.log 2>&1 || (type build\ternary_moe.log & exit /b 1)
echo kernel tests passed
build\test_engine.exe > build\test_engine.log 2>&1 || (findstr /c:"FAIL" build\test_engine.log & exit /b 1)
echo engine tests passed
if "%cuda%"=="ON" (
  build\test_gpu.exe > build\test_gpu.log 2>&1
  findstr /c:"FAIL" build\test_gpu.log
  findstr /c:"all passed" build\test_gpu.log > nul && echo GPU tests passed || echo GPU tests: see build\test_gpu.log ^(two known failures: docs\LAPTOP.md, GPU options^)
)
echo All set. Training: build\cxxgpt-gpu.exe train --gpu --help
