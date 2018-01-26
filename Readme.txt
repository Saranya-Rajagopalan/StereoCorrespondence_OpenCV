Readme.txt
-------------------------------------------------------------------------------------
Aim: StereoCorrespondence using the traditional Sum of the Squared Differences 
algorithm between two linearly displaced images using features detected with SIFT 
and ORB
-------------------------------------------------------------------------------------
Copy the SIFT.cpp and ORB.cpp file to a project in Visual Studio.
-------------------------------------------------------------------------------------
Dependencies: Add library and include dependencies for the project as below:
C/C++ > Additional Include Libraries: Add "C:\OpenCV3.3\OpenCV\build\include"

Linker > General > Additional Library Directories: Add
"C:\OpenCV3.3\OpenCV\build\x64\vc14\lib"

Linker > Input > Additional Dependencies: Add opencv_world330d.lib

Confirm the configuration mode is Debug
-------------------------------------------------------------------------------------
Compile: Press Ctrl+F5 to build and run the program.
-------------------------------------------------------------------------------------
Sample input and output files are uploaded
