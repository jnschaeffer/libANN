#libANN verion 0.1
#GPL goes here

cmake_minimum_required(VERSION 2.8)

#Project name
PROJECT(libANN)

#Set variables
SET(CMAKE_INSTALL_PREFIX /usr)

#CMake minimum required version
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#Include all of the header files
INCLUDE_DIRECTORIES(include)

#Run cmake in the following directories
ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(src)
