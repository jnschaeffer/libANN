#CMakeLists file for the include directory.
#GPL goes here

cmake_minimum_required(VERSION 2.8)

#Set variables
SET(HEADER_FILES ANN_config.h ANN_main.h ANN_layer.h ANN_ffnetwork.h)
SET(INCLUDE_DIR include/ANN)

#Check to see whether the required header files are present
INCLUDE(${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
CHECK_INCLUDE_FILE("stdlib.h" HAVE_STDLIB_H)
CHECK_INCLUDE_FILE("stdio.h" HAVE_STDIO_H)
CHECK_INCLUDE_FILE("time.h" HAVE_TIME_H)
CHECK_INCLUDE_FILE("math.h" HAVE_MATH_H)

#Build ANN_config.h
CONFIGURE_FILE(ANN_config.h.in ANN_config.h)

#Install!
INSTALL(FILES ${HEADER_FILES} DESTINATION ${INCLUDE_DIR})
