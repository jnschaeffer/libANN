cmake_minimum_required(VERSION 2.8)

#Set variables
SET(ALL_FILES ANN_layer.c ANN_ffnetwork.c)
SET(LIB_DIR lib)

#Compile the library
ADD_LIBRARY(ann SHARED ${ALL_FILES})
ADD_LIBRARY(ann_static STATIC ${ALL_FILES})

#Install!
TARGET_LINK_LIBRARIES(ann m)
TARGET_LINK_LIBRARIES(ann_static m)
INSTALL(TARGETS ann ann_static
LIBRARY DESTINATION ${LIB_DIR}
ARCHIVE DESTINATION ${LIB_DIR}
)
