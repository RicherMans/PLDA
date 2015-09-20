# - Try to find Kaldi
# Once done this will define
#
#  KALDI_FOUND      - system has Kaldi
#  KALDI_SRC_DIR    - Kaldi src directory
#  KALDI_TOOLS_DIR  - Kaldi tools directory
#  KALDI_LIBRARIES  - link these to use Kaldi
#  KALDI_VERSION    - Kaldi revision number

if (KALDI_SRC_DIR AND KALDI_TOOLS_DIR AND KALDI_LIBRARIES)
  set(KALDI_FOUND TRUE)
else (KALDI_SRC_DIR AND KALDI_TOOLS_DIR AND KALDI_LIBRARIES)

  find_path(KALDI_SRC_DIR NAMES base/kaldi-common.h 
            PATHS ${KALDI_ROOT}/src NO_DEFAULT_PATH)
  find_path(KALDI_TOOLS_DIR NAMES install_portaudio.sh 
            PATHS ${KALDI_ROOT}/tools NO_DEFAULT_PATH)
  
  if (KALDI_SRC_DIR AND KALDI_TOOLS_DIR)
    if (NOT Kaldi_FIND_QUIETLY)
      message (STATUS "Kaldi root: ${KALDI_ROOT}")
    endif (NOT Kaldi_FIND_QUIETLY)
    
    if(IS_DIRECTORY ${KALDI_ROOT}/.svn)
      execute_process(COMMAND svnversion -n ${KALDI_ROOT}
                      OUTPUT_VARIABLE KALDI_VERSION)
      message(STATUS "Kaldi revision: ${KALDI_VERSION}")
    else ()
      message("Kaldi root is not an svn checkout. Kaldi revision unknown.")
    endif ()
    
  else (KALDI_SRC_DIR AND KALDI_TOOLS_DIR)
    message ("Could not find Kaldi root at ${KALDI_ROOT}")
  endif (KALDI_SRC_DIR AND KALDI_TOOLS_DIR)
  
  if (KALDI_SRC_DIR)
    
    find_library(KALDI_BASE_LIBRARY      NAMES base/kaldi-base.a 
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_DECODER_LIBRARY   NAMES decoder/kaldi-decoder.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_FEAT_LIBRARY      NAMES feat/kaldi-feat.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_GMM_LIBRARY       NAMES gmm/kaldi-gmm.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_HMM_LIBRARY       NAMES hmm/kaldi-hmm.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_LAT_LIBRARY       NAMES lat/kaldi-lat.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_MATRIX_LIBRARY    NAMES matrix/kaldi-matrix.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_TRANSFORM_LIBRARY NAMES transform/kaldi-transform.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_TREE_LIBRARY      NAMES tree/kaldi-tree.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_UTIL_LIBRARY      NAMES util/kaldi-util.a
                 PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    find_library(KALDI_IVECTOR_LIBRARY      NAMES ivector/kaldi-ivector.a
                PATHS ${KALDI_SRC_DIR} NO_DEFAULT_PATH)
    
    foreach(LIBNAME KALDI_DECODER_LIBRARY KALDI_FEAT_LIBRARY KALDI_GMM_LIBRARY KALDI_HMM_LIBRARY KALDI_LAT_LIBRARY KALDI_MATRIX_LIBRARY KALDI_TRANSFORM_LIBRARY KALDI_TREE_LIBRARY KALDI_UTIL_LIBRARY KALDI_BASE_LIBRARY KALDI_IVECTOR_LIBRARY)
      if(${LIBNAME})
        SET(KALDI_LIBRARIES ${KALDI_LIBRARIES} ${${LIBNAME}} )
      else(${LIBNAME})
        message("${LIBNAME} not found.")
        set(KALDI_MISSING_LIBRARY TRUE)
      endif (${LIBNAME})
    endforeach(LIBNAME)
    
    if (KALDI_LIBRARIES)
      if (NOT Kaldi_FIND_QUIETLY)
        message (STATUS "Kaldi libraries: ${KALDI_LIBRARIES}")
      endif (NOT Kaldi_FIND_QUIETLY)
    endif (KALDI_LIBRARIES)
    
  endif(KALDI_SRC_DIR)
  
  if (KALDI_TOOLS_DIR)
    
    find_path(OPENFST_INCLUDE_DIR NAMES fst/fstlib.h
              PATHS ${KALDI_TOOLS_DIR}/openfst/include NO_DEFAULT_PATH)
    if (OPENFST_INCLUDE_DIR)
      if (NOT Kaldi_FIND_QUIETLY)
        message(STATUS "OpenFst include: ${OPENFST_INCLUDE_DIR}")
      endif (NOT Kaldi_FIND_QUIETLY)
    else (OPENFST_INCLUDE_DIR)
      message ("Openfst header files not found at ${KALDI_TOOLS_DIR}/openfst/include")
    endif (OPENFST_INCLUDE_DIR)
    
    find_library(OPENFST_LIBRARY NAMES libfst.a 
                 PATHS ${KALDI_TOOLS_DIR}/openfst/lib NO_DEFAULT_PATH)
    if (OPENFST_LIBRARY)
      if (NOT Kaldi_FIND_QUIETLY)
        message(STATUS "OpenFst library: ${OPENFST_LIBRARY}")
      endif (NOT Kaldi_FIND_QUIETLY)
    else (OPENFST_LIBRARY)
      message ("Openfst library not found at ${KALDI_TOOLS_DIR}/openfst/lib")
    endif (OPENFST_LIBRARY)
    
    
    if (OPENFST_INCLUDE_DIR AND OPENFST_LIBRARY)
      set(KALDI_DEPENDENCIES_FOUND TRUE)
    endif()
    
  endif (KALDI_TOOLS_DIR)
  
  if (KALDI_SRC_DIR AND KALDI_TOOLS_DIR AND NOT KALDI_MISSING_LIBRARY
      AND KALDI_DEPENDENCIES_FOUND)
    set(KALDI_FOUND TRUE)
  endif ()

endif (KALDI_SRC_DIR AND KALDI_TOOLS_DIR AND KALDI_LIBRARIES)

if(Kaldi_FIND_REQUIRED AND NOT KALDI_SRC_DIR AND NOT KALDI_TOOLS_DIR)
  message(FATAL_ERROR "Could not find Kaldi.")
endif()

if(Kaldi_FIND_REQUIRED AND KALDI_MISSING_LIBRARY)
  message(FATAL_ERROR "Could not find some of the required Kaldi libraries.")
endif()

if(Kaldi_FIND_REQUIRED AND NOT KALDI_DEPENDENCIES_FOUND)
  message(FATAL_ERROR "Could not find some of the required Kaldi dependencies.")
endif()
