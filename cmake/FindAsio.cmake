# FindAsio.cmake - Find standalone asio library
# 
# This module defines:
#  ASIO_FOUND - True if asio is found
#  ASIO_INCLUDE_DIRS - Include directories for asio
#  asio::asio - Imported target for asio

find_path(ASIO_INCLUDE_DIR
    NAMES asio.hpp
    PATHS
        /usr/include
        /usr/local/include
        /opt/local/include
        ${ASIO_ROOT}/include
    PATH_SUFFIXES
        asio
    NO_DEFAULT_PATH
)

find_path(ASIO_INCLUDE_DIR
    NAMES asio.hpp
    PATH_SUFFIXES
        asio
)

if(ASIO_INCLUDE_DIR)
    # Check version if asio/version.hpp exists
    set(ASIO_VERSION_FILE "${ASIO_INCLUDE_DIR}/asio/version.hpp")
    if(EXISTS "${ASIO_VERSION_FILE}")
        file(STRINGS "${ASIO_VERSION_FILE}" ASIO_VERSION_STRING
            REGEX "^#define[ \t]+ASIO_VERSION[ \t]+[0-9]+")
        if(ASIO_VERSION_STRING)
            string(REGEX REPLACE "^#define[ \t]+ASIO_VERSION[ \t]+([0-9]+).*" "\\1"
                ASIO_VERSION_INT "${ASIO_VERSION_STRING}")
            math(EXPR ASIO_VERSION_MAJOR "${ASIO_VERSION_INT} / 100000")
            math(EXPR ASIO_VERSION_MINOR "(${ASIO_VERSION_INT} / 100) % 1000")
            math(EXPR ASIO_VERSION_PATCH "${ASIO_VERSION_INT} % 100")
            set(ASIO_VERSION "${ASIO_VERSION_MAJOR}.${ASIO_VERSION_MINOR}.${ASIO_VERSION_PATCH}")
        endif()
    endif()
    
    set(ASIO_INCLUDE_DIRS ${ASIO_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(asio
    REQUIRED_VARS ASIO_INCLUDE_DIRS
    VERSION_VAR ASIO_VERSION
)

if(asio_FOUND AND NOT TARGET asio::asio)
    add_library(asio::asio INTERFACE IMPORTED)
    set_target_properties(asio::asio PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ASIO_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "ASIO_STANDALONE;ASIO_HAS_STD_CHRONO"
    )
endif()

mark_as_advanced(ASIO_INCLUDE_DIR)
