# SEP Template CMake Functions
# Provides standardized library and executable creation

# Function to create a standard SEP library
function(add_sep_library name)
    set(options STATIC SHARED)
    set(oneValueArgs PCH_HEADER)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Default to STATIC if not specified
    if(NOT ARG_STATIC AND NOT ARG_SHARED)
        set(ARG_STATIC TRUE)
    endif()
    
    # Create library
    if(ARG_STATIC)
        add_library(${name} STATIC ${ARG_SOURCES} ${ARG_HEADERS})
    else()
        add_library(${name} SHARED ${ARG_SOURCES} ${ARG_HEADERS})
    endif()
    
    # Set include directories
    target_include_directories(${name} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/sep>
    )
    
    # Add dependencies
    if(ARG_DEPENDENCIES)
        target_link_libraries(${name} PUBLIC ${ARG_DEPENDENCIES})
    endif()
    
    # Set C++20 standard (matching root CMakeLists.txt)
    target_compile_features(${name} PUBLIC cxx_std_20)

    if(ARG_PCH_HEADER)
        target_precompile_headers(${name} PUBLIC ${ARG_PCH_HEADER})
    endif()
    
    # Set position independent code for all libraries (needed for shared libraries)
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    
    # Common compile options
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -O3
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${name} PRIVATE
            /W3
            /O2
            /EHsc
            /std:c++20
            /bigobj
        )
    endif()
endfunction()

# Function to create a standard SEP executable
function(add_sep_executable name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES DEPENDENCIES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Create executable
    add_executable(${name} ${ARG_SOURCES})
    
    # Set include directories
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    
    # Link required core dependencies and any additional libraries
    target_link_libraries(${name} PRIVATE sep_core_deps sep_fetchcontent_deps)
    if(ARG_DEPENDENCIES)
        target_link_libraries(${name} PRIVATE ${ARG_DEPENDENCIES})
    endif()
    
    # Set C++20 standard (matching root CMakeLists.txt)
    target_compile_features(${name} PUBLIC cxx_std_20)
    
    # Common compile options
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -O3
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${name} PRIVATE
            /W3
            /O2
            /EHsc
            /std:c++20
            /bigobj
        )
    endif()
endfunction()
