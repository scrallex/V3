# SEP Template CMake Functions
# Provides standardized library and executable creation

# Function to create a standard SEP library
function(add_sep_library name)
    set(options STATIC SHARED)
    set(oneValueArgs "")
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES CUDA_SOURCES)
    
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Default to STATIC if not specified
    if(NOT ARG_STATIC AND NOT ARG_SHARED)
        set(ARG_STATIC TRUE)
    endif()
    
    # Combine regular sources and CUDA sources
    set(ALL_SOURCES ${ARG_SOURCES} ${ARG_CUDA_SOURCES} ${ARG_HEADERS})
    
    # Create library
    if(ARG_STATIC)
        add_library(${name} STATIC ${ALL_SOURCES})
    else()
        add_library(${name} SHARED ${ALL_SOURCES})
    endif()
    
    # Set CUDA properties if CUDA sources are present
    if(ARG_CUDA_SOURCES)
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        
        # Enable CUDA language
        enable_language(CUDA)
    endif()
    
    # Set include directories
    target_include_directories(${name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    
    # Add dependencies
    if(ARG_DEPENDENCIES)
        target_link_libraries(${name} PUBLIC ${ARG_DEPENDENCIES})
    endif()
    
    # Set C standard
    target_compile_features(${name} PUBLIC cxx_std_17)
    
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
            /W4
            /O2
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
    )
    
    # Add dependencies
    if(ARG_DEPENDENCIES)
        target_link_libraries(${name} PRIVATE ${ARG_DEPENDENCIES})
    endif()
    
    # Set C standard
    target_compile_features(${name} PUBLIC cxx_std_17)
    
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
            /W4
            /O2
        )
    endif()
endfunction()
