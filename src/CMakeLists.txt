cmake_minimum_required(VERSION 3.18)
project(sep_trader LANGUAGES CXX C)

add_definitions(-DGLM_COMPILER=0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch spdlog
include(FetchContent)

set(SPDLOG_BUILD_SHARED FALSE)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    FIND_PACKAGE_ARGS
)

FetchContent_MakeAvailable(spdlog)

# Gather C++ source files only from existing directories
file(GLOB_RECURSE CXX_SOURCES
    "core/*.cpp"
    "app/*.cpp"
)

# No C source files are currently needed - all utilities use C++/spdlog
set(C_SOURCES)

# Exclude facade_original.cpp only (OANDA files are needed for frontend)
list(FILTER CXX_SOURCES EXCLUDE REGEX ".*/facade_original\\.cpp$")
# QBSA quantum algorithm now enabled - core functionality for SEP system
# list(FILTER CXX_SOURCES EXCLUDE REGEX ".*/core/qbsa\\.cpp$")

# Separate main files from library source files
set(LIB_CXX_SOURCES)
foreach(file ${CXX_SOURCES})
    if(NOT file MATCHES "main.cpp$")
        list(APPEND LIB_CXX_SOURCES ${file})
    endif()
endforeach()

# Create C++ library with PCH (no C sources needed)
add_sep_library(sep_lib SOURCES ${LIB_CXX_SOURCES})
# Link core deps privately to avoid leaking shared deps to dependents
target_link_libraries(sep_lib
    PRIVATE
        sep_core_deps
        sep_fetchcontent_deps
)
# Note: Include directories are handled by add_sep_library template with proper generator expressions
# Note: sep_fetchcontent_deps is PRIVATE - available for building sep_lib but not part of export interface

# Collect public headers for installation from existing directories
file(GLOB_RECURSE SEP_LIB_PUBLIC_HEADERS
    "app/*.h" "app/*.hpp" "app/*.cuh"
    "core/*.h" "core/*.hpp" "core/*.cuh"
)
set_target_properties(sep_lib PROPERTIES PUBLIC_HEADER "${SEP_LIB_PUBLIC_HEADERS}")

install(TARGETS sep_lib
    PUBLIC_HEADER DESTINATION include/sep
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Shared library exposing system quantum metrics
add_sep_library(quantum_metrics SHARED
    SOURCES app/quantum_metrics_c_api.cpp
)
target_link_libraries(quantum_metrics PRIVATE sep_lib)
# Ensure runtime can locate fmt/spdlog without LD_LIBRARY_PATH
set_target_properties(quantum_metrics PROPERTIES
    INSTALL_RPATH "${SEP_RUNTIME_LIBPATH}:$ORIGIN/../lib"
    BUILD_RPATH "${SEP_RUNTIME_LIBPATH};${CMAKE_BINARY_DIR}/_deps/spdlog-build;${CMAKE_BINARY_DIR}/_deps/fmt-build"
)

target_link_libraries(quantum_metrics PRIVATE spdlog::spdlog)

# Set RPATH for TBB library path resolution - environment-driven configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Allow environment override for library path
if(DEFINED ENV{SEP_LIB_PATH})
  set(SEP_RUNTIME_LIBPATH "$ENV{SEP_LIB_PATH}")
else()
  set(SEP_RUNTIME_LIBPATH "${CMAKE_BINARY_DIR}/lib")
endif()

# Set RPATH for installed binaries
set(CMAKE_INSTALL_RPATH "${SEP_RUNTIME_LIBPATH}:$ORIGIN/../lib")
message(STATUS "SEP Runtime Library Path: ${SEP_RUNTIME_LIBPATH}")

# Stage built artifacts into ${CMAKE_SOURCE_DIR}/bin for packaging/dev
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
add_custom_command(TARGET quantum_metrics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:quantum_metrics> ${CMAKE_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:quantum_metrics>
    COMMENT "Staging libquantum_metrics.so to bin/"
)

# (Install rules moved to end of file to ensure targets exist)

find_package(OpenSSL)
if(OpenSSL_FOUND AND TARGET OpenSSL::Crypto)
    # Data downloader utility (point directly to main)
    add_sep_executable(data_downloader SOURCES app/data_downloader_main.cpp)
    target_link_libraries(data_downloader PRIVATE sep_lib sep_core_deps sep_fetchcontent_deps OpenSSL::Crypto)
    set_target_properties(data_downloader PROPERTIES
        INSTALL_RPATH "${SEP_RUNTIME_LIBPATH}:$ORIGIN/../lib"
        BUILD_RPATH "${SEP_RUNTIME_LIBPATH};${CMAKE_BINARY_DIR}/_deps/spdlog-build;${CMAKE_BINARY_DIR}/_deps/fmt-build}")
    add_custom_command(TARGET data_downloader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:data_downloader> ${CMAKE_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:data_downloader>
        COMMENT "Staging data_downloader to bin/"
    )
else()
    message(WARNING "OpenSSL not found; skipping data_downloader target (manifold_generator unaffected)")
endif()

# Manifold generator - Enhanced Market Manifold Node processing kernel
add_sep_executable(manifold_generator SOURCES app/manifold_generator_main.cpp)
target_link_libraries(manifold_generator PRIVATE sep_lib sep_core_deps sep_fetchcontent_deps)
set_target_properties(manifold_generator PROPERTIES
    INSTALL_RPATH "${SEP_RUNTIME_LIBPATH}:$ORIGIN/../lib"
    BUILD_RPATH "${SEP_RUNTIME_LIBPATH};${CMAKE_BINARY_DIR}/_deps/spdlog-build;${CMAKE_BINARY_DIR}/_deps/fmt-build}")
add_custom_command(TARGET manifold_generator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:manifold_generator> ${CMAKE_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:manifold_generator>
    COMMENT "Staging manifold_generator to bin/"
)

# Byte stream manifold analyser for arbitrary payloads
add_sep_executable(byte_stream_manifold SOURCES app/byte_stream_manifold_main.cpp)
target_link_libraries(byte_stream_manifold PRIVATE sep_lib sep_core_deps sep_fetchcontent_deps)
set_target_properties(byte_stream_manifold PROPERTIES
    INSTALL_RPATH "${SEP_RUNTIME_LIBPATH}:$ORIGIN/../lib"
    BUILD_RPATH "${SEP_RUNTIME_LIBPATH};${CMAKE_BINARY_DIR}/_deps/spdlog-build;${CMAKE_BINARY_DIR}/_deps/fmt-build}")
add_custom_command(TARGET byte_stream_manifold POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:byte_stream_manifold> ${CMAKE_SOURCE_DIR}/bin/$<TARGET_FILE_NAME:byte_stream_manifold>
    COMMENT "Staging byte_stream_manifold to bin/"
)

# Context tokenizer and context manifold tools were moved to sep-context-engine.
# Remove their build targets from this repository to keep only essential binaries.
