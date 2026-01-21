cmake_minimum_required(VERSION 3.20)

# Fewer surprise warnings from third-party junk
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Suppress deprecation warnings" FORCE)

# Environment-driven compiler configuration
if(UNIX AND NOT APPLE)
  # Allow environment override of compiler paths
  set(SEP_GCC_PATH "$ENV{SEP_GCC_PATH}")
  if(NOT SEP_GCC_PATH)
    set(SEP_GCC_PATH "/usr/bin")
  endif()
  
  if(EXISTS "${SEP_GCC_PATH}/gcc-11")
    set(CMAKE_C_COMPILER        "${SEP_GCC_PATH}/gcc-11")
    set(CMAKE_CXX_COMPILER      "${SEP_GCC_PATH}/g++-11")
    message(STATUS "Using GCC-11 from ${SEP_GCC_PATH}")
  else()
    message(STATUS "GCC-11 not found at ${SEP_GCC_PATH}; using system compiler")
  endif()
endif()

# Workspace and module path
if(DEFINED ENV{SEP_WORKSPACE_PATH})
  set(SEP_WORKSPACE_PATH "$ENV{SEP_WORKSPACE_PATH}")
else()
  set(SEP_WORKSPACE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "SEP_WORKSPACE_PATH: ${SEP_WORKSPACE_PATH}")

# Compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)

project(sep_trading_tools VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

# Provide project root path to C++ sources
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Environment-driven directory configuration
if(DEFINED ENV{SEP_CACHE_DIR})
  set(CACHE_DIR "$ENV{SEP_CACHE_DIR}")
else()
  set(CACHE_DIR "${CMAKE_SOURCE_DIR}/cache/")
endif()

if(DEFINED ENV{SEP_CONFIG_DIR})
  set(CONFIG_DIR "$ENV{SEP_CONFIG_DIR}")
else()
  set(CONFIG_DIR "${CMAKE_SOURCE_DIR}/config/")
endif()

if(DEFINED ENV{SEP_LOG_DIR})
  set(LOG_DIR "$ENV{SEP_LOG_DIR}")
else()
  set(LOG_DIR "${CMAKE_SOURCE_DIR}/logs/")
endif()

message(STATUS "SEP Cache Directory: ${CACHE_DIR}")
message(STATUS "SEP Config Directory: ${CONFIG_DIR}")
message(STATUS "SEP Log Directory: ${LOG_DIR}")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Minimal, targeted defines
add_compile_definitions(
  _GLIBCXX_DEBUG_PEDANTIC=0
  _GLIBCXX_CONCEPT_CHECKS=0
)

add_compile_options(-Wall -Wextra -Wpedantic -O3 -Wno-error=stringop-overflow)

# If you actually use TBB in algorithms, this is the only PSTL hint you need
if(SEP_USE_TBB)
  add_compile_definitions(PSTL_USE_TBB)
endif()

# fmt/spdlog configuration: prefer header-only to avoid runtime .so deps
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use external fmt" FORCE)
set(SPDLOG_HEADER_ONLY ON CACHE BOOL "Use header-only spdlog" FORCE)
set(FMT_HEADER_ONLY ON CACHE BOOL "Use header-only fmt" FORCE)

include(cmake/template.cmake)

add_compile_definitions(NLOHMANN_JSON_USE_GLOBAL_UDLS=0)

option(BUILD_TESTING "Enable building of tests" ON)
include(CTest)
enable_testing()
if(BUILD_TESTING)
  include(cmake/gtest.cmake)
endif()

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)
find_package(benchmark QUIET)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# PostgreSQL (libpqxx + libpq)
if(WIN32)
  find_package(libpqxx CONFIG REQUIRED)
  find_package(unofficial-libpq CONFIG REQUIRED)
  add_library(PostgreSQL::pqxx ALIAS libpqxx::pqxx)
  add_library(PostgreSQL::pq   ALIAS unofficial::libpq::libpq)
  message(STATUS "PostgreSQL via vcpkg on Windows")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PQXX REQUIRED libpqxx)
  pkg_check_modules(PQ   REQUIRED libpq)

  add_library(PostgreSQL::pqxx INTERFACE IMPORTED)
  set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQXX_INCLUDE_DIRS}")
  set_property(TARGET PostgreSQL::pqxx PROPERTY INTERFACE_LINK_LIBRARIES "${PQXX_LIBRARIES}")

  add_library(PostgreSQL::pq INTERFACE IMPORTED)
  set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${PQ_INCLUDE_DIRS}")
  set_property(TARGET PostgreSQL::pq PROPERTY INTERFACE_LINK_LIBRARIES "${PQ_LIBRARIES}")

  message(STATUS "PostgreSQL via pkg-config on Linux")
endif()

include(FetchContent)

find_package(glm REQUIRED)

find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Handle both system package and FetchContent target names
if(TARGET nlohmann_json::nlohmann_json AND NOT TARGET nlohmann_json)
  add_library(nlohmann_json ALIAS nlohmann_json::nlohmann_json)
endif()

# ---------- yaml-cpp ----------
if(WIN32)
  find_package(yaml-cpp CONFIG REQUIRED)
  add_library(YAML::yaml-cpp ALIAS yaml-cpp)
  message(STATUS "yaml-cpp: vcpkg (Windows)")
else()
  find_package(yaml-cpp QUIET)
  if(NOT yaml-cpp_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
    add_library(YAML::yaml-cpp INTERFACE IMPORTED)
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIRS}")
    set_property(TARGET YAML::yaml-cpp PROPERTY INTERFACE_LINK_LIBRARIES     "${YAML_CPP_LIBRARIES}")
  else()
    # Handle different target names from system packages
    if(TARGET yaml-cpp AND NOT TARGET YAML::yaml-cpp)
      add_library(YAML::yaml-cpp ALIAS yaml-cpp)
    elseif(TARGET yaml-cpp::yaml-cpp AND NOT TARGET YAML::yaml-cpp)
      add_library(YAML::yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif()
  endif()
endif()

# ---------- hiredis (OPTIONAL) ----------
set(HIREDIS_FOUND FALSE)
if(WIN32)
  find_package(hiredis CONFIG QUIET)
  if(hiredis_FOUND)
    add_library(Redis::hiredis ALIAS hiredis::hiredis)
    set(HIREDIS_FOUND TRUE)
    message(STATUS "hiredis: vcpkg (Windows)")
  endif()
else()
  find_package(hiredis QUIET CONFIG)
  if(hiredis_FOUND)
    add_library(Redis::hiredis ALIAS hiredis::hiredis)
    set(HIREDIS_FOUND TRUE)
    message(STATUS "hiredis: CMake config (Linux)")
  else()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(HIREDIS QUIET hiredis)
      if(HIREDIS_FOUND)
        add_library(Redis::hiredis INTERFACE IMPORTED)
        set_property(TARGET Redis::hiredis PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${HIREDIS_INCLUDE_DIRS}")
        set_property(TARGET Redis::hiredis PROPERTY INTERFACE_LINK_LIBRARIES     "${HIREDIS_LIBRARIES}")
        message(STATUS "hiredis: pkg-config (Linux)")
      endif()
    endif()
  endif()
endif()

if(NOT HIREDIS_FOUND)
  message(STATUS "hiredis: NOT FOUND - Redis features will be disabled")
  add_compile_definitions(SEP_NO_REDIS=1)
endif()

# ---------- oneTBB (optional) ----------
option(SEP_USE_TBB "Enable TBB support" ON)
if(SEP_USE_TBB)
  find_package(TBB QUIET)
  if(NOT TBB_FOUND)
    include(FetchContent)
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests" FORCE)
    FetchContent_Declare(
      oneTBB
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2021.12.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(oneTBB)
    if(NOT TARGET TBB::tbb AND TARGET tbb)
      add_library(TBB::tbb ALIAS tbb)
    endif()
  endif()
endif()

# ---------- fmt / spdlog ----------
find_package(fmt 11 QUIET)
if(NOT fmt_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
  )
  # Set PIC for fmt to ensure it can be used in shared libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_MakeAvailable(fmt)
  set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

find_package(spdlog 1.15 QUIET)
if(NOT spdlog_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE
  )
  # Set PIC for spdlog to ensure it can be used in shared libraries
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_MakeAvailable(spdlog)
  set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

# ---------- Core dependency umbrella ----------
add_library(sep_core_deps INTERFACE)

foreach(_t
  Threads::Threads
  CURL::libcurl
  OpenSSL::SSL
  OpenSSL::Crypto
  ZLIB::ZLIB
  glm::glm
  PostgreSQL::pqxx
  PostgreSQL::pq
  YAML::yaml-cpp
  TBB::tbb
  benchmark::benchmark
  GTest::gtest
  GTest::gmock
)
  if(TARGET ${_t})
    target_link_libraries(sep_core_deps INTERFACE ${_t})
  endif()
endforeach()

# Optional Redis dependency
if(TARGET Redis::hiredis)
  target_link_libraries(sep_core_deps INTERFACE Redis::hiredis)
endif()

add_library(sep_fetchcontent_deps INTERFACE)
if(TARGET spdlog)
  get_target_property(_spdlog_inc spdlog INTERFACE_INCLUDE_DIRECTORIES)
  if(_spdlog_inc)
    target_include_directories(sep_fetchcontent_deps INTERFACE ${_spdlog_inc})
  endif()
endif()
if(TARGET fmt)
  get_target_property(_fmt_inc fmt INTERFACE_INCLUDE_DIRECTORIES)
  if(_fmt_inc)
    target_include_directories(sep_fetchcontent_deps INTERFACE ${_fmt_inc})
  endif()
endif()
target_link_libraries(sep_fetchcontent_deps INTERFACE
  $<$<TARGET_EXISTS:nlohmann_json::nlohmann_json>:nlohmann_json::nlohmann_json>
  $<$<TARGET_EXISTS:fmt::fmt-header-only>:fmt::fmt-header-only>
  $<$<TARGET_EXISTS:spdlog::spdlog_header_only>:spdlog::spdlog_header_only>
)

# ---------- JSON (header-only) ----------
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---------- cpr (HTTP client) ----------
find_package(cpr QUIET)
if(NOT cpr_FOUND)
  include(FetchContent)
  set(BUILD_CPR_TESTS OFF CACHE BOOL "" FORCE)
  set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.10.5
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cpr)
endif()

# Expose optional third-party targets via core deps umbrella if available
if(TARGET cpr::cpr)
  target_link_libraries(sep_core_deps INTERFACE cpr::cpr)
endif()
if(TARGET nlohmann_json::nlohmann_json)
  target_link_libraries(sep_core_deps INTERFACE nlohmann_json::nlohmann_json)
endif()

# ---------- cxxopts (CLI options) ----------
find_package(cxxopts QUIET)
if(NOT cxxopts_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(cxxopts)
endif()
if(TARGET cxxopts::cxxopts)
  target_link_libraries(sep_core_deps INTERFACE cxxopts::cxxopts)
endif()

# ---------- Kernel library (removed obsolete paths) ----------
# CUDA kernels are built within src/ as part of sep_lib when present.

# ---------- Tests & subdirs ----------
if(BUILD_TESTING)
#  add_subdirectory(tests)
endif()
add_subdirectory(src)

# ---------- Install ----------
# Note: This is a standalone application, not a library for external consumption
