cmake_minimum_required(VERSION 3.5)
project(taylor_gui)

# Konfigurasi dasar
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# 1. Temukan Qt5 (dengan komponen yang diperlukan)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core)

# 2. Set path absolut ke library Rust
set(RUST_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release")
message(STATUS "Mencari library Rust di: ${RUST_LIB_DIR}")

# 3. Verifikasi file library ada
if(NOT EXISTS "${RUST_LIB_DIR}/libtaylor_lookup.so")
    message(WARNING "Library Rust tidak ditemukan di: ${RUST_LIB_DIR}")
endif()

# 4. Link directory
link_directories(${RUST_LIB_DIR})

# 5. File sumber
set(SOURCE_FILES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# 6. Buat executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 7. Link library
target_link_libraries(${PROJECT_NAME}
    Qt5::Widgets
    Qt5::Core
    taylor_lookup  # Harus sama persis dengan nama library Rust
)