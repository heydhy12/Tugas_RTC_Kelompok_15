cmake_minimum_required(VERSION 3.5)

project(TaylorQtApp)

# ✅ Auto-generate UI, MOC, dan QRC
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# ✅ Standar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ✅ Path Qt bawaan Linux Qt Installer (ubah kalau beda)
set(CMAKE_PREFIX_PATH "/opt/Qt/5.15.2/gcc_64/lib/cmake")

# ✅ Temukan Qt5
find_package(Qt5 REQUIRED COMPONENTS Widgets)

# ✅ Include lib Rust
include_directoriSes(lib)

# ✅ Buat executable
add_executable(TaylorQtApp
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# ✅ Link Qt dan .so Rust
target_link_libraries(TaylorQtApp
    Qt5::Widgets
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/librust_taylor.so
    dl
)
