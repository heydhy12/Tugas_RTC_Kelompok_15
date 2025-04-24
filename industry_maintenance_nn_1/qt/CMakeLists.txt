cmake_minimum_required(VERSION 3.5)

project(IndustryMaintenancePredictor)

set(CMAKE_CXX_STANDARD 11)

# Enable auto features
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Set Qt path (adjust according to your installation)
set(CMAKE_PREFIX_PATH "/opt/Qt/5.15.2/gcc_64/lib/cmake")

# Find Qt5 Widgets
find_package(Qt5 REQUIRED COMPONENTS Widgets)

# Include directories
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(lib)

# Add executable
add_executable(IndustryMaintenancePredictor
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

# Link libraries
target_link_libraries(IndustryMaintenancePredictor
    ${Qt5Widgets_LIBRARIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/../target/release/libindustry_maintenance_nn.so
)

# Copy the Rust library after build
add_custom_command(TARGET IndustryMaintenancePredictor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/../target/release/libindustry_maintenance_nn.so
    ${CMAKE_CURRENT_BINARY_DIR}
)
