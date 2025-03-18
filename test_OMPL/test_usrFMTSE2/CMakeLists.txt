option(OMPL_BUILD_DEMOS "Build OMPL demos" ON)
add_feature_info(OMPL_BUILD_DEMOS "${OMPL_BUILD_DEMOS}" "Whether to build the OMPL demo programs.")

find_package(OpenCV 4.0 REQUIRED)
message(STATUS "OpenCV is set to ${OpenCV_INCLUDE_DIRS}")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test_usrFMTdemo)
macro(add_ompl_demo demo_name)
    add_executable(${ARGV})
    target_link_libraries("${demo_name}"
        ompl
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})
endmacro(add_ompl_demo)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

# add_ompl_demo(testAIT testAIT.cpp)
add_ompl_demo(  testABIT_SE2 testABIT.cpp)
add_ompl_demo(   testBIT_SE2 testBIT.cpp)
add_ompl_demo(   testFMT_SE2 testFMT.cpp)
add_ompl_demo(   testRRT_SE2 testRRT.cpp)
add_ompl_demo(testinfRRT_SE2 testinfRRT.cpp)
add_ompl_demo(   testPRM_SE2 testPRM.cpp)



# file(GLOB OMPL_DEMO_PY_FILES "*.py")
# install_python(PROGRAMS ${OMPL_DEMO_PY_FILES}
#     DESTINATION "${OMPL_DEMO_INSTALL_DIR}"
#     COMPONENT ompl)
# file(GLOB OMPL_DEMO_CXX_FILES "*.cpp")
# install(FILES ${OMPL_DEMO_CXX_FILES}
#     DESTINATION "${OMPL_DEMO_INSTALL_DIR}"
#     COMPONENT ompl)
# install(DIRECTORY Koules
#     DESTINATION "${OMPL_DEMO_INSTALL_DIR}"
#     COMPONENT ompl)
# install(DIRECTORY VFRRT
#     DESTINATION "${OMPL_DEMO_INSTALL_DIR}"
#     COMPONENT ompl)
