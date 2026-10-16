# A platform-independent identity for the native source and dependency pins.
# Reconfigure automatically when any input changes, including uncommitted edits.
file(GLOB_RECURSE feather_inputs CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}"
     "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.cc"
     "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt"
     "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/plugin/*.cc"
     "${PROJECT_SOURCE_DIR}/plugin/*.h" "${PROJECT_SOURCE_DIR}/plugin/CMakeLists.txt"
     "${PROJECT_SOURCE_DIR}/cmake/*.cmake" "${PROJECT_SOURCE_DIR}/cmake/*.patch")
list(APPEND feather_inputs CMakeLists.txt)
list(SORT feather_inputs)
set(feather_content "Feather native source identity v1\n")
foreach(input IN LISTS feather_inputs)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/${input}")
  file(SHA256 "${PROJECT_SOURCE_DIR}/${input}" digest)
  string(APPEND feather_content "${input}:${digest}\n")
endforeach()
string(SHA256 FEATHER_SOURCE_ID "${feather_content}")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/feather")
file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/feather/build_id.h"
     CONTENT "#define MJ_FEATHER_BUILD_ID \"@FEATHER_SOURCE_ID@\"\n" @ONLY)
target_include_directories(mujoco PRIVATE "${PROJECT_BINARY_DIR}/feather")
