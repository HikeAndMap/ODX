set(_proj_name mvstexturing)
set(_SB_BINARY_DIR "${SB_BINARY_DIR}/${_proj_name}")

ExternalProject_Add(${_proj_name}
  DEPENDS           mve
  PREFIX            ${_SB_BINARY_DIR}
  TMP_DIR           ${_SB_BINARY_DIR}/tmp
  STAMP_DIR         ${_SB_BINARY_DIR}/stamp
  #--Download step--------------
  DOWNLOAD_DIR      ${SB_DOWNLOAD_DIR}/${_proj_name}
  GIT_REPOSITORY    https://github.com/WebODM/mvs-texturing
  GIT_TAG           a56b5e7f95f0bb1dd1e1eb89ce5da202349c0e01
  #--Update/Patch step----------
  UPDATE_COMMAND    ""
  # PATCH4: --forced_views; restore the files first so the patch also applies when the step runs again
  PATCH_COMMAND     git checkout -- apps/texrecon/arguments.cpp libs/tex/calculate_data_costs.cpp libs/tex/settings.h libs/tex/texture_view.h
            COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/../patches/mvs-texturing-forced-views.patch
  #--Configure step-------------
  SOURCE_DIR        ${SB_SOURCE_DIR}/${_proj_name}
  CMAKE_ARGS
    -DRESEARCH=OFF
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${SB_INSTALL_DIR}
    ${WIN32_CMAKE_ARGS}
    ${APPLE_CMAKE_ARGS}
  #--Build step-----------------
  BINARY_DIR        ${_SB_BINARY_DIR}
  #--Install step---------------
  INSTALL_DIR       ${SB_INSTALL_DIR}
  #--Output logging-------------
  LOG_DOWNLOAD      OFF
  LOG_CONFIGURE     OFF
  LOG_BUILD         OFF
)
