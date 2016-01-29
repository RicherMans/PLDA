find_program(PYTHON "python")



if (PYTHON)
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS        ${CMAKE_CURRENT_SOURCE_DIR}/liblda/__init__.py)
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    #add_custom_command(INSTALL_PY ${OUTPUT}
        #COMMAND ${PYTHON} ${SETUP_PY} install --user)

    if(USER)
        message("User mode is on, please not that other uses may not be able to import the library")
        install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user )")
        add_custom_command(OUTPUT ${OUTPUT}
            COMMAND ${PYTHON} ${SETUP_PY} install --user
            COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
            DEPENDS ${DEPS})
    else()
        message("User mode is off, if installation is unsucessful please pass -DUSER=ON")
        install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install )")
        add_custom_command(OUTPUT ${OUTPUT}
            COMMAND ${PYTHON} ${SETUP_PY} install
            COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
            DEPENDS ${DEPS})
    endif()
    add_custom_target(target ALL DEPENDS ${OUTPUT})

endif()
