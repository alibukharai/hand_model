idf_build_get_property(target IDF_TARGET)

set(srcs            app_main.cpp
                   
                    /home/ali/work/learning/hand_model/deploy/model/handrecognition_coefficient.cpp)

set(src_dirs        ../model)

set(include_dirs    ../model)

idf_component_register(SRCS ${srcs} SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})

set(lib     libdl.a)

if(${IDF_TARGET} STREQUAL "esp32")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/esp32")

elseif(${IDF_TARGET} STREQUAL "esp32s2")
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/esp32s2")

elseif(${IDF_TARGET} STREQUAL "esp32s3")
    set(links   "-L /home/ali/work/learning/colab_trained_model_with_results/deploy/components/esp-dl/lib/esp32s3")

elseif(${IDF_TARGET} STREQUAL "esp32c3")    
    set(links   "-L ${CMAKE_CURRENT_SOURCE_DIR}/../../../lib/esp32c3")

endif()


target_link_libraries(${COMPONENT_TARGET} ${links})
target_link_libraries(${COMPONENT_TARGET} ${lib})
