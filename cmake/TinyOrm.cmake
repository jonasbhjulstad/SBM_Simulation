
if (NOT TARGET TinyOrm::TinyOrm)
find_package(QT NAMES Qt6 COMPONENTS Core REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core REQUIRED)
# find_package(TinyOrm 0.36.3 CONFIG REQUIRED HINTS "/usr/local/lib/cmake/TinyOrm/")
CPMAddPackage("gh:silverqx/TinyORM@0.36.5")
set(TINY_ORM_LIBRARIES Qt${QT_VERSION_MAJOR}::Core TinyOrm::TinyOrm)
endif()