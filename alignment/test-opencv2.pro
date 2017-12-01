TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    test-face_detection.cpp

INCLUDEPATH +="/usr/locale/include/"

LIBS += `pkg-config --libs opencv`
