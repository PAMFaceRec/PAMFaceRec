TEMPLATE = lib
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    ImagePreprocessing.cpp

INCLUDEPATH +="/usr/locale/include/"

LIBS += `pkg-config --libs opencv`

HEADERS += \
    ImagePreprocessing.h
