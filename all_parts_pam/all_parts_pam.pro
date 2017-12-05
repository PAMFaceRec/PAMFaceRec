TEMPLATE = lib
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH +="/usr/locale/include/"

LIBS += `pkg-config --libs opencv`
