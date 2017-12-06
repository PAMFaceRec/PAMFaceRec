#-------------------------------------------------
#
# Project created by QtCreator 2017-12-04T00:08:34
#
#-------------------------------------------------

QT       += core gui
QT += concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = all_parts_gui
TEMPLATE = app

CONFIG += console c++11

SOURCES += main.cpp\
        mainwindow.cpp \
    LoggingCategories.cpp \
    ImagePreprocessing.cpp

HEADERS  += mainwindow.h \
    LoggingCategories.h \
    ImagePreprocessing.h

FORMS    += mainwindow.ui

INCLUDEPATH +="/usr/locale/include/"

LIBS += `pkg-config --libs opencv`
