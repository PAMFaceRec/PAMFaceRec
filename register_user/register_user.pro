#-------------------------------------------------
#
# Project created by QtCreator 2017-11-27T21:55:35
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = register_user
TEMPLATE = app

CONFIG += console c++11


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH +="/usr/locale/include/"

LIBS += `pkg-config --libs opencv`
