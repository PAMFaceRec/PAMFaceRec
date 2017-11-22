#ifndef TESTCLASS_H
#define TESTCLASS_H

#include <iostream>
using namespace std;

class testClass
{
public:
    testClass() : a(1), b(2) {}
    void print() const {
       std::cout << this->a << std::endl;
       std::cout << this->b << std::endl;
    }
private:
    int a;
    int b;
};

#endif // TESTCLASS_H
