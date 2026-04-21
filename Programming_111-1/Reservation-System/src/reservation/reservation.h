#ifndef RESERVATION_H_INCLUDED
#define RESERVATION_H_INCLUDED

#include <string>
#include <iostream>
using namespace std;
class reservation
{
    public:

        void setData(string, string);
        void setID(string);
        void setTeacher(string);
        string getID();
        string getTeacher();
        void printID();
        void printTeacher();


    private:
        string student_id;
        string teacher;

};
#endif // RESERVATION_H_INCLUDED
