#include <iostream>
#include <string>
#include <iomanip>
#include "reservation.h"
using namespace std;


void reservation::setData(string s1, string s2)
{
    setID(s1);
    setTeacher(s2);
}

void reservation::setID(string s1)
{
        student_id = ((s1[0]=='B' || s1[0]=='b') &&
                      s1[1]=='1' && s1[2]=='1' &&
                      s1[3]=='9' && s1[4]=='0' && s1[5]=='1')? s1 : "0";
}

void reservation::setTeacher(string s2)
{
    teacher = (s2=="Wei" || s2=="wei" || s2=="Hsieh" || s2=="hsieh"
               || s2=="Lin" || s2=="lin")? s2 : "0";
}

string reservation::getID()
{
    return student_id;
}

string reservation::getTeacher()
{
    return teacher;
}

void reservation::printID()
{
    cout << setw(15) << student_id;
}

void reservation::printTeacher()
{
    cout << "    " << teacher;
}
