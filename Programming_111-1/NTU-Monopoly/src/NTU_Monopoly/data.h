#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED
#include "Player.h"
#include "Property.h"
#include "block.h"
#include "corner.h"
#include "chanceandchest.h"
#include "event.h"
player P1(25000, "P1");
player P2(25000, "P2");
player P3(25000, "P3");
player P4(25000, "P4");
corner gongguan (0, "MRT Gongguan Station", 5000);
property geo (1, "Dept of Geoscience", 3200, 2500, 2500);
property math(2, "Astro-Math Bldg", 4300, 2500, 2500);
community_chest chestblock1(3);
property psysics(4, "Dept of Physics", 2800, 2500, 2500);
event tuition (5, "Pay Tuition Fees", -1200);
chance chanceblock1(6);
property cosmology(7, "CCL Cosmology Hall", 4100, 2500, 2500);
chance chanceblock2(8);
property shih (9, "Shih-Liang Hall", 2300, 2500, 2500);
corner uncle (10, "Shuiyuan Uncle", -800, -1,true);
property socialscience (11, "Social Science Bldg", 8000, 2500, 2500);
event dorm (12, "Pay Dormitory Fee", -1000);
property manage2 (13, "Management Bldg 2", 4500, 2500, 2500);
chance chanceblock3(14);
community_chest chestblock2(15);
property manage1 (16, "Management Bldg 1", 4000, 2500, 2500);
event weak (17, "Pretend Weak", -1000);
property dev (18, "Grad Inst of Dev Studies", 4300, 2500, 2500);
property social (19, "Dept of Social", 4000, 2500, 2500);
corner alley (20, "118 Alley", -600, -1,true);
property mk (21, "MK Innovation Hall", 6000, 2500, 2500);
chance chanceblock4(22);
property ee2 (23, "EE Bldg No.2", 5000, 2500, 2500);
property barrylam (24, "Barry Lam Hall", 7000, 2500, 2500);
community_chest chestblock3(25);
property dertian (26, "Der Tian Hall", 6200, 2500, 2500);
property mingda (27, "Ming Da Hall", 5400, 2500, 2500);
event student (28, "Pay Student Assoc Fees", -1500);
property ee1 (29, "EE Bldg No.1", 4200, 2500, 2500);
corner gym (30, "Gym Center", -600, -1,true);
property zonghe (31, "Zonghe Lecture Bldg", 3000, 2500, 2500);
property xingsheng (32, "Xinsheng Lecture Bldg", 3400, 2500, 2500);
event towaway(33, "Bicycle Towed Away", -800, 10, true);
property putong (34, "Putong Lecture Bldg", 3500, 2500, 2500);
chance chanceblock5(35);
property boya (36, "Boya Lecture Bldg", 3300, 2500, 2500);
property library (37, "Main Library", 6000, 2500, 2500);
community_chest chestblock4(38);
property gongtong (39, "Gongtong Lecture Bldg", 3100, 2500, 2500);
#endif // DATA_H_INCLUDED
