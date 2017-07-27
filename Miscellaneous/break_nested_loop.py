#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:11:38 2017

@author: mmrosek
"""

for i in range(10):
    
    for j in range(20):
        
        print(i*j)
        
        if i * j == 15:
            
            return()
        
for x in range(10):
    print('XXXXXXXXXXXXXXx')
    for y in range(10):
        print(x*y)
        if x*y > 50:
            break
    else:
        continue  # executed if the loop ended normally (no break)
    break  # executed if 'continue' was skipped (break)