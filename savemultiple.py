# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:46:29 2019

@author: Srivatsan
"""

import time
import zbar
cv2.namedWindow("camera", 1)

capture = cv2.VideoCapture(0)

i = 0
while True:
    set_zbar=zbar.ImageScanner()
    cv2.imshow("camera", img)
    cv2.imwrite('pic{:>05}.jpg'.format(i), img)
    if cv2.WaitKey(10) == 27:
        break
    i += 1