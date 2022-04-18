import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#baca citra sebagai BGR
bgr = cv.imread('gagak.jpg')

#menetapkan lower threshold dan upper threshold
lower_white = np.array([0,0,140])
higher_white = np.array([255,255,255])

#masking dan operasi bitwise And
mask = cv.inRange(bgr, lower_white, higher_white)
res = cv.bitwise_and(bgr,bgr, mask= mask)

#deklarasi Structuring Element ukuran 3x3 dan 5x5
se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

#Operasi morfologi
dst_dilate = cv.dilate(mask, se_3, iterations = 1)
dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)

#plotting hasil morfologi
plt.subplot(321),plt.imshow(dst_dilate, cmap = 'gray')
plt.title('Hasil dilasi citra dengan SE 3, 1 kali iterasi'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(dst_erosi, cmap = 'gray')
plt.title('Hasil erosi citra dengan SE 3, 2 kali iterasi'), plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(dst_dilate2, cmap = 'gray')
plt.title('Hasil dilasi citra dengan SE 5, 2 kali iterasi'), plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(dst_erosi2, cmap = 'gray')
plt.title('Hasil dilasi citra dengan SE 5, 3 kali iterasi'), plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(dst_dilate3, cmap = 'gray')
plt.title('Hasil dilasi citra dengan SE 3, 1 kali iterasi'), plt.xticks([]), plt.yticks([])
plt.show()

#operasi bitwise pada citra bgr dengan mask hasil operasi morfologi
res1 = cv.bitwise_and(bgr,bgr, mask= dst_dilate3)
#menerapkan medianBlur pada citra hasil operasi bitwise "res1"
blur = cv.medianBlur(res1,25)

""""
rubah gambar hasil operasi bitwise and dengan mask hasil operasi morfologi ("blur)
yang sudah diterapkan median blur karena
untuk menggunakan fungsi findCountours hanya bisa menerima
parameter gambar grayScale
"""
grayBlur = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

#mencari countours pada gambar blur ("blur") yang sudah diubah menjadi citra Gray Scale
(cnt, hierarchy) = cv.findContours(grayBlur.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#Gambar countours pada gambar blur BGR yang belum diubah menjadi grayScale
cv.drawContours(blur , cnt, -1, (0,0,255),2)

#tampilkan citra hasil proses
cv.imshow('gambar asli', bgr)
cv.imshow('blur', blur)
cv.imshow('mask', mask)
cv.imshow('result operasi bitwise pada citra bgr dengan mask hasil threhsolding awal', res)
cv.imshow('hasil operasi bitwise pada citra bgr dengan mask hasil operasi morfologi', res1)

#hitung jumlah burung pada gambar 
print("Jumlah burung pada gambar berjumlah",len(cnt), "ekor")

cv.waitKey()
