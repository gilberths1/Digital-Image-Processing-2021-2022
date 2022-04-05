import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Resize image
height = 500
width = 500
dimensions = (width, height)

#read covid_image
imgasli = cv.imread('covid_image.png',0)
#resize citra covid_image
imgasli = cv.resize(imgasli, dimensions, interpolation=cv.INTER_LINEAR)

#read mask image
imgfilter = cv.imread('mask.png',0)
#resize citra mask
imgfilter = cv.resize(imgfilter, dimensions, interpolation = cv.INTER_LINEAR)

#do bitwise operation AND
bitwiseAnd = cv.bitwise_and(imgasli, imgfilter)
#show result of bitwise operation AND
cv.imshow("AND", bitwiseAnd)
cv.imwrite("res_bitwise.jpg", bitwiseAnd)

#Remove noise with median filter, gaussian filter, and average filter
redcnoise = cv.medianBlur(bitwiseAnd, 3)
gausnoise = cv.GaussianBlur(bitwiseAnd, (3,3),0)
avgnoise = cv.blur(bitwiseAnd, (3,3))

#Compare the results of noise removal 
plt.subplot(3,1,1), plt.imshow(redcnoise,cmap='gray')
plt.title('median filter'),plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2), plt.imshow(gausnoise,cmap='gray')
plt.title('gaussian filter'),plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3), plt.imshow(avgnoise,cmap='gray')
plt.title('average filter'),plt.xticks([]), plt.yticks([])
plt.savefig('perbandinganFilter.jpg')
plt.show()
cv.imwrite("res_noise_removal.jpg", redcnoise)

#show the histogram of the image after removing noise with median filter 
plt.hist(redcnoise.flatten(), 256, [0, 256])
plt.savefig('hist_awal.jpg')
plt.show()

#stretch the histogram 
hist, bins = np.histogram(redcnoise.ravel(), 256, [0, 256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
hasil_stretch = cdf[redcnoise]

#save the result of histogram stretching
cv.imwrite("res_contras.jpg", hasil_stretch)
cv.imshow('Hasil Contras', hasil_stretch)

#plot the histogram after image was stretched
plt.hist(hasil_stretch.ravel(), 256, [0, 256])
plt.savefig('hist_akhir.jpg')
plt.show()

#do binary thresholding to detect the infection on the lungs
ret, hasil = cv.threshold(redcnoise,125,255,cv.THRESH_BINARY)
cv.imwrite("final.jpg", hasil)
cv.imshow('Hasil Final', hasil)

cv.waitKey()
cv.killAllWindows()
