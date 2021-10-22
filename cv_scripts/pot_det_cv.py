#!/usr/bin/python

"""
Created on Thu Oct 22 11:45:06 2020

https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/

@author: cvlab
cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3,      15,      3,          5,      1.2,        0) 
Parameters
    prev : First input image in 8-bit single channel format.
    next : Second input image of same type and same size as prev.
    pyr_scale : parameter specifying the image scale to build pyramids for each 
                image (scale < 1). A classic pyramid is of generally 0.5 scale, 
                every new layer added, it is halved to the previous one.
    levels : levels=1 says, there are no extra layers (only the initial image) 
             It is the number of pyramid layers including the first image.
    winsize : It is the average window size, larger the size, the more robust 
              the algorithm is to noise, and provide fast motion detection, 
              though gives blurred motion fields.
    iterations : Number of iterations to be performed at each pyramid level.
    poly_n : It is typically 5 or 7, it is the size of the pixel neighbourhood 
             which is used to find polynomial expansion between the pixels.
    poly_sigma : standard deviation of the gaussian that is for derivatives 
                 to be smooth as the basis of the polynomial expansion. 
                 It can be 1.1 for poly= 5 and 1.5 for poly= 7.
    flow : computed flow image that has similar size as prev and type to be CV_32FC2.
    flags : It can be a combination of- OPTFLOW_USE_INITIAL_FLOW uses input flow as initial apporximation.
                                        OPTFLOW_FARNEBACK_GAUSSIAN uses gaussian winsize*winsize filter.
"""


from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import math
import matplotlib.pyplot as plt
import os

from skimage.measure import label, regionprops, regionprops_table
    
from scipy.spatial import distance
   
from cv_scripts.libs.skeletonize import skeletonize
from cv_scripts.libs.crossing_number import calculate_minutiaes

rng.seed(12345)
STEP = 4
QUIVER = (0,255,255) # yellow

PORCENTAJE = 2.5 # 40%


# noise removal
kernel = np.ones((3,3),np.uint8)
# kernel[0,0]=0
# kernel[0,2]=0
# kernel[2,0]=0
# kernel[2,2]=0

#global variable to keep track of 
show = False
show2 = False
def onTrackbarActivity(x):
    global show
    show = True
    pass

def nothing(x):
    pass


def encuentra_box(elipse):
    a = elipse[1][0]
    b = elipse[1][1]
    th = elipse[2]*math.pi/180
    # xa = sqrt(a^2 cos[th]^2 + b^2 sin[th]^2) 
    xa =  np.sqrt( a*a * math.cos(th)**2 + b*b * math.sin(th)**2 )/2
    #ya = sqrt[a^2 sin[th]^2 + b^2 cos[th]^2]}
    ya =  np.sqrt( a*a * math.sin(th)**2 + b*b * math.cos(th)**2 )/2
    boxx = np.zeros((4), dtype=np.float32)
    boxx2 = np.zeros((4, 2), dtype=np.float32)
    boxx2[0][0] = elipse[0][0]-xa
    boxx2[0][1] = elipse[0][1]-ya
    boxx2[1][0] = elipse[0][0]-xa
    boxx2[1][1] = elipse[0][1]+ya
    boxx2[2][0] = elipse[0][0]+xa
    boxx2[2][1] = elipse[0][1]+ya
    boxx2[3][0] = elipse[0][0]+xa
    boxx2[3][1] = elipse[0][1]-ya
    
    # esq sup izda y esq inf dcha
    boxx[0] = boxx2[0][0]
    boxx[1] = boxx2[0][1]
    boxx[2] = boxx2[2][0]
    boxx[3] = boxx2[2][1]
    
    return (boxx, boxx2)






cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
       (0, 1),  (1, 1),  (1, 0),            # p8    p4
      (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5

# cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
#        (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
#       (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5



def filtrar_contornos(skel, skel_sin, minutiaeBif):
    nuevos_contornos = np.zeros((skel.shape[0], skel.shape[1]), dtype=np.uint8)
    nuevos_contornos2 = np.zeros((skel.shape[0], skel.shape[1]), dtype=np.uint8)
    contornos_eliminados = np.zeros((skel.shape[0], skel.shape[1]), dtype=np.uint8)
    nuevos_contornos = minutiaeBif
    
    
    # primero determino en skel_sin que contornos son los mas cortos
    th_length = 20 # si es menor que este valor elimino ese tramo
    
    label_img = label(skel_sin, background=-1)

    
    # fig = plt.figure()
    # plt.imshow(label_img)
    # plt.show()   
            
            
    object_features = regionprops(label_img)
     
    i=1
    for props in object_features:
       
        if i>1 :
            aa = np.where(label_img == i)
        
            #print(props.area)
            if props.area < th_length:
                for ii in range(len(aa[0])):
                    xx = aa[0][ii]
                    yy = aa[1][ii]
                    contornos_eliminados[xx, yy] = 1                             
            else:
                for ii in range(len(aa[0])):
                    xx = aa[0][ii]
                    yy = aa[1][ii]
                    nuevos_contornos2[xx, yy] = 1                          
        i+=1

    # dilatar contornos_eliminados un pixel
    contornos_eliminados = cv.dilate(contornos_eliminados, kernel, iterations = 1)
    contornos_eliminados1 = cv.dilate(contornos_eliminados, kernel, iterations = 2)
    
    # fig = plt.figure()
    # plt.imshow(skel)
    # plt.show() 

    # fig = plt.figure()
    # plt.imshow(contornos_eliminados)
    # plt.show()    

    
    contornos_eliminados2 = cv.bitwise_and(contornos_eliminados, np.uint8(skel))

    # fig = plt.figure()
    # plt.imshow(contornos_eliminados2)
    # plt.show()    

   
    nuevos_contornos = cv.bitwise_and(np.uint8(skel),  cv.bitwise_not(contornos_eliminados2))    
        
    # fig = plt.figure()
    # plt.imshow(nuevos_contornos)
    # plt.show()      
    
    añadir = cv.bitwise_and(contornos_eliminados1,  nuevos_contornos)    

    # fig = plt.figure()
    # plt.imshow(añadir)
    # plt.show()  
    
    # fig = plt.figure()
    # plt.imshow(nuevos_contornos2)
    # plt.show()  

    nuevos_contornos3 = cv.bitwise_or(nuevos_contornos2,  añadir)    
    nuevos_contornos3 = cv.bitwise_or(nuevos_contornos3,  minutiaeBif)    
    
    # fig = plt.figure()
    # plt.imshow(nuevos_contornos3)
    # plt.show()  
       
    return nuevos_contornos3, contornos_eliminados 


        

def thresh_callback(threshold, src_gray, img, todas_elipses, video_writer1, video_writer3):
           
    #average brightness
    avg, null, null, null = cv.mean(src_gray)
    threshold = int(avg)
    
    ## [Canny]
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold/2, threshold)
    ##     canny_output = encimera *  canny_output
    # cv.imshow("canny_output", canny_output)       

    # fig = plt.figure()
    # plt.imshow(canny_output)
    # plt.show()    

    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    #contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    # contours = sorted(contours, key=cv.contourArea, reverse=True)
    # image = cv.cvtColor(img ,cv.COLOR_BGR2RGB)
    # cv.drawContours(image, contours, -1, (0,0,255), thickness = 1)
    # fig, ax = plt.subplots(1, figsize=(12,8))
    # plt.imshow(image)


   

# =============================================================================
#     Filtrado por tamaño y ejes
# =============================================================================
    ratio = 1.4             
    tamano = 50
    thh = 8
    nume = 5  # numero de erosiones y dilataciones
    
    # fuegos = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # contornos = np.zeros((canny_output.shape[0], canny_output.shape[1]), dtype=np.uint8)
    
    # for i, c in enumerate(contours):        
    #     if c.shape[0] > tamano:
    #         cv.drawContours(contornos, contours, i, 1)
    
    # contours, contornos1 = minucias(contornos, nume)
    
    contours, contornos = minucias(255*canny_output, nume)
    
 
    
    contornos_unidos = []
    
    if len(todas_elipses)==0:      
        #print('\n')
        #print('m_iou')
        ## obtener matriz de intersecion vs union 21-09-21 
        contours_finales, contornos_finales, polys, elipses = detectar_elipses(src_gray, contours, ratio, thh, tamano)
        m_iou = interseccion_vs_union(elipses)        
        contornos_unidos = hausdorff(m_iou, contornos_finales, polys, thh=50.0)
        elipses = elipses_finales(canny_output, contornos_unidos, contours_finales, ratio, video_writer1)    
                         
    else:
        ## seguimiento de elises con contornos acutales  04-10-21         
        contornos_unidos, contours_finales = does_it_fit(todas_elipses, contours, 10)
        elipses2 = elipses_finales(canny_output, contornos_unidos, contours_finales, ratio, video_writer1)                         
        # ## si he perdido alguna elipse hago deteccion  08-10-21    
        # if not(len(todas_elipses) == len(elipses2)):
        #     # print('se ha perdido una elipse')
        #     elipses_perdidas = []
        #     for i in range(len(todas_elipses)):
        #         ellip1 = todas_elipses[i]
        #         for j in range (len(elipses2)):
        #             ellip2 = elipses2[j]
        #             iguales = compara_dos_elipses(ellip1, ellip2, 1)
        #             if iguales==False:
        #                 elipses_perdidas.append(i)
        #             # print('¿son iguales? ', iguales)
   
        #     print('he perdido ', elipses_perdidas)    
        #     # busco elipses nuevas
        #     contours_finales, contornos_finales, polys, elipses = detectar_elipses(src_gray, contours, ratio, thh, tamano)
        #     m_iou = interseccion_vs_union(elipses)        
        #     contornos_unidos = hausdorff(m_iou, contornos_finales, polys, thh=50.0)
        #     elipses = elipses_finales(canny_output, contornos_unidos, contours_finales, ratio, []) 
                        
        # else:
        elipses = elipses2
          
 
    contornos_u = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for  i, c in enumerate(contours_finales):       
        cv.drawContours(contornos_u, np.array([c]), 0, QUIVER, 3)
        

    #cv.imshow("contornos_unidos", contornos_u)     
    if video_writer3:   
        video_writer3.write(contornos_u)          
    
    # print('len(contornos_finales) = ', len(contours_finales) )
    # print('contornos_unidos = ', contornos_unidos)           
    # print("Numero elipses finales = ", len(todas_elipses))
          
    return elipses




def detectar_elipses(src_gray, contours, ratio, thh, tamano):
        
    drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
    fuegos = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)


    contours_finales = []
    contornos_finales = []
    elipses = []
    polys = []
    n_box = 0   
 

    #print('len(contours) = ', len(contours))  
    
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # print('longitud c = ', c.shape[0])
        cc2 = reordenar(c[0:c.shape[0],0] )
         
        if len(cc2) > tamano:
            # print('longitud cc2 = ', cc2.shape[0])       
            

            # ######################## PARA DEBUGAR  ######################## 

            # if len(cc2) < 350:                               
            #     ellip = cv.fitEllipse(c) #center_coordinates, axis,  angle
            #     image2 = cv.cvtColor(src_gray, cv.COLOR_GRAY2BGR)# cv.cvtColor(img ,cv.COLOR_BGR2RGB)
            #     cv.drawContours(image2, c, -1, (255,255,0), thickness = 4)
            #     cv.ellipse(image2, ellip, (0,0,255), 1)
            #     fig, ax = plt.subplots(1, figsize=(12,8))
            #     plt.imshow(image2)     
            
            # ######################## PARA DEBUGAR  ######################## 


            ellip = cv.fitEllipse(cc2) #center_coordinates, axis,  angle          
            poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                    (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)            
        

            ########################  02-10-22   ######################## 
            # cc21, cc22 = evalua_elipse_inicial(src_gray, cc2, thh)           
            n_cc1 = cc2
            n_cc2 = []                
   
            Y = distance.cdist(cc2, poly, 'euclidean')
                                           
            dd=np.amin(Y, axis=1)
            maximo = np.amax(dd)
            #print('maximo =', maximo)        
            idx_maximo = np.argwhere(dd == maximo)
                       
            if maximo > thh:   
                ii =  idx_maximo[0][0]
                n_cc1 = cc2[0:ii,:]  
                n_cc2 = cc2[ii:len(cc2),:] 
                    
            ########################  02-10-22   ######################## 
           
            for i in range(2):
                if i>0:
                    cc2 = n_cc2
                else:
                    cc2 = n_cc1 
                    
                    
                recortar = 1
                while recortar>0:
                    # print('recortar = ', recortar)
                    cc2, poly, ellip, recortar = evalua_elipse2(src_gray, cc2, thh, 100, color, 0)
    
                # cc2, poly, ellip, recortar = evalua_elipse2(src_gray, cc2, 8, ratio2, color, 0)
    
                ## recupero parte del contorno perdido en n_cc correspondiente a cc2
                if len(cc2)>5: #There should be at least 5 points to fit the ellipse in function
                    cc2 = contorno_intersecta_elipse(cc2, ellip, thh)
                    if len(cc2)>5:
                        ellip = cv.fitEllipse(cc2) #center_coordinates, axis,  angle          
                        poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                                (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)            
                ## recupero parte del contorno perdido en n_cc correspondiente a cc2
            
                        # # copying image to another image object
                        # im1 = src_gray.copy()
                        # im1 = cv.cvtColor(im1, cv.COLOR_GRAY2BGR)
                        # cv.drawContours(im1, [cc2], 0, color,3)
                        # cv.ellipse(im1, ellip, color, 1)   
                        # fig = plt.figure()
                        # plt.imshow(im1)
                        # plt.show()            
                        
                        
                
                if len(poly) == 0:  
                    pass
                    #print('vacia')
                elif  ellip[1][0]<ratio*ellip[1][1] and ellip[1][1]<ratio*ellip[1][0] and  cc2.shape[0] > 100:
    
                    # cc2, poly, ellip, recortar = evalua_elipse(cc2, 5, ratio, color, 1)      
    
                    n_box = n_box+1
                    polys.append(poly.tolist())
                    elipses.append(list(ellip))
                    
                    contornos_finales.append(cc2.tolist())
                    # print('long. contorno = ', cc2.shape[0])    
                                    
                     
                    boxx, boxx2 = encuentra_box(ellip)                                                   
                    box = np.intp(boxx2) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                    #minRect [i] = boxx
                    cv.drawContours(fuegos, [cc2], 0, color,3)
                    cv.drawContours(fuegos, [box], 0, color)
                    
                    cv.putText(fuegos, str(int(n_box)), (box[0,0], box[0,1]),cv.FONT_HERSHEY_SIMPLEX, 1.5 , color)
    
                    cv.ellipse(drawing, ellip, color, 2)                
                    # cv.drawContours(drawing, contours, i, color)
                    cv.drawContours(drawing, [cc2], 0, color,3)
                    cv.putText(drawing, str(int(n_box)), (box[0,0]+4*n_box, box[0,1]+4*n_box),cv.FONT_HERSHEY_SIMPLEX, 1.4 , color)
                    contours_finales.append(cc2)
                   
                    
                    # # contours = sorted(contours, key=cv.contourArea, reverse=True)
                    # image = cv.cvtColor(img ,cv.COLOR_BGR2RGB)
                    # cv.drawContours(image, c, -1, (0,0,255), thickness = 1)
                    # fig, ax = plt.subplots(1, figsize=(12,8))
                    # plt.imshow(image)
                else:
                    pass
                    #print('no tiene forma eliptica')

    
    #cv.imshow("fuegos", fuegos)               
    #cv.imshow("elipses", drawing)   

    #print("Numero elipses inicial = ", len(contours_finales))     
    
    return contours_finales, contornos_finales, polys, elipses 




def does_it_fit(todas_elipses, contours, thh): 
    
    contornos_unidos = [] 
    
    contornos_finales = [] 
    fit = np.zeros((len(todas_elipses), len(contours)), dtype=np.uint8)   
    indices = np.zeros(len(contours), dtype=np.uint8)
    aux = np.zeros(len(contours), dtype=np.uint8)
    
    for ii in range(len(todas_elipses)):
        ellip = todas_elipses[ii]
        poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                   (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)   
        
        for jj, c in enumerate(contours):       
            cc2 = reordenar(c[0:c.shape[0],0] )                     
            Y = distance.cdist(cc2, poly, 'euclidean')            
            dd=np.amin(Y, axis=1)
            idx = np.argwhere(dd < thh)   

            if len(idx) > 50:
                fit[ii,jj] = 1
                aux[jj] = 1+ii
                # conviertre primero a tupla
                idx1 = tuple(idx.reshape(1, -1)[0])
                contornos_finales.append( cc2[idx1,:])
                contornos_unidos.append(1+ii)

    return np.array(contornos_unidos), contornos_finales
    
    

def elipses_finales(canny_output, contornos_unidos, contours_finales, ratio, video_writer1):
    elipses_aux = []
    indices = np.ones(len(contornos_unidos), dtype=np.uint8)

    #colors = np.int8( list(np.ndindex(2, 2, 2)) ) * 255
     
    # vis = cv.cvtColor(src_gray, cv.COLOR_GRAY2BGR)

    canny_output3 = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    canny_output2 = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    
    conta = 0 


    for ll in range(len(contornos_unidos)): 
        if indices[ll]==1:
            indices[ll]=0
            conta +=1
            cc0 = contours_finales[ll]
            unir = contornos_unidos[ll]
            aux = np.argwhere(contornos_unidos == unir)  
            for u in range(1, len(aux)):
                indices[aux[u]]=0
                cc1=contours_finales[aux[u][0]]
                cc0=np.concatenate([cc0, cc1]) 
                # cv.drawContours(canny_output2, contornos_unidos, ll, color)                

            if len(cc0)>5: #There should be at least 5 points to fit the ellipse in function
                #color1 = colors[conta % len(colors)]
                # Convert numpy array to tuple
                # color = tuple(color1.reshape(1, -1)[0])
                #color = (int(color1[0]), int(color1[1]), int(color1[2]))
                # color0 = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                
                ellipse = cv.fitEllipse(cc0)    
                boxx, boxx2 = encuentra_box(ellipse)
                box = np.intp(boxx2)
                
                # cv.drawContours(canny_output2, [cc0], 0, (0,255,255), 3)                
                # fig = plt.figure()
                # plt.imshow(canny_output2)
                # plt.show()                      
                
                if ellipse[1][0]<ratio*ellipse[1][1] and ellipse[1][1]<ratio*ellipse[1][0]:                                         
                    if (completa2(boxx, cc0)==1 and completa(boxx, cc0)==1) and cc0.shape[0] > 200:
                                              
                        # recortar = 1
                        # while recortar>0:
                        #     print('recortar = ', recortar)
                        #     cc0, poly, ellipse, recortar = evalua_elipse(cc0, 3, ratio, color, 0)

                        
                        # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))     
                        # cv.ellipse(vis, ellipse, color, 4)     
                        #cv.ellipse(canny_output3, ellipse, color, 4)    
                        # if len(todas_elipses)==0:
                        elipses_aux.append(ellipse)                                                
                       
                    #else:
                        # cv.ellipse(canny_output2, ellipse, color, 4)
                        #cv.drawContours(canny_output2, [box], 0, color, 4)
                        #cv.ellipse(canny_output2, ellipse, (255,0,255), 4)
                        #cv.drawContours(canny_output2, [cc0], 0, (0,255,255), 3)
                        # print(cc0.shape[0])
                #else:
                    # cv.ellipse(canny_output2, ellipse, color, 4)     
                    #cv.drawContours(canny_output2, [box], 0, color)
                    #cv.drawContours(canny_output2, [cc0], 0, (0,255,255))
                    # cv.ellipse(canny_output2, ellipse, color, 2)
                
    #cv.imshow("canny_output2", canny_output2) 
    #cv.imshow("canny_output3", canny_output3) 

    if video_writer1:   
        video_writer1.write(canny_output3)
    
    return elipses_aux




def minucias(contornos, nume):

    contornos_gruesos2 = cv.dilate(contornos, kernel, iterations = nume)
    contornos_gruesos2 = cv.erode(contornos_gruesos2, kernel, iterations = nume)  
    # fig = plt.figure()
    # plt.imshow(contornos_gruesos2)
    # plt.show()  

    
    contornos = cv.dilate(contornos, kernel, iterations = 0)
    contornos_gruesos = cv.erode(contornos, kernel, iterations = 0)  

    # fig = plt.figure()
    # plt.imshow(contornos_gruesos)
    # plt.show()  

    canny_output2 = 255*contornos_gruesos
    #cv.imshow("canny_output2", canny_output2)       
    
    
    # from skimage.morphology import skeletonize as skelt
    # thin_image = skelt(canny_output2)
    
    NOT = cv.bitwise_not(canny_output2)


    # # thinning oor skeletonize
    thin_image = skeletonize(NOT)

    
    skel=np.ones(thin_image.shape)
    skel[thin_image==255]=0
    

    # cv.imshow("skeletonize", skel)       
    # fig = plt.figure()
    # plt.imshow(skel)
    # plt.show()  
   
    
    (rows, cols) = skel.shape
    minutiaeTerm = np.zeros(skel.shape,  dtype=np.uint8)
    minutiaeBif = np.zeros(skel.shape,  dtype=np.uint8)
    minutiae = np.zeros(skel.shape,  dtype=np.uint8)
    otros = np.ones(skel.shape, dtype=np.uint8)
    otros2 = np.zeros(skel.shape, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (skel[i][j] == 1):
                
                values = [skel[i + l][j + k] for k, l in cells]
        
                # count crossing how many times it goes from 0 to 1
                crossings = 0
                for k in range(0, len(values)-1):
                    crossings += abs(values[k] - values[k + 1])
                crossings //= 2
        
                # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
                # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
                if crossings == 1:
                    minutiaeTerm[i, j] = 1 #"ending"
                    otros[i - 1:i + 2, j - 1:j + 2]=0
                    # otros[i, j]=0
                elif crossings == 3:
                    minutiaeBif[i, j] = 1  #"bifurcation"
                    otros[i - 1:i + 2, j - 1:j + 2]=0
                    #otros[i, j]=0
                else:
                    otros2[i, j] = 1;
                  

    
# =============================================================================
#     fig = plt.figure()
#     plt.imshow(skel)
#     plt.show()    
# =============================================================================


    fgMask2 = cv.bitwise_and(otros, otros, mask=otros2)
    fgMask2 = cv.bitwise_or(fgMask2,minutiaeTerm)
    
    # fig = plt.figure()
    # plt.imshow(fgMask2)
    # plt.show()  
    
    nuevos_contornos = fgMask2
    
    # # 08-09-21
    nuevos_contornos, contornos_eliminados = filtrar_contornos(skel, fgMask2, minutiaeBif)
    # cv.imshow("nuevos_contornos", 255*nuevos_contornos)    
    
    # fig = plt.figure()
    # plt.imshow(nuevos_contornos)
    # plt.show()  
    
    # contours, hierarchy = cv.findContours(cv.bitwise_not(thin_image), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv.findContours(nuevos_contornos, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(nuevos_contornos, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
       
    
    return contours, nuevos_contornos
 
    
     
def completa2(boxx,cc0):
    th0 = np.abs(boxx[2]-boxx[0])/PORCENTAJE # 40%
    th1 = np.abs(boxx[3]-boxx[1])/PORCENTAJE # 40%

    min_x = np.min(cc0[:,0])
    min_y = np.min(cc0[:,1])
    max_x = np.max(cc0[:,0])
    max_y = np.max(cc0[:,1])
    
    flag = 1
    if abs(min_y - boxx[1]) > th1: # toca 1
        flag = 0
    if abs(max_y - boxx[3]) > th1: # toca 3
        flag = 0
    if abs(max_x - boxx[2]) > th0: # toca 2
        flag = 0
    if abs(min_x - boxx[0]) > th0: # toca 4
        flag = 0
        
    return flag            
    
    
def completa(boxx,cc0):
    th0 = np.abs(boxx[2]-boxx[0])/25 # 4%
    th1 = np.abs(boxx[3]-boxx[1])/25 # 4%

    min_x = np.min(cc0[:,0])
    min_y = np.min(cc0[:,1])
    max_x = np.max(cc0[:,0])
    max_y = np.max(cc0[:,1])
    
    conta = 0
    if abs(min_y - boxx[1]) < th1: # toca 1
        conta += 1
    if abs(max_y - boxx[3]) < th1: # toca 3
        conta += 1
    if abs(max_x - boxx[2]) < th0: # toca 2
        conta += 1
    if abs(min_x - boxx[0]) < th0: # toca 4
        conta += 1
        
    if conta >2 :
        flag = 1
    else:
        flag = 0
        
    return flag        
        
'''
    Bounding Box de la elipse: esq sup izda y esq inf dcha
    min_x_box = boxx[0] = boxx2[0][0]
    min_y_box = boxx[1] = boxx2[0][1]
    max_x_box = boxx[2] = boxx2[2][0]
    max_y_box = boxx[3] = boxx2[2][1]
    
                1
             __________
            |         |
        4   |         |  2
            |         |
            |_________|                        
                3
                                             
        1.- if abs(min_y - min_y_box) < th0  # toca 1
        2.- if abs(max_x - max_x_box) < th0  # toca 2
        3.- if abs(max_y - max_y_box) < th0  # toca 3
        4.- if abs(min_x - min_x_box) < th0  # toca 4               
'''    





def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)


def contorno_intersecta_elipse(cc2, ellip, thh):    
    poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                       (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)            
    
    Y = distance.cdist(cc2, poly, 'euclidean')

    dd=np.amin(Y, axis=1)     
    idx = np.argwhere(dd <= thh)
    
    n_cc = np.zeros((len(idx),2), dtype=np.uint32)
    
    for i in range(len(idx)):
        n_cc[i] = cc2[idx[i]]           
            
    return n_cc.astype('int') 



def evalua_elipse2(imagen, cc2, thh, ratio, color, ver):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))                   
    recortar = 0
    n_cc = cc2
    poly = []   
    ellip = []
    if len(cc2)>5: #There should be at least 5 points to fit the ellipse in function
        ellip = cv.fitEllipse(cc2) #center_coordinates, axis,  angle
        if  ellip[1][0]>ratio*ellip[1][1] or ellip[1][1]>ratio*ellip[1][0] :     
            N=3
        else:
            poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                               (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)            

            n_cc = cc2      
            
            
            # # copying image to another image object
            # im1 = imagen.copy()
            # im1 = cv.cvtColor(im1, cv.COLOR_GRAY2BGR)
            # cv.drawContours(im1, [cc2], 0, color,3)
            # cv.ellipse(im1, ellip, color, 1)   
            # fig = plt.figure()
            # plt.imshow(im1)
            # plt.show()  

            Y = distance.cdist(cc2, poly, 'euclidean')
            
            # checking the equation of ellipse with the given point
            # ellip = cv.fitEllipse(c) #center_coordinates, axis,  angle
            hh,kk = ellip[0]
            aa,bb = ellip[1]
            angle = ellip[2]
            cos_angle = np.cos(np.radians(180.-angle))
            sin_angle = np.sin(np.radians(180.-angle))
    
 
            ll = len(cc2)# c.shape[0]
            #cc = np.zeros((2, 1), dtype=np.uint8)
            negativos = -np.ones((ll, 1), dtype=np.float32)
            signo = np.zeros((ll, 1), dtype=float)
            for jj in range(ll):
                #cc = c[jj]                       
                px = cc2[jj][0]
                py = cc2[jj][1]
    
                xc = px - hh
                yc = py - kk
                xct = xc * cos_angle - yc * sin_angle
                yct = xc * sin_angle + yc * cos_angle 
                signo[jj] = (xct**2/(aa/2.)**2) + (yct**2/(bb/2.)**2)    
                #p = ((math.pow((x - h), 2) // math.pow(a, 2)) +  (math.pow((y - k), 2) // math.pow(b, 2)))
    
            zz = 2*np.float32(np.array(signo)>1)+negativos
            
            # cruces por cero
            zero_cross = []
            for cr in range(len(zz)-1):
                if abs(zz[cr]-zz[cr+1]) > 1:
                    zero_cross.append(cr)
                
                
                    
            dd=np.amin(Y, axis=1)
            maximo = np.amax(dd)
            #print('maximo =', maximo)        
            idx_maximo = np.argwhere(dd == maximo)
    
                

            if maximo > thh:                
                if zero_cross == []:
                    poly = []                  
                elif abs(idx_maximo[0]-len(dd)) > idx_maximo[0]: # pto cercano al inicio                   
                    idx_prox = np.argwhere(zero_cross >= idx_maximo[0])                        
                    if len(idx_prox)==0:
                        ii=[]
                        ii.append(0)
                    else:
                        ii=idx_prox[0]
                    n_cc = cc2[zero_cross[ii[0]]:len(cc2),:]                    
                    # recortar = 1          
    
                else:                                    # pto cercano al final
                    # idx_prox = np.argwhere(zero_cross < idx_maximo[0])    
                    # val = idx_prox[len(idx_prox)-1]-1
                    # # val = idx_prox[len(idx_prox)-1]
                    # fin = zero_cross[val[0]]
                    # n_cc = cc2[0:fin, :]                    
                    # recortar = 1                      
                    idx_prox = np.argwhere(zero_cross < idx_maximo[0])    
                    if len(idx_prox)>1:
                        val = idx_prox[len(idx_prox)-1]-1
                        fin = zero_cross[val[0]]
                    elif len(idx_prox)>0:
                        val = idx_prox[0]-1
                        fin = zero_cross[val[0]]
                    else: # no corta con ninguno
                        fin = len(cc2)                       
                    n_cc = cc2[0:fin, :]                       

                # print('len cc2 = ', len(cc2), 'len n_cc = ', len(n_cc))

                # img = imagen.copy()
                # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)    
                # cv.drawContours(img, [n_cc], 0, color,3)
                # if len(n_cc)>5:
                #     ellip = cv.fitEllipse(n_cc)
                #     cv.ellipse(img, ellip, color, 1)                      
                # fig = plt.figure()
                # plt.imshow(img)
                # plt.show()                      
                    
                # ## recupero parte del contorno perdido en n_cc correspondiente a cc2
                # if len(n_cc)>5: #There should be at least 5 points to fit the ellipse in function
                #     n_cc = contorno_intersecta_elipse(cc2, cv.fitEllipse(n_cc), thh)
                #     if len(n_cc)>5:
                #         ellip = cv.fitEllipse(n_cc) #center_coordinates, axis,  angle          
                #         poly = cv.ellipse2Poly((int(ellip[0][0]), int(ellip[0][1])), 
                #                (int(ellip[1][0] / 2), int(ellip[1][1] / 2)), int(ellip[2]), 0, 360, 5)            
                # ## recupero parte del contorno perdido en n_cc correspondiente a cc2

                # # copying image to another image object
                # im2 = imagen.copy()
                # im2 = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
                # cv.drawContours(im2, [n_cc], 0, color,3)
                # if len(n_cc)>5:
                #     ellip = cv.fitEllipse(n_cc)                 
                #     cv.ellipse(im2, ellip, color, 1)   
                # fig = plt.figure()
                # plt.imshow(im2)
                # plt.show()  
    
               
                if ver == 1:  
                    
                     dd2 = np.zeros((len(dd), 1), dtype=np.float)
                     for gg in range(len(dd)):
                         dd2[gg]=dd[gg]               
                    
                     auxx = np.multiply(dd2,zz)
        
                     colour = np.random.uniform(0, 1, 3)                
                    # colour=np.ones((3,1),dtype=np.float )
                
                     colour[1]=color[1]/256
                     colour[2]=color[0]/256
                     colour[0]=color[2]/256            
                
                     xx = np.arange(0, len(auxx))
                     fig = plt.figure()        
                     plt.plot(xx, auxx,
                     linewidth=0.5,
                     linestyle='--',
                     color='b',
                     marker='o',
                     markersize=10,
                     markerfacecolor=colour)       
                     plt.grid(True)
                     plt.show()    
                     
    if len(cc2)==len(n_cc):
        recortar = 0
    else:
        recortar = 1                     
                        
    return (n_cc, poly, ellip, recortar)


def  interseccion_vs_union(elipses):                       
    matriz=np.zeros((len(elipses),len(elipses)), dtype=np.float32)
    for ll in range(len(elipses)):      
        ellip=np.array(elipses[ll], dtype=object)
        boxx1, boxxi = encuentra_box(ellip)                                                   
        for ll2 in range(len(elipses)):   
            ellip=np.array(elipses[ll2], dtype=object)              
            boxx2, boxxi = encuentra_box(ellip)                                                   
            """
            Computes IUO between two bboxes in the form [x1,y1,x2,y2]
            """
            matriz[ll,ll2]=iou(boxx1,boxx2)
            
    #print(np.around(matriz,2))    
    
    return matriz

    
def check_if_equal(list_1, list_2):
    """ Check if both the lists are of same length and if yes then compare
    sorted versions of both the list to check if both of them are equal
    i.e. contain similar elements with same frequency. """
    if len(list_1) != len(list_2):
        return False
    return sorted(list_1) == sorted(list_2)

    

def hausdorff(m_iou, contornos_finales, polys, thh):
    ## Unir contornos 18-09-21    
    tabla_maximos=np.zeros((len(polys),len(polys)), dtype=np.float32)
    for ll in range(len(polys)):      
        con2=np.array(contornos_finales[ll])
        # print(len(con2))
        for ll2 in range(len(polys)):      
            poly1 = np.array(polys[ll2])      
            Y = distance.cdist(con2[0:], poly1, 'euclidean')
            dd=np.amin(Y, axis=1)
            maximo = np.amax(dd)            
            tabla_maximos[ll,ll2]=maximo

    #print('\n')     
    #print('hausdorff') 
    #print(np.around(tabla_maximos,1))
    
    hausdorf=np.zeros((len(polys),len(polys)), dtype=np.float32)
    for ll in range(len(polys)): 
        for ll2 in range(len(polys)):
            if tabla_maximos[ll,ll2] <  tabla_maximos[ll2,ll]:
                hausdorf[ll, ll2] = tabla_maximos[ll,ll2] 
            else:
                hausdorf[ll, ll2] = tabla_maximos[ll2,ll] 

    #print('\n')
    #print(np.around(hausdorf,1))

    

    ## unir con iou 21-09-21    
    # multiply-numpy-matrix-elementwise-without-for-loops
    zz=100*np.add(m_iou, 0.1)
    hausdorf2 = np.divide(hausdorf,zz)
        
    #print('\n')
    #print('Multiplicar por 100')
    #print(np.around(hausdorf2,1))   
    
    hausdorf2 *= 100
    
    ## unir con iou 21-09-21    
    
    
    
    contornos_unidos = np.zeros((len(polys),len(polys)), dtype=np.uint8)
    for ll in range(len(polys)): 
        arr = hausdorf2[ll,:]
        aux = np.argwhere(arr < thh)    
        for kk in range(len(aux)):            
            contornos_unidos[ll, aux[kk]] = 1

    #print('\n')
    #print(contornos_unidos)
    
    
    conta = 1
    contornos_fin = np.zeros(len(polys), dtype=np.uint8)
    for ll in range(len(polys)):      
        lista_old = np.zeros(len(polys), dtype=np.uint8)
        lista_new = contornos_unidos[ll]
        # print(check_if_equal(lista_old, lista_new))
        while check_if_equal(lista_old, lista_new)==False:
            lista_old = lista_new
            for li in range(len(lista_new)):
                if lista_new[li]==1:
                    lista_new = lista_new | contornos_unidos[li]
        
        contornos_fin[np.argwhere(lista_new == 1) ] = conta
        conta +=1  
               
    # contornos_unidos = np.zeros(len(polys), dtype=np.uint8)
    # indices = np.ones(len(polys), dtype=np.uint8)
    # for ll in range(len(polys)): 
    #     if indices[ll]==1:
    #         contornos_unidos[ll]=ll
    #         for ll2 in range(len(polys)): 
    #             arr = hausdorf[ll,:]
    #             aux = np.argwhere(arr < thh)                
    #             contornos_unidos[aux] = ll
    #             indices[aux]=0
    
    #print(contornos_fin)
       
    return contornos_fin
    


def reordenar(con):

    ll = con.shape[0]
    aux = np.zeros((ll,2), dtype=np.int32)
    cc = aux
    repes = []
    for jj in range(ll):
        pt1 = con[jj]
        aux[jj] = con[jj]    #  (px,py)
        if jj < ll-2:
            pt2 = con[jj+2]
            
            if pt1[0]==pt2[0] and pt1[1]==pt2[1]:
#                print(jj+1)
                repes.append(jj)
    
    if len(repes)==2:        
        cc = aux[repes[0]+1:1+repes[1]+1]
    elif len(repes)==1:   
        if repes[0] > len(aux)-repes[0]: # cerca fin
            cc = aux[0:repes[0]+1]
        else:                           # cerca inicio
            cc = aux[repes[0]:len(aux)]    
        
    elif len(repes)==0:    
        cc = aux
    else:    
        #print('No se como cortar esta curva',len(repes) )    
        # image2 = cv.cvtColor(src_gray, cv.COLOR_GRAY2BGR)# cv.cvtColor(img ,cv.COLOR_BGR2RGB)
        # cv.drawContours(image2, [cc], 0, (255,255,0), thickness = 4)
        # fig, ax = plt.subplots(1, figsize=(12,8))
        # plt.imshow(image2)     
        cc = aux

    
    return cc

def compara_dos_elipses(ellip1, ellip2, modo):
    # modo=1 -> se compara centro y dim ejes
    # modo=2 -> se compara dim ejes y aspect ratio
    #center_coordinates, axis,  angle
    ratio = 0.1 # 10%
    iguales = True
    if modo == 1:
        if abs(ellip1[0][0] - ellip2[0][0]) > 20 or abs(ellip1[0][1] - ellip2[0][1]) > 40:
            iguales = False
        if abs(ellip1[1][0] - ellip2[1][0]) > ratio*ellip1[1][0] or abs(ellip1[1][1] - ellip2[1][1]) > ratio*ellip1[1][1]:
            iguales = False

           
    else:
        if abs(ellip1[1][0] - ellip2[1][0]) > ratio*ellip1[1][0] or abs(ellip1[1][1] - ellip2[1][1]) > ratio*ellip1[1][1]:
            iguales = False
        aspect_ratio1 = ellip1[1][0] / ellip1[1][1]
        aspect_ratio2 = ellip2[1][0] / ellip2[1][1]
        if abs(aspect_ratio1 - aspect_ratio2) > ratio*aspect_ratio1:
            iguales = False
     
    return iguales


def convierte_tuple2(ee):    
    # convierte un array en tupla
    aaa = []
    aa = (ee[0] ,ee[1]) # es una tupla de 2
    aa0 = list(aa)
    bb = (ee[2] ,ee[3]) # es una tupla de 2
    bb0 = list(bb)    

    aaa.append(aa0)    
    aaa.append(bb0)    
    aaa.append(ee[4])
  
    return tuple(aaa)


def detect_pots_cv(cap, init_frame, search_frames, blurri, VIS=False):

    todas_elipses = []
    elipses_totales = []
    num_elipses_totales = []

    idx = 0
    for idx in range(search_frames):         
        cap.set(cv.CAP_PROP_POS_FRAMES, init_frame+idx)    
        #print('vamos por la imagen: ', init_frame+idx) 
        
        ret, img = cap.read()    
        if not ret:
            print('Unable to open' )        
            # cv.destroyAllWindows()
            cap.release()
            break
        else:            
            src_gray2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            src_gray2 = cv.blur(src_gray2, (blurri,blurri))
            # subrutina principal
            auxx = thresh_callback(100, src_gray2, img, todas_elipses, None, [])         
            elipses_totales.append(auxx) 
            num_elipses_totales.append(len(auxx))        
            #if len(auxx)==0:
                #print('lista vacia')
                #vis = cv.cvtColor(src_gray2, cv.COLOR_GRAY2BGR)
                #cv.rectangle(vis, (10, 2), (100,20), (255,255,255), -1)
                #cv.putText(vis, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                #cv.imshow(source_window, vis)               
            #else:
                #for e in range(len(auxx)):
                    #ellip=auxx[e]
                    #color1 = colors[1+e]
                    # Convert numpy array to tuple
                    #color = (np.int(color1[0]), np.int(color1[1]), np.int(color1[2]))            
                    #cv.ellipse(img, ellip, color, 4)

                #cv.rectangle(img, (10, 2), (100,20), (255,255,255), -1)
                #cv.putText(img, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                #cv.imshow(source_window, img)
    
    # INICIO -- Encontramos esas elipses  # 13-10-21

    num_elips = np.max(num_elipses_totales)
    inicio = 0
    for i in range(len(elipses_totales)):
        if num_elipses_totales[i] == num_elips:
            break
        else:
            inicio = inicio+1
            
    # reordeno para q la primera tenga todas las elipses encontradas
    if inicio > 0: # reordeno
        new_elipses_totales = []
        for i in range(inicio, len(elipses_totales)):
            new_elipses_totales.append(elipses_totales[i])
        for i in range(inicio):
            new_elipses_totales.append(elipses_totales[i])
    else:
        new_elipses_totales = elipses_totales
        
    # comparo el resto con la primera
    compara = np.zeros((len(elipses_totales), num_elips),  dtype=np.int32)  
    cuantas_iguales = np.ones(num_elips,  dtype=np.int32)   
    for i in range(len(new_elipses_totales[0])): # max = num_elips
        ellip1 = new_elipses_totales[0][i]
        compara[0][i] = 1+i # consigo misma
        for j in range(1, len(elipses_totales)):
            for k in range(len(new_elipses_totales[j])):
                ellip2 = new_elipses_totales[j][k]
                if compara_dos_elipses(ellip1, ellip2, 1) == True:
                    compara[j][k] = 1+i
                    cuantas_iguales[i] +=1
                    
    #print(compara)                
            
    todas_elipses = [] 
    #print(new_elipses_totales[0])
    for i in range(num_elips): # max = num_elips
        elipse_mediana = np.zeros((cuantas_iguales[i], 5),  dtype=np.int32) 
        idx=0
        for j in range(len(elipses_totales)):
            for k in range(num_elips):
                if compara[j][k] == 1+i:
                    aux = k 
                    ellipse = new_elipses_totales[j][aux]
                    elipse_mediana[idx][0]  = ellipse[0][0]
                    elipse_mediana[idx][1]  = ellipse[0][1]
                    elipse_mediana[idx][2]  = ellipse[1][0]
                    elipse_mediana[idx][3]  = ellipse[1][1]
                    elipse_mediana[idx][4]  = ellipse[2]
                    idx +=1
                                
        ee = np.median(elipse_mediana,0)
        todas_elipses.append(convierte_tuple2(ee))

    #print('\n')    
    #print(todas_elipses)

    for ellip in todas_elipses:
        cv.ellipse(img, ellip, (0,0,255), 4)
    
    if VIS:
        cv.imshow("LAS ELIPSES FINALES", img)
    #cv.waitKey(0)

    return todas_elipses        


def main():

    ############################################################################
    # INICIO -- inicializacion de variables y demas
    ############################################################################


    # video = 'DCA6327319E6_2020_07_03_211702.mp4'
    # frame=6945

    # video = 'DCA6327319E6_2020_09_24_130541.mp4'
    # frame=200

    # video = 'DCA6327319E6_2020_07_01_213929.mp4'
    # frame=100

    # video = 'DCA6327319E6_2020_11_21_135628.mp4'
    # frame=40


    # video = 'DCA6327319BA_2021_01_21_110822.mp4'
    # tiempo = ' un minuto y 33 sgs = 93sgs'
    # fps = 25
    # frame=fps*93


    # video = 'DCA6327319E6_2020_09_07_140251.mp4'
    # tiempo = ' 11 minutos y 59 sgs '
    # fps = 25
    # frame=18031


    # video = 'DCA6327319E6_2020_09_24_130541.mp4'
    # frame=3702 - 30


    # video =  'DCA6327319BA_2021_01_21_110822.mp4'
    # frame  = 7728

    # cap = cv.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/out_remover/remover_1.avi")
    # frame = 1

    # cap = cv.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/videos de campana/DCA6327319E6_2020_09_24_130541.mp4")


    video = 'DCA6327319E6_2020_09_24_130541.mp4'
    frame=3699  
        

    # video = 'DCA6327319E6_2020_09_06_133156.mp4'
    # # tiempo = ' 26 minutos y 48 sgs '
    # # fps = 25
    # # frame=fps*(25*60+34)

    # frame=38350

    cap = cv.VideoCapture("E:/tfm/actions/videos/" + video)



    # video = 'remover_3.avi'
    # frame = 10
    # cap = cv.VideoCapture("D:/Carlos/Proyectos investigacion/BSH/Videos/out_remover/" + video)


    if not cap.isOpened():
        print('Unable to open' )
        exit(0)  

    # numero de fotogramas
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # fotograma actual en ms
    videoTimestamp = int(cap.get(cv.CAP_PROP_POS_MSEC))   
    # fotograma actual s
    videoFrameNumber = int(cap.get(cv.CAP_PROP_POS_FRAMES))  
    # # For example, to start reading 101th frame of the video you can use:
    cap.set(cv.CAP_PROP_POS_FRAMES, frame-1)

    fps = int(cap.get(cv.CAP_PROP_FPS))



    ret, img = cap.read()
    #calculate the x percent of original dimensions
    scale_percent = 1
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent )

    dsize = (width, height)


    fourcc = cv.VideoWriter_fourcc(*'mp4v') # set video extension type
    codec = cv.VideoWriter_fourcc(*'DIVX')
    save_as = "D:/Carlos/python/data/clips/output"
    video_writer1 = None
    video_writer2 = None
    video_writer3 = None

    #source_window = 'Source'
    #cv.namedWindow(source_window, cv.WINDOW_AUTOSIZE)


    blurri = 5

    todas_elipses = []

    elipses_totales = []
    num_elipses_totales = []

    colors = np.int8( list(np.ndindex(2, 2, 2)) ) * 255


    print(detect_pots_cv(cap,frame,10,blurri))
    exit()



    ############################################################################
    # INICIO -- identificacion automatica de las elipses a partir de 10 frames
    ############################################################################

    idx = 0
    for idx in range(10):         
        cap.set(cv.CAP_PROP_POS_FRAMES, frame+idx)    
        print('vamos por la imagen: ', frame+idx) 
        
        ret, img = cap.read()    
        if not ret:
            print('Unable to open' )        
            # cv.destroyAllWindows()
            cap.release()
            break
        else:            
            src_gray2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            src_gray2 = cv.blur(src_gray2, (blurri,blurri))
            # subrutina principal
            auxx = thresh_callback(100, src_gray2, img, todas_elipses, video_writer1, [])         
            elipses_totales.append(auxx) 
            num_elipses_totales.append(len(auxx))        
            if len(auxx)==0:
                print('lista vacia')
                vis = cv.cvtColor(src_gray2, cv.COLOR_GRAY2BGR)
                cv.rectangle(vis, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(vis, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv.imshow(source_window, vis)               
            else:
                for e in range(len(auxx)):
                    ellip=auxx[e]
                    color1 = colors[1+e]
                    # Convert numpy array to tuple
                    color = (np.int(color1[0]), np.int(color1[1]), np.int(color1[2]))            
                    cv.ellipse(img, ellip, color, 4)

                cv.rectangle(img, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(img, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv.imshow(source_window, img)
        
        
    if not video_writer1:        
        video_writer1 = cv.VideoWriter(save_as + '/elipses1_' +str(frame)+'_'+video, fourcc, 10.0, dsize) # path_name, video type, frame rate, (image_width, image_height)            video_writer1 = cv2.VideoWriter(save_as + '/OF_hog_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, (width, height))
    if not video_writer2:        
        video_writer2 = cv.VideoWriter(save_as + '/elipses2_' +str(frame)+'_'+video, fourcc, 10.0, dsize) # path_name, video type, frame rate, (image_width, image_height)            video_writer1 = cv2.VideoWriter(save_as + '/OF_hog_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, (width, height))
    if not video_writer3:        
        video_writer3 = cv.VideoWriter(save_as + '/elipses3_' +str(frame)+'_'+video, fourcc, 10.0, dsize) # path_name, video type, frame rate, (image_width, image_height)            video_writer1 = cv2.VideoWriter(save_as + '/OF_hog_1300 DCA6327319E6_2020_07_03_211702_2_4.mp4', codec, 20.0, (width, height))
            


    ############################################################################
    # INICIO -- Encontramos esas elipses  # 13-10-21
    ############################################################################

    num_elips = np.max(num_elipses_totales)
    inicio = 0
    for i in range(len(elipses_totales)):
        if num_elipses_totales[i] == num_elips:
            break
        else:
            inicio = inicio+1
            
    # reordeno para q la primera tenga todas las elipses encontradas
    if inicio > 0: # reordeno
        new_elipses_totales = []
        for i in range(inicio, len(elipses_totales)):
            new_elipses_totales.append(elipses_totales[i])
        for i in range(inicio):
            new_elipses_totales.append(elipses_totales[i])
    else:
        new_elipses_totales = elipses_totales
        
    # comparo el resto con la primera
    compara = np.zeros((len(elipses_totales), num_elips),  dtype=np.int32)  
    cuantas_iguales = np.ones(num_elips,  dtype=np.int32)   
    for i in range(len(new_elipses_totales[0])): # max = num_elips
        ellip1 = new_elipses_totales[0][i]
        compara[0][i] = 1+i # consigo misma
        for j in range(1, len(elipses_totales)):
            for k in range(len(new_elipses_totales[j])):
                ellip2 = new_elipses_totales[j][k]
                if compara_dos_elipses(ellip1, ellip2, 1) == True:
                    compara[j][k] = 1+i
                    cuantas_iguales[i] +=1
                    
    print(compara)                
            
    todas_elipses = [] 
    print(new_elipses_totales[0])
    for i in range(num_elips): # max = num_elips
        elipse_mediana = np.zeros((cuantas_iguales[i], 5),  dtype=np.int32) 
        idx=0
        for j in range(len(elipses_totales)):
            for k in range(num_elips):
                if compara[j][k] == 1+i:
                    aux = k 
                    ellipse = new_elipses_totales[j][aux]
                    elipse_mediana[idx][0]  = ellipse[0][0]
                    elipse_mediana[idx][1]  = ellipse[0][1]
                    elipse_mediana[idx][2]  = ellipse[1][0]
                    elipse_mediana[idx][3]  = ellipse[1][1]
                    elipse_mediana[idx][4]  = ellipse[2]
                    idx +=1
                                
        ee = np.median(elipse_mediana,0)
        todas_elipses.append(convierte_tuple2(ee))

    print('\n')    
    print(todas_elipses)




    ############################################################################
    # PRINCIPAL -- una vez detectadas:todas_elipses, se buscan en los fotogramas
    ############################################################################

    elipses_totales = []

    for idx in range(200):
        
        PORCENTAJE = 1.8
        # el porcentaje de solapamiento es menor dado que parto de las elipses
        
        cap.set(cv.CAP_PROP_POS_FRAMES, frame+idx)
        
        print('vamos por la imagen: ', frame+idx)
        
        ret, img = cap.read()
        
        if not ret:
            print('Unable to open' )        
            # cv.destroyAllWindows()
            cap.release()
            video_writer1 and video_writer1.release()
            video_writer2 and video_writer2.release()   
            video_writer3 and video_writer3.release()   
            break
        else:            
            src_gray2 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            src_gray2 = cv.blur(src_gray2, (blurri,blurri))
        
            auxx = thresh_callback(100, src_gray2, img, todas_elipses, video_writer1, video_writer3)         
            elipses_totales.append(len(auxx)) 
            
            if len(auxx)==0:
                print('lista vacia')
                vis = cv.cvtColor(src_gray2, cv.COLOR_GRAY2BGR)
                cv.rectangle(vis, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(vis, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv.imshow(source_window, vis)               
                video_writer2.write(vis) 
            else:
                for e in range(len(auxx)):
                    ellip=auxx[e]
                    for f in range(len(todas_elipses)):
                        ellip2 = todas_elipses[f]
                        if compara_dos_elipses(ellip, ellip2, 1) == True:
                            color1 = colors[1+f]
                    # Convert numpy array to tuple
                    color = (np.int(color1[0]), np.int(color1[1]), np.int(color1[2]))            
                    # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)) 
                    cv.ellipse(img, ellip, color, 4)

                cv.rectangle(img, (10, 2), (100,20), (255,255,255), -1)
                cv.putText(img, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv.imshow(source_window, img)
                video_writer2.write(img) 
        
        
        keyboard = cv.waitKey(1) & 0xFF
        if  keyboard == 'q' or keyboard == 27:
            cap.release()
            video_writer1 and video_writer1.release()
            video_writer2 and video_writer2.release()
            video_writer3 and video_writer3.release()   
            print('Total de elipses: ', np.sum(elipses_totales))
            break
        
    # cv.destroyAllWindows()
    cap.release()
    video_writer1 and video_writer1.release()
    video_writer2 and video_writer2.release()
    video_writer3 and video_writer3.release()   
    print('Total de elipses: ', np.sum(elipses_totales))    

if __name__ == '__main__':
    main()
    

    










                