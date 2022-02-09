# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:43:18 2022

@author: hbuzzi
"""
import cv2
import math
import numpy as np
import sys

def f_template_cartas(cartas, inf_lim, sup_lim, percentage):
    numero = list(cartas.keys())
    naipe = list(cartas[numero[0]].keys())
    for i in numero:
        for j in naipe:
            ret, bin_image = cv2.threshold(cv2.cvtColor(cartas[i][j], cv2.COLOR_BGR2GRAY), inf_lim, sup_lim, cv2.THRESH_BINARY_INV) #Criação da imagem binária do template
            contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0])
            ajust_x, ajust_y = round((x+w)*percentage),round((y+h)*percentage)
            roi_image = bin_image[y+ajust_y:y+h-ajust_y,x+ajust_x:x+w-ajust_x]
            symbol_contours, __ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dif_ant = float('inf')
            u,v = roi_image.shape
            u,v = u//2,v//2
            dist = math.sqrt(u**2 + v**2)
            for cnt in symbol_contours:
                x,y,w,h = cv2.boundingRect(cnt)
                dif = abs(dist - math.sqrt((y+(h//2))**2 + (x+(w//2))**2))
                if dif < dif_ant:
                    dif_ant = dif
                    roi_center = roi_image[y:y+h,x:x+w]
            cartas[i][j] = roi_center
    return cartas

def f_ident_cards(carta_analisa, template, inf_lim, sup_lim, border_percentage, filter_percentage):
    def f_template_matching(template, roi, method):
        '''Algoritmo de template_matching implementado com 6 métodos diferentes
        
        Input: Dicionário do alfabeto, letra que está sendo identificada, métododo
        
        Output: Caractere que identifica a letra'''
        roi = np.float64(roi) #Transforma a matriz de imagem da letra em float64 para que as computações matemáticas ocorram corretamente
        Id = {} #Pré-alocação do dicionário de valores de identificação
        numero = list(template.keys())
        naipe = list(template[numero[0]].keys())
        if(method == 'sad'): #Método Sum of Absolute Differences
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC) #Transforma a letra do template para o mesmo tamanho que a letra encontrada no texto
                    Id.update({i + ' ' + j:np.sum(np.abs(roi - T))}) #Realiza o cálculo matemático de similaridade e salva em sua respectiva posição de letra
            return min(Id, key=Id.get) #Retorna qual posição do alfabeto foi calculado o menor valor
        elif(method == 'zsad'): #Método SAD com zero-offset
            #Funcionamento idêntico ao descrito no método SAD, apenas alterando o cálculo matemático para verificação da diferença das letras
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC)
                    Id.update({i + ' ' + j:np.sum(np.abs((roi - np.mean(roi)) - (T - np.mean(T))))})
            return min(Id, key=Id.get)
        elif(method == 'ssd'): #Método Sum of Squared Differences
            #Funcionamento idêntico ao descrito no método SAD, apenas alterando o cálculo matemático para verificação da diferença das letras
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC)
                    Id.update({i + ' ' + j:np.sum((roi - T)**2)})
            return min(Id, key=Id.get)
        elif(method == 'zssd'): #Método SSD com zero-offset
            #Funcionamento idêntico ao descrito no método SAD, apenas alterando o cálculo matemático para verificação da diferença das letras
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC)
                    Id.update({i + ' ' + j:np.sum(((roi - np.mean(roi)) - (T - np.mean(T)))**2)})
            return min(Id, key=Id.get)
        elif(method == 'ncc'): #Método Normalized Cross Correlation
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC)
                    Id.update({i + ' ' + j:np.sum(roi*T)/np.sqrt(np.sum(roi**2)*np.sum(T**2))})
            return max(Id, key=Id.get) #A diferença do método NCC é que o valor máximo que indica a melhor similaridade
        elif(method == 'zncc'): #Método NCC com zero-offset
            m_l = np.mean(roi)
            for i in numero:
                for j in naipe:
                    T = template[i][j] #Pega a letra do template
                    T = cv2.resize(T, roi.shape[::-1], interpolation = cv2.INTER_CUBIC)
                    m_T = np.mean(T)
                    Id.update({i + ' ' + j:np.sum((roi-m_l)*(T-m_T))/np.sqrt(np.sum((roi-m_l)**2)*np.sum((T-m_T)**2))})
            return max(Id, key=Id.get)
        else: #Se informou uma string inválida para o método retorna erro
            raise RuntimeError('Invalid Method Selected for alphabet_matching')
            sys.exit()
        return
        
    
    cards = {}
    ret, bin_image = cv2.threshold(cv2.cvtColor(carta_analisa, cv2.COLOR_BGR2GRAY), inf_lim, sup_lim, cv2.THRESH_BINARY_INV) #Criação da imagem binária
    card_contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c_c in card_contours:
        x,y,w,h = cv2.boundingRect(c_c)
        ajust_x, ajust_y = round((x+w)*border_percentage),round((y+h)*border_percentage)
        roi_image = bin_image[y+ajust_y:y+h-ajust_y,x+ajust_x:x+w-ajust_x]
        cv2.imshow('b',roi_image)
        cv2.waitKey(0)
        u,v = roi_image.shape
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (1,u))
        u,v = u//2,v//2
        dilated_symbols = cv2.dilate(roi_image, rect, iterations = 1) #Dilata usando o retângulo
        cv2.imshow('c',dilated_symbols)
        cv2.waitKey(0)
        dilated_contours, __ = cv2.findContours(dilated_symbols, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1,y1,w1,h1 = cv2.boundingRect(dilated_contours[0])
        roi_image[y1:y1+h1,x1:x1+w1] = 0
        x1,y1,w1,h1 = cv2.boundingRect(dilated_contours[-1])
        roi_image[y1:y1+h1,x1:x1+w1] = 0
        symbol_contours, _ = cv2.findContours(roi_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_area = 0
        area_list = []
        for sc in symbol_contours:
            area = cv2.contourArea(sc)
            area_list.append(area)
            if area>big_area:
                big_area = area
                big_sc = sc
        xs,ys,ws,hs = cv2.boundingRect(big_sc)
        roi_symbol = roi_image[ys:ys+hs,xs:xs+ws]
        cv2.imshow('a',roi_symbol)
        cv2.waitKey(0)
        ident = f_template_matching(template, roi_symbol, 'zncc')
        ident = ident.split()
        n = len(symbol_contours)
        for value in area_list:
            if value < big_area*filter_percentage:
                n -= 1
        if n > 1:
            ident[0] = str(n)
        cards.update({str(x) + ' ' + str(y + h//2):ident})
    return cards
        
def f_plot_ident(image,ident):
    key = list(ident.keys())
    for k in key:
        print(k)
        u,v = k.split()
        cv2.putText(image, ident[k][0] + ' de ' + ident[k][1], (int(u),int(v)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0),2, cv2.LINE_AA)
    cv2.imshow('result',image)
    cv2.waitKey(0)
    return
    
cartas = {'A':{'Copas':cv2.imread('Cartas/Nivel_1/A_copas.png'),
               'Espadas':cv2.imread('Cartas/Nivel_1/A_espada.png'),
               'Ouros':cv2.imread('Cartas/Nivel_1/A_ouro.png'),
               'Paus':cv2.imread('Cartas/Nivel_1/A_paus.png')},
          'J':{'Copas':cv2.imread('Cartas/Nivel_1/J_copas.png'),
               'Espadas':cv2.imread('Cartas/Nivel_1/J_espada.png'),
               'Ouros':cv2.imread('Cartas/Nivel_1/J_ouro.png'),
               'Paus':cv2.imread('Cartas/Nivel_1/J_paus.png')},
          'Q':{'Copas':cv2.imread('Cartas/Nivel_1/Q_copas.png'),
               'Espadas':cv2.imread('Cartas/Nivel_1/Q_espada.png'),
               'Ouros':cv2.imread('Cartas/Nivel_1/Q_ouro.png'),
               'Paus':cv2.imread('Cartas/Nivel_1/Q_paus.png')},
          'K':{'Copas':cv2.imread('Cartas/Nivel_1/K_copas.png'),
               'Espadas':cv2.imread('Cartas/Nivel_1/K_espada.png'),
               'Ouros':cv2.imread('Cartas/Nivel_1/K_ouro.png'),
               'Paus':cv2.imread('Cartas/Nivel_1/K_paus.png')}}

cartas_template = f_template_cartas(cartas, 250, 255, 0.02)

carta_analisa = cv2.imread('Cartas/Nivel_2/fig2_n2.png')
# carta_analisa = cv2.imread('Cartas/Nivel_1/Nove_paus.png')

cards = f_ident_cards(carta_analisa, cartas_template, 250, 255, 0.02, 0.8)

f_plot_ident(carta_analisa,cards)

