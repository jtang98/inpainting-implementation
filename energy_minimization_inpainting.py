# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as scnd
import scipy.signal as scs
import scipy.misc as misc
import matplotlib.image as mpimg
import scipy.misc
from skimage import io as skio
from skimage import color as skcolor
import imageio
from skimage.color import rgb2lab, rgb2gray, gray2rgb
from patchmatch import patch_match
from scipy import interpolate
import PIL
from index import impaint
from index import find_min_SSD
import time
from skimage.transform import rescale, resize, downscale_local_mean




filename = "oiseau.png"
cote_patch = 7

lambda_texture = 0

x0 = 160
y0 = 580
x1 = 280
y1 = 600

def ImagePyramid(u, L):
    pyramid = []
    pyramid.append(u)
    for i in range(1, L):
        pyramid.append(get_level_down(pyramid[i - 1]))
    return pyramid

def TexturePyramid(T,L):
    pyramid = []
    pyramid.append(T)
    for i in range(1, L):
        pyramid.append(np.array([get_level_down(pyramid[i - 1][0]), get_level_down(pyramid[i-1][1])]))
    return pyramid

def up_sample(phi):

    '''phi_ = phi.repeat(2, axis = 0).repeat(2, axis = 1)'''
    n, m = phi.shape[0], phi.shape[1]
    x = np.array(range(m))
    y = np.array(range(n))
    #xx, yy = np.meshgrid(x, y)
    a = phi[:,:,0]
    b = phi[:,:,1]

    f_0 = interpolate.interp2d(x, y, a, kind='linear')
    f_1 = interpolate.interp2d(x, y, b, kind='linear')
    xnew = np.linspace(0, m, 2*m)
    ynew = np.linspace(0, n, 2*n)
    znew_0 = f_0(xnew, ynew)
    znew_1 = f_1(xnew, ynew)
    return np.moveaxis(np.around(np.array([znew_0, znew_1])), 0, -1).astype(int)

#def up_sample(phi, factor):

def rescale_H(H):
    (n,m) = H.shape
    H_bis = []
    H_ter= []
    for i in range(0, n, 2):
        H_bis.append(H[i])
    (n,m) = np.array(H_bis).shape

    for i in range(n):
        ligne = []
        for j in range(0, m, 2):

            ligne.append(H_bis[i][j])
        H_ter.append(ligne)
    H_ter = np.array(H_ter)
    return H_ter




def Occlusion_Pyramid(H,L):
    pyra = []
    pyra.append(H)
    for i in range(1, L):
        shape = pyra[i-1].shape
        pyra.append(rescale_H(pyra[i-1]))

    return pyra

def get_level_down(u):
    u_ = scnd.gaussian_filter(u, 1.0, truncate=1) #À vérifier, truncate = 0 pour w = 1 avec mu = 1.0
    u__ = scs.decimate(scs.decimate(u_, 2, axis=1), 2, axis=0)
    return u__

def reconstruction_image(u, phi, H, cote_patch):
    (n,m) = u.shape
    v = u.copy()
    for i in range(n):
        for j in range(m):
            if(H[i,j] == 1.0):
                #v[i,j] = u[i+phi[i,j,0], j + phi[i,j,1]]
                #(phi_x, phi_y) = phi[i,j]
                x_min = max(0, i - cote_patch // 2)
                x_max = min(i + cote_patch // 2, n - 1) + 1
                y_min = max(0, j - cote_patch // 2)
                y_max = min(j+ cote_patch // 2, m - 1) + 1
                nb_pixels = (x_max - x_min)*(y_max - y_min)
                sum = 0
                for k in range(x_min, x_max):
                    for l in range(y_min, y_max):
                        xDisp = k + phi[k,l,0]
                        yDisp = l + phi[k,l,1]
                        xDispShift = xDisp - (k-i)
                        yDispShift = yDisp - (l-j)
                        if (xDispShift >= 0 and xDispShift < n and yDispShift >= 0 and yDispShift < m):
                            sum += u[xDispShift, yDispShift]
                        else:
                            pass
                            #nb_pixels -= 1
                if(nb_pixels > 0):
                    v[i,j] = sum/nb_pixels
    return v



def final_reconstruction(u, T, phi, H, cote_patch, color):
    if(color):
        (n,m, o) = u.shape
    else:
        (n,m) = u.shape
    v = u.copy()
    for i in range(n):
        for j in range(m):
            if(H[i,j] == 1.0):
                q = find_min_q(i,j,u, T,phi,cote_patch, color)
                phi_q = phi[q[0], q[1]]
                v[i,j] = u[i+ phi_q[0], j+phi_q[1]]
    return v




def find_min_q(x,y,u, T,phi, cote_patch, color): #trouver q* dans equation 5 de l'article
    i = cote_patch // 2  # indice de ligne
    j = cote_patch // 2  # indice de colonne
    if (color):
        (n, m, o) = u.shape  # dimension de l'image
    else:
        (n, m) = u.shape
    x_min = max(0, x - cote_patch // 2)
    x_max = min(x + cote_patch // 2, n - 1) + 1
    y_min = max(0, y - cote_patch // 2)
    y_max = min(y + cote_patch // 2, m - 1) + 1

    min_indice = (x, y)
    min_dist = np.inf
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):

            xDispShift = x + phi[i,j,0] #verifie que les coordonnes sont potentiellement valides
            yDispShift = y + phi[i,j,1]
            if (xDispShift >= 0 and xDispShift < n and yDispShift >= 0 and yDispShift < m):
                dist = distance_patch_phi(i, j, i + phi[i, j][0], j+ phi[i, j][1], u, T,cote_patch, color)
            else:
                dist = np.inf
            if min_dist > dist:
                min_dist = dist
                min_indice = (i, j)


    return min_indice


def distance_patch_phi(x, y, x1, y1, u, T, a, color):  # A modifier pour la couleur
    # x1,y1 les coordonnées du point centrale en phiq
    x_min = max(0, x - cote_patch // 2)
    x_max = min(x + cote_patch // 2, n - 1) + 1
    y_min = max(0, y - cote_patch // 2)
    y_max = min(y + cote_patch // 2, m - 1) + 1


    x1_min = max(0, x1 - cote_patch // 2)
    x1_max = min(x1 + cote_patch // 2, n - 1) + 1
    y1_min = max(0, y1 - cote_patch // 2)
    y1_max = min(y1 + cote_patch // 2, m - 1) + 1

    if color:
        v = rgb2lab(u)
        phi_p = np.array(v[x - a // 2: x + a // 2 + 1, y - a // 2: y + a // 2 + 1, :])
        phi_q = np.array(v[x1 - a // 2: x1 + a // 2 + 1, y1 - a // 2: y1 + a // 2 + 1, :])
        if (phi_p.shape != phi_q.shape):
            return np.inf
        b = (phi_p - phi_q) ** 2
        z = np.sqrt(b[:, :, 0] + b[:, :, 1] + b[:, :, 2])
        return z.sum()
    else:
        phi_p = np.array(u[x_min:x_max, y_min:y_max])
        phi_p_textureX = np.array(T[0,x_min:x_max, y_min:y_max])
        phi_p_textureY = np.array(T[1,x_min:x_max, y_min:y_max])

        phi_q = np.array(u[x1_min:x1_max, y1_min:y1_max])
        phi_q_textureX = np.array(T[0, x1_min:x1_max, y1_min:y1_max])
        phi_q_textureY = np.array(T[1, x1_min:x1_max, y1_min:y1_max])

        if(phi_p.shape != phi_q.shape):
            return np.inf
        b = (phi_p - phi_q) ** 2 + lambda_texture * ((phi_p_textureX
                                                      - phi_q_textureX)**2 +(phi_p_textureY - phi_q_textureY)**2)

        return b.sum()







def ANNsearch(u, H, patch_size, color = False):
    if(color):
        u_occulte = u * (1 - gray2rgb(H))
    else:
        u_occulte = u * (1 - H)
    print(u.shape)
    if not color:
        (n, m) = u.shape
        u = u[:, :, np.newaxis]
        u_occulte = u_occulte[:, :, np.newaxis]
        cimg0 = np.rollaxis(u, 2).copy().astype('float32')
        cimg1 = np.rollaxis(u_occulte, 2).copy().astype('float32')
        answer = np.moveaxis(patch_match(cimg0, cimg1, patch_size, 10), 0, -1).astype(int)
        answer_ = np.zeros((n, m, 2)).astype(int)
        answer_[:,:,0] = answer[:,:,1].copy()
        answer_[:,:,1] = answer[:,:,0].copy()
        '''for k in range(n):
            for l in range(m):
                if answer[k, l, 0] + k > n or answer[k, l, 1] + l > m:
                    print('coucou')'''
        return answer_

    else:
        (n, m, o) = u.shape
        cimg0 = np.rollaxis(u, 2).copy().astype('float32')
        cimg1 = np.rollaxis(u_occulte, 2).copy().astype('float32')
        answer = np.moveaxis(patch_match(cimg0, cimg1, patch_size, 10), 0, -1).astype(int)
        answer_ = np.zeros((n, m, 2))
        answer_[:,:,0] = answer[:,:,1].copy()
        answer_[:,:,1] = answer[:,:,0].copy()
        return answer_




def get_texture_features(u, mu_radius): #radius OU côté comme cote_patch
    shape = u.shape
    grad = np.array(np.gradient(u))
    T_x = np.zeros(shape)
    T_y = np.zeros(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):

            x_min = max(0, x - mu_radius // 2)
            x_max = min(x + mu_radius // 2, shape[0] - 1) + 1
            y_min = max(0, y - mu_radius // 2)
            y_max = min(y + mu_radius // 2, shape[1] - 1) + 1


            card_v = (x_max - x_min) * (y_max - y_min)
            if(card_v == 0):
                print(card_v)

            T_x[x,y] = (1 / card_v) * np.sum(np.abs(grad[0,x_min : x_max,y_min: y_max]))
            T_y[x,y] = (1 / card_v) * np.sum(np.abs(grad[1,x_min : x_max,y_min: y_max]))

    T = np.array([T_x, T_y])

    return T

def make_random_phi(n,m):
    return np.zeros((n,m))

def get_L(cote_patch, H_0):
    H_L = H_0.copy()
    N_0 = 0
    elem = scnd.generate_binary_structure(2, 2)

    while(H_L.sum() > 0):
        H_L = scnd.binary_erosion(H_L, elem).astype(H_L.dtype)
        N_0 += 1

    L = np.log2(2* N_0/cote_patch)

    return L

if __name__ == '__main__':

    start_time = time.time()

    u_1  = skio.imread(filename)

    DIM = len(u_1.shape)

    if DIM == 2:
        (n, m) = u_1.shape
        color = False
        u_0 = u_1

    else:
        if u_1.shape[2] == 1:
            (n, m, o) = u_1.shape
            color = False
            u_0 = u_1[:, :, 0]
        elif u_1.shape[2] == 3:
            (n,m,o) = u_1.shape
            color = True
            u_0 = u_1
        else:
            (n, m, o) = u_1.shape
            color = True
            u_0 = img = np.copy(u_1[:, :, 0:3])


    H_0 = np.zeros((n,m))
    H_0[x0:x1, y0:y1] = 1.0
    L = int(get_L(cote_patch, H_0)) + 1
    print("L= " + str(L))

    phi = make_random_phi(n,m) #TODO: make random phi

    print(len(u_0.shape))
    e = 1
    k = 0
    u_0 = impaint(H_0, u_0, color, cote_patch)

    T_0 = get_texture_features(u_0, 2**L)

    u_pyramid = np.array(ImagePyramid(u_0,L))
    T_pyramid = TexturePyramid(T_0,L)
    H_pyramid = np.array(Occlusion_Pyramid(H_0, L))

    for l in range(L-1, -1, -1):
        e=1
        k = 0
        u = u_pyramid[l]
        print(u.shape)
        T = T_pyramid[l]
        H = H_pyramid[l]
        phrase = "Etape l=" + str(l)
        print(phrase)
        while e>0.1 and k < 10:
            v = u.copy()
            phi = ANNsearch(u, H, cote_patch, color)
            u = reconstruction_image(u, phi, H, cote_patch)
            T[0] = reconstruction_image(T[0], phi, H, cote_patch)
            T[1] = reconstruction_image(T[1], phi, H, cote_patch)
            e = (abs(u-v)).sum() / (3*H.sum())
            print("e=")
            print(e)
            k += 1
            print(k)

        if(l==0):
            u = final_reconstruction(u, T, phi, H, cote_patch, color)
            imageio.imwrite('outfile_new_algo.jpg', u)
        else:
            phi = up_sample(phi)
            u_pyramid[l-1] = reconstruction_image(u_pyramid[l-1], phi, H_pyramid[l-1], cote_patch)
            T_pyramid[l-1][0] = reconstruction_image(T_pyramid[l-1][0], phi, H_pyramid[l-1], cote_patch)
            T_pyramid[l-1][1] = reconstruction_image(T_pyramid[l-1][0], phi, H_pyramid[l-1], cote_patch)



    end_time = time.time()
    print(end_time - start_time)
