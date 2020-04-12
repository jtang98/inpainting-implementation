
# coding: utf-8
import numpy as np
import scipy.ndimage as scnd
from skimage import io as skio
from skimage import color as skcolor
import imageio
from skimage.color import rgb2lab
from skimage.color import rgb2yiq
from skimage.color import lab2rgb
from skimage.color import gray2rgb


def select_to_fill(img):
    pass


def get_front(u_mask):
    element_structurant = np.ones((3, 3))
    return u_mask - scnd.morphology.binary_erosion(u_mask, element_structurant)


def calculate_grad_approx(i, j, gradient, old_front_approx):
    if old_front_approx[i - 1, j] == 1.0:
        return gradient[i - 1, j]
    elif old_front_approx[i, j - 1] == 1.0:
        return gradient[i, j - 1]
    elif old_front_approx[i + 1, j] == 1.0:
        return gradient[i + 1, j]
    elif old_front_approx[i, j + 1] == 1.0:
        return gradient[i, j + 1]
    #On passe aux diagonales
    elif old_front_approx[i - 1, j - 1] == 1.0:
        return gradient[i - 1, j - 1]
    elif old_front_approx[i - 1, j + 1] == 1.0:
        return gradient[i - 1, j + 1]
    elif old_front_approx[i + 1, j - 1] == 1.0:
        return gradient[i + 1, j - 1]
    elif old_front_approx[i + 1, j + 1] == 1.0:
        return gradient[i + 1, j + 1]
    return 0


def compute_priorities(front, C, D, cote_patch, mask, img, color):
    (n, m) = C.shape
    P = np.zeros((n, m))
    coordonnees = np.stack((np.nonzero(front)[0], np.nonzero(front)[1]), axis = -1)
    if(color):
        imgYIQ = rgb2yiq(img)
        imgY = imgYIQ[:,:,0]


        raw_gradient = np.gradient(imgY)
        gradient = np.stack((raw_gradient[0], raw_gradient[1]), axis = -1)
    else:
        raw_gradient = np.gradient(img)
        gradient = np.stack((raw_gradient[0], raw_gradient[1]), axis = -1)


    raw_normal_vectors = compute_n(mask)
    normal_vectors = np.stack((raw_normal_vectors[0], raw_normal_vectors[1]), axis = -1)

    element_structurant = np.ones((3, 3))
    #temp_mask = scnd.morphology.binary_dilation(mask, element_structurant).astype('float')
    old_front_approx = abs(mask - scnd.morphology.binary_dilation(mask, element_structurant))

    for psi_p in coordonnees:
        gradient[psi_p[0], psi_p[1]] = calculate_grad_approx(psi_p[0], psi_p[1], gradient, old_front_approx)
        D[psi_p[0], psi_p[1]] = abs(np.vdot(np.array([gradient[psi_p[0], psi_p[1]][1], gradient[psi_p[0], psi_p[1]][0]]), normal_vectors[psi_p[0], psi_p[1]]))

        p_x_min = max(0, psi_p[0] - cote_patch // 2)
        p_x_max = min(psi_p[0] + cote_patch // 2, n - 1) + 1
        p_y_min = max(0, psi_p[1] - cote_patch // 2)
        p_y_max = min(psi_p[1] + cote_patch // 2, m - 1) + 1

        #confidence = np.sum(C[p_x_min: p_x_max, p_y_min: p_y_max]) / ((p_x_max - p_x_min) * (p_y_max - p_y_min))
        norme = (p_x_max - p_x_min) * (p_y_max - p_y_min) # Approximation
        sum = 0
        for i in range (p_x_min, p_x_max):
            for j in range(p_y_min, p_y_max):
                if mask[i, j] == 0.0:
                    sum += C[i,j]

        C[psi_p[0], psi_p[1]] = sum/norme

    for i in range(n):
        for j in range(m):
            if front[i, j] == 1.0:
                P[i, j] = C[i, j] * D[i,j]

    return P, C


def find_most_priority_in_front(P):

    (n, m) = P.shape

    #donner les coordonnées du point de plus haute priorité dans P pour changer son patch
    max_indice = (0,0)
    max_value = 0
    for o in range(n):
        for p in range(m):
            if P[o, p] > max_value: # On prend en compte le front, car les valeurs en dehors sont égales à 0
                max_value = P[o, p]
                max_indice = (o, p)

    return max_indice


def find_min_SSD(psi_p, img, cote_patch, mask, color): #calcul argmin d(phi p , phi q)
    i = cote_patch // 2 #indice de ligne
    j = cote_patch // 2 #indice de colonne
    if(color):
        (n, m, o) = img.shape #dimension de l'image
    else:
        (n,m) = img.shape
    min_indice = (i,j)
    min_dist = np.inf
    while j + cote_patch // 2 <= m - 1: #Verifie que l'on atteigne pas le bas de l'image
        if i + cote_patch // 2 >= n - 1:
            j += 1
            i = cote_patch // 2
        else:
            if mask[i - cote_patch // 2: i+ cote_patch//2 +1, j-cote_patch//2: j+ cote_patch//2 +1].sum() == 0.0:

                dist = distance_SSD(psi_p[0], psi_p[1], i, j, img, cote_patch, mask, color)
                if dist == -1:
                    return -1
                if min_dist > dist:
                    min_dist = dist
                    min_indice = (i,j)
            i +=1

    return min_indice


def distance_SSD(x,y, x1, y1, u, a, mask, color): #A modifier pour la couleur
    #x1,y1 les coordonnées du point centrale en phiq

    try:
        if color:

            phi_p = np.array(u[x-a//2: x+ a//2 +1, y-a//2: y+ a//2 +1, :])
            phi_q = np.array(u[x1-a//2: x1+ a//2 +1, y1-a//2: y1+ a//2 +1, :])
            b = (phi_p - phi_q)**2
            z = np.sqrt(b[:,:,0] + b[:,:,1]+ b[:,:,2])
            z = z * (1 - mask[x-a//2: x+ a//2 +1, y-a//2: y+ a//2 +1])
            return z.sum()
        else:
            phi_p = np.array(u[x - a // 2: x + a // 2 + 1, y - a // 2: y + a // 2 + 1])
            phi_q = np.array(u[x1 - a // 2: x1 + a // 2 + 1, y1 - a // 2: y1 + a // 2 + 1])

            b = (phi_p - phi_q) ** 2
            b = b * (1 - mask[x - a // 2: x + a // 2 + 1, y - a // 2: y + a // 2 + 1])

            return b.sum()

    except ValueError as e:
        print("a")
        print(e)
        print("ee")
        if(color):
            return -1
            imageio.imwrite('outfile_oiseau.jpg', lab2rgb(u))
        else:
            imageio.imwrite('outfile.jpg', u)
            return -1






def copy_patch(psi_p, psi_q, img, mask, cote_patch, color): #Travaille directement sur img/u
    if color:
        (n, m, o) = img.shape
    else:
        (n,m) = img.shape
    p_x_min = max(0, psi_p[0] - cote_patch // 2)
    p_x_max = min(psi_p[0] + cote_patch // 2, n - 1) + 1
    p_y_min = max(0, psi_p[1] - cote_patch // 2)
    p_y_max = min(psi_p[1] + cote_patch //2, m - 1) + 1

    q_x_min = max(0, psi_q[0] - cote_patch // 2)
    q_y_min = max(0, psi_q[1] - cote_patch // 2)

    for x in range(p_x_max - p_x_min):
        for y in range(p_y_max - p_y_min):
            if mask[p_x_min + x, p_y_min + y] != 0:
                if(color):
                    img[p_x_min + x, p_y_min + y,:] = img[q_x_min + x, q_y_min + y,:]
                else:
                    img[p_x_min + x, p_y_min + y] = img[q_x_min + x, q_y_min + y]


def update_C(C, psi_p, mask, cote_patch):
    (n, m) = C.shape
    p_x_min = max(0, psi_p[0] - cote_patch // 2)
    p_x_max = min(psi_p[0] + cote_patch // 2, n - 1) + 1
    p_y_min = max(0, psi_p[1] - cote_patch // 2)
    p_y_max = min(psi_p[1] + cote_patch //2, m - 1) + 1

    for k in range(p_x_max - p_x_min):
        for l in range(p_y_max - p_y_min):
            if mask[k, l] == 1.0:
                C[p_x_min + k, p_y_min + l] = C[psi_p[0], psi_p[1]]
    return C


def update_mask(mask, psi_p, n, m, cote_patch):

    p_x_min = max(0, psi_p[0] - cote_patch // 2)
    p_x_max = min(psi_p[0] + cote_patch // 2, n - 1) + 1
    p_y_min = max(0, psi_p[1] - cote_patch // 2)
    p_y_max = min(psi_p[1] + cote_patch //2, m - 1) + 1

    mask[p_x_min : p_x_max, p_y_min : p_y_max] = 0

    return mask

def compute_n(mask):
    grad_mask = np.gradient(mask)
    norms = np.linalg.norm(grad_mask)
    gradient = [np.where(norms == 0, 0, i / norms) for i in grad_mask]

    return gradient


def impaint(H_0, u_0, color, cote_patch):

    img = u_0
    mask = H_0
    if color:
        img = img * (1-gray2rgb(H_0))
        (n,m,o) = img.shape
    else:
        img = img * (1-H_0)
        (n,m)  = img.shape

    C, D = 1 - mask, np.ones((n, m))

    imageio.imwrite('input_with_mask.jpg', img)



    while True:

        #1a
        front = get_front(mask)

        if not np.any(front):
            imageio.imwrite('outfile_old_algo.jpg',img)

            print("fini")
            return img
            break

        #1b
        P, C = compute_priorities(front, C, D, cote_patch, mask, img, color)

        #2a : find patch
        psi_p = find_most_priority_in_front(P) #Renvoie un tuple de coordonnées (x, y)


        #2b : find examplar
        if(color):
            psi_q = find_min_SSD(psi_p, rgb2lab(img), cote_patch, mask, color)
            if(psi_q==-1):
                return img

        else:
            psi_q = find_min_SSD(psi_p, img, cote_patch, mask, color)
            if(psi_q == -1):
                return  img



        #2c : copy patch
        copy_patch(psi_p, psi_q, img, mask, cote_patch, color)

        print(np.sum(mask))

        #3 : update confidence
        C = update_C(C, psi_p, mask, cote_patch)

        #1a : vrai début
        mask = update_mask(mask, psi_p, n, m, cote_patch)
