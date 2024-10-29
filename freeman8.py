import cv2 as cv
import numpy as np
import csv
from functools import reduce


directions = [
    (1,0),
    (1,1),
    (0,1),
    (-1,1),
    (-1,0),
    (-1,-1),
    (0,-1),
    (1,-1)
]

def main(border_image, n):
    #Get border image from Moore algorithm
    #border_image = cv.imread('output_images\\boundary_0.png', cv.IMREAD_GRAYSCALE)
    pad_image = np.pad(border_image,1)
    visited = set()
    codes = []
    name1 = f"freeman_conn8\\chains8_{n}.csv"
    name2 = f"freeman_conn8\\chains_firstdiff8_{n}.csv"
    name3 = f"freeman_conn8\\chains_minmag8_{n}.csv"
    with open(name1, mode = 'w', newline ='') as file:
        writer = csv.writer(file)
        #Does not register lone pixels as chains
        for x in range(1, border_image.shape[0]):
            for y in range(1, border_image.shape[1]):
                if pad_image[x, y] == 255 and (x, y) not in visited:
                    #Valid pixel found, add to visited and start a new code
                    visited.add((x, y))
                    code = []
                    current = (x, y) #No need to add origin point as we store all visited points
                    while True: #Search until return to origin or no new neighbors found
                        found = False
                        for d in range(8):
                            neighbor = (current[0] + directions[d][1], current[1] + directions[d][0])
                            if neighbor not in visited and pad_image[neighbor[0],neighbor[1]] == 255:
                                found = True
                                visited.add(neighbor)
                                code.append(d)
                                current = neighbor
                                break
                        if not found:#Chain should be finished at this point
                            break
                    codes.append(code)
        #print(codes)

        codes_clean = [sub_array for sub_array in codes if len(sub_array) >= 10]
        

        for c in codes_clean:
            writer.writerow(c)
    #Apply first difference
    with open(name2, mode = 'w', newline ='') as file:
        writer = csv.writer(file)
        f_diff = []
        for chain in codes_clean:
            number = []
            for digit in range(len(chain)-1):
                number.append((chain[digit+1] - chain[digit] + 8) % 8)
            f_diff.append(number)
        for f in f_diff:
            writer.writerow(f)
            
    
    #Apply min magnitude
    with open(name3, mode = 'w', newline ='') as file:
        writer = csv.writer(file)
        min_mag = []
        for chain in codes_clean:
            if  not all(x == chain[0] for x in chain):
                # Rotate elements until chain[0] is no longer equal to chain[-1]
                while chain[0] == chain[-1]:
                    chain = [chain[-1]] + chain[:-1]
            
            min_val = min(chain)
            digits = []

            position = 0
            while position < len(chain):
                if chain[position] == min_val:
                    digit = [chain[position]]
                    position += 1  # Move to the next position
                    
                    # Collect digits that are the same as the first found minimum
                    while position < len(chain) and chain[position] == digit[0]:
                        digit.append(chain[position])
                        position += 1
                    
                    # Collect subsequent digits until the next occurrence of min_val
                    while position < len(chain) and chain[position] != min_val:
                        digit.append(chain[position])
                        position += 1
                    
                    digits.append(digit)
                else: position += 1
            numerical_values = [int(''.join(map(str, digit))) for digit in digits]

            smalles_position = np.argmin(numerical_values)


            for _ in range(smalles_position):
                digits = digits[1:] + [digits[0]]

            digits_flat = reduce(lambda x, y: x + y, digits)
            min_mag.append(digits_flat)
        for m in min_mag:
            writer.writerow(m)










                    
                

    

if __name__ == '__main__':
    border_image = cv.imread('output_images\\boundary_0.png', cv.IMREAD_GRAYSCALE)
    main(border_image)
