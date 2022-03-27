from pyexpat.model import XML_CTYPE_MIXED


def tree(x):
    for i in range(0,x):
        space_string = ""
        star_string = ""
        for j in range(0,(x+2-i)):
            if j % 2 == 0:
                space_string = space_string + " "
        for k in range(0,(i+1)):
            star_string = star_string + "*"
        print(space_string + star_string)
    print('{:^10}'.format("*"))
    print('{:^10}'.format("*"))
        

tree(6)


