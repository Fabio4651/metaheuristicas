"""def addStreet:
    streetName = input("Street Name:")
    streetVertices = input("Street Vertices:")"""

def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;
 
# Driver Code
x = 1;
y = 1;
circle_x = 0;
circle_y = 1;
rad = 2;
if(isInside(circle_x, circle_y, rad, x, y)):
    print("Inside");
else:
    print("Outside");
 
# This code is contributed
# by mits.    


"""
print("1 - Add Street")
print("2 - Remove Street")
print("3 - Edit Street")
print("4 - Run")
print(" ")
menu = input("Select an option:")
print(" ")


match menu:
    case "1":
        print("You can become a web developer.")

    case "2":
        print("You can become a Data Scientist")

    case "3":
        print("You can become a backend developer")
    
    case "4":
        print("You can become a Blockchain developer")

    case "Java":
        print("You can become a mobile app developer")
    case _:
        print("Please, select a valid option: ")
"""

"""        
print("Enter an array")
# read 9 lines into the list
arr = [input() for _ in range(9)] 

while True:
    # Read a line of coordinates, split into two elements, convert to integers
    x, y = map(int, input("Enter coordinates: ").split(' ', 2))
    # Stop if sentinel in either coordinate
    if -1 in (x, y):
        print("DONE")
        break
    # print the element at the specified coordinates
    print(f"Value = {arr[x][y]}")        """