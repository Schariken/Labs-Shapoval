from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
from stl import mesh
# Camera position
camera = [0, -4.9, 0]
eye = [0,0,0]
look_x, look_y = 81, 48

# Animation variables
is_animating = False
animation_time = 0.0

mesh_data = mesh.Mesh.from_file("barrel.stl")
display_list = []


def setup_display_list():
    global display_list
    display_list = glGenLists(1)
    glNewList(display_list, GL_COMPILE)
    #glBegin(GL_TRIANGLES)
    for triangle in mesh_data.vectors:
        glColor3f(105/256, 61/256, 16/256)
        draw_triangle(*triangle)
        glColor3f(1, 1, 1)
        draw_wireframe_triangle(*triangle)
    #glEnd()
    glEndList()

def draw_wireframe_triangle(p1, p2, p3):
    glBegin(GL_LINES)

    glVertex3fv(p1)
    glVertex3fv(p2)

    glVertex3fv(p2)
    glVertex3fv(p3)

    glVertex3fv(p3)
    glVertex3fv(p1)

    glEnd()

def draw_triangle(p1, p2, p3):
    glBegin(GL_TRIANGLES)
    glVertex3fv(p1)
    glVertex3fv(p2)
    glVertex3fv(p3)
    glEnd()

def draw_quadrilateral(p1, p2, p3, p4):
    glBegin(GL_QUADS)
    glVertex3fv(p1)
    glVertex3fv(p2)
    glVertex3fv(p3)
    glVertex3fv(p4)

    glEnd()

def animate(_):
    global animation_time, camera, is_animating, eye

    if is_animating:
        # Update camera position based on a simple trajectory
        animation_time += 0.001
        if animation_time >=1 :
            is_animating = not is_animating
            return
        camera[0] = 2* np.sqrt(1 - animation_time**2) *np.cos(3 * np.pi * animation_time)
        camera[1] = 1+ 1 * np.sin(np.pi * animation_time) * (1 - animation_time**2)
        camera[2] = 2* np.sqrt(1 - animation_time**2) *np.sin(5 * np.pi * animation_time)

        eye[0] = -2* np.sqrt(1 - animation_time**2) *np.cos(3 * np.pi * animation_time)
        eye[1] = -1+ 1 * (1 - animation_time**2)
        eye[2] = -2* np.sqrt(1 - animation_time**2) *np.sin(5 * np.pi * animation_time)

        glutPostRedisplay()
        glutTimerFunc(16, animate, 0)  # Schedule the next animation frame

def draw():
    global camera, eye
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (1200 / 800), 0.1, 50.0)  # Adjust window aspect ratio
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    camera_distance = 1
    if not is_animating:
        eye = np.array([
            camera_distance * np.sin(np.radians(look_x))* np.cos(np.radians(look_y)),
            camera_distance * np.cos(np.radians(look_x)),
            camera_distance * np.sin(np.radians(look_x))* np.sin(np.radians(look_y)),
        ])  + camera
    #print(camera, eye)
    gluLookAt(camera[0], camera[1], camera[2], eye[0], eye[1], eye[2], 0.0, 1.0, 0.0)

    # Draw room
    glPushMatrix()
    glColor3f(0.678, 0.847, 0.902);  # Set color to gray
    glScalef(20, 10, 20)
    glutSolidCube(1.0)
    glPopMatrix()

    # Draw wireframe walls
    glPushMatrix()
    glScalef(19.9, 9.9, 19.9)
    glColor3f(1, 1, 1)  # Set color to black
    glutWireCube(1.0)
    glPopMatrix()

    # Draw cube
    glPushMatrix()
    glColor3f(128/256, 0, 128/256)  # Set color to red
    glTranslatef(-1.0, -4.5, -3.0)  # Move cube up
    glutSolidCube(1.0)
    glPopMatrix()

    # Draw teapot
    glPushMatrix()
    glColor3f(255/256, 218/256, 185/256)  
    glTranslatef(1.0, -4.7, 1.0)  
    glRotatef(-150, 0.0, 1.0, 0.0) 
    glutSolidTeapot(0.5)
    glColor3f(1, 1, 1)
    glutWireTeapot(0.5)
    glPopMatrix()

    # Draw sphere in the corner
    glPushMatrix()
    glColor3f(50/256, 15/256, 102/256)  # Set color to green
    glTranslatef(-1, -4.5, -1.3)  # Position at the corner
    glutSolidSphere(0.5, 20, 20)
    glPopMatrix()

    glPushMatrix()
    glColor3f(0.0, 1.0, 0.0)  # Set color to green
    glTranslatef(-3, -5, -3)
    glScalef(0.5, 2, 0.5)

    glColor3f(105/256, 61/256, 16/256)  
    draw_quadrilateral((1,0,1), (1,0,-1), (-1,0,-1), (-1,0,1))#дно

    draw_quadrilateral((1,0,1), (1,0,-1), (1,1,-1), (1,1,1))#боки стовпа
    draw_quadrilateral((1,0,-1), (-1,0,-1), (-1,1,-1), (1,1,-1))
    draw_quadrilateral((-1,0,-1), (-1,0,1), (-1,1,1), (-1,1,-1))
    draw_quadrilateral((1,0,1),(-1,0,1), (-1,1,1), (1,1,1))

    glColor3f(2/256, 47/256, 30/256)  
    draw_quadrilateral((4,1,4), (4,1,-4), (-4,1,-4), (-4,1,4))#дно

    draw_triangle((4,1,4), (4,1,-4), (0,3,0))
    draw_triangle((-4,1,-4), (4,1,-4), (0,3,0))
    draw_triangle((-4,1,4), (4,1,4), (0,3,0))
    draw_triangle((-4,1,-4), (-4,1,4), (0,3,0))

    glColor3f(0/256, 128/256, 0/256)  
    draw_quadrilateral((3,2,3), (3,2,-3), (-3,2,-3), (-3,2,3))#дно

    draw_triangle((3,2,3), (3,2,-3), (0,4,0))
    draw_triangle((-3,2,-3), (3,2,-3), (0,4,0))
    draw_triangle((-3,2,3), (3,2,3), (0,4,0))
    draw_triangle((-3,2,-3), (-3,2,3), (0,4,0))
    
    glColor3f(57/256, 122/256, 76/256)  
    draw_quadrilateral((2,3,2), (2,3,-2), (-2,3,-2), (-2,3,2))#дно

    draw_triangle((2,3,2), (2,3,-2), (0,5,0))
    draw_triangle((-2,3,-2), (2,3,-2), (0,5,0))
    draw_triangle((-2,3,2), (2,3,2), (0,5,0))
    draw_triangle((-2,3,-2), (-2,3,2), (0,5,0))

    glPopMatrix()

    glPushMatrix()
    glColor3f(105/256, 61/256, 16/256)
    glTranslatef(3, -5, 3)
    glRotatef(-90, 0.0, 1.0, 0.0) 
    glRotatef(-90, 1.0, 0.0, 0.0) 
    glCallList(display_list)
    glPopMatrix()

    glutSwapBuffers()

def keyboard(key, x, y):
    global camera, is_animating, animation_time
    global look_x, look_y, look_z
    step = 0.1
    if key == b'w':
        camera[2] -= step
    if key == b's':
        camera[2] += step
    if key == b'a':
        camera[0] -= step
    if key == b'd':
        camera[0] += step
    if key == b'q':
        camera[1] -= step
    if key == b'e':
        camera[1] += step
    if key == b'j':
        look_y += 10*step
    if key == b'l':
        look_y -= 10*step
    if key == b'i':
        look_x += 10*step
    if key == b'k':
        look_x -= 10*step
    if key == b' ':
        # Pressing the space bar toggles animation
        is_animating = not is_animating
        if is_animating:
            animation_time = -1.0
            glutTimerFunc(16, animate, 0)  # Start animation timer
    else:
        if is_animating:
            is_animating = not is_animating
    print(look_x, look_y)

    camera_distance = np.linalg.norm(camera)
    eye = np.array([
            camera_distance * np.sin(np.radians(look_x)),
            camera_distance * np.sin(np.radians(look_y)),
            camera_distance * np.cos(np.radians(look_x))
        ])  +camera
    
    print(eye)

def main():

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1200, 800)
    glutCreateWindow("test")

    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    setup_display_list()

    glutDisplayFunc(draw)
    glutIdleFunc(draw)
    glutKeyboardFunc(keyboard)

    gluPerspective(45, (1200 / 800), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -10)

    glutMainLoop()


if __name__ == "__main__":
    main()
