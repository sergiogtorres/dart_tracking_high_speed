import glfw
from OpenGL.GL import *

def create_texture():
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex_id

def upload_image(tex_id, image):
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0,
                 GL_RGB, GL_UNSIGNED_BYTE, image)

def draw_textured_quad_vertical_only(index, total, tex_id):
    x_start = -1.0 + (2.0 * index / total)     # from -1 to +1 range
    x_end = -1.0 + (2.0 * (index + 1) / total) # next slice

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x_start, -1)
    glTexCoord2f(1, 1); glVertex2f(x_end, -1)
    glTexCoord2f(1, 0); glVertex2f(x_end,1)
    glTexCoord2f(0, 0); glVertex2f(x_start, 1)
    glEnd()

def draw_textured_quad(index, total, tex_id, layout="horizontal"):
    if layout == "horizontal":
        x_start = -1.0 + (2.0 * index / total)
        x_end = -1.0 + (2.0 * (index + 1) / total)
        y_start = -1.0
        y_end = 1.0
    elif layout == "vertical":
        x_start = -1.0
        x_end = 1.0
        y_start = 1.0 - (2.0 * (index + 1) / total)
        y_end = 1.0 - (2.0 * index / total)
    else:
        raise ValueError("Invalid layout type. Use 'horizontal' or 'vertical'.")

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x_start, y_start)
    glTexCoord2f(1, 1); glVertex2f(x_end, y_start)
    glTexCoord2f(1, 0); glVertex2f(x_end, y_end)
    glTexCoord2f(0, 0); glVertex2f(x_start, y_end)
    glEnd()