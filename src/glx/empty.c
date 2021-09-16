#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#ifdef __APPLE__
// iOS build lacks these functions for unknown reasons
GLboolean
glAreTexturesResidentEXT(GLsizei n, const GLuint * textures,
                         GLboolean * residences)
{
    return glAreTexturesResident(n, textures, residences);
}

void glDeleteTexturesEXT(GLsizei n, const GLuint *textures)
{
    glDeleteTextures(n, textures);
}

void glGenTexturesEXT(GLsizei n, GLuint *textures)
{
    glGenTextures(n, textures);
}

GLboolean glIsTextureEXT(GLuint texture)
{
    return glIsTexture(texture);
}
#endif
