import moderngl as ModernGL
from PyQt5 import QtCore, QtOpenGL, QtWidgets

import numpy as np

import io
from PIL import Image


class QGLControllerWidget(QtOpenGL.QGLWidget):
    def __init__(self):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        super(QGLControllerWidget, self).__init__(fmt, None)

    def initializeGL(self):
        self.ctx = ModernGL.create_context()

        prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 vert;
                in vec2 pos;
                in vec2 vil;

                out vec4 col;

                uniform float scale;

                void main() {
                    gl_Position = vec4((scale*0.05*vert + pos), 0.0, 1.0);
                    float c = 0.5*(vil.x*vil.x + vil.y*vil.y);
                    col = vec4(c, 1-c, 0, 0.4);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec4 col;
                out vec4 color;
                void main() {
                    color = col;
                }
            ''')

        self.transform = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_pos;
                in vec2 in_vil;

                out vec2 out_pos;
                out vec2 out_vil;
                
                uniform vec2 mass1;
                uniform vec2 mass2;

                float lensq(vec2 v) {
                    return v.x*v.x + v.y*v.y;                
                }

                vec2 gravity(vec2 x) {
                    return -x/pow(lensq(x), 1.5);                
                }

                vec2 field(vec2 pos) {
                    return 1.5*gravity(pos - mass1) + 0.5*gravity(pos - mass2);                
                }

                void main() {
                    vec2 f = field(in_pos);
                    out_pos = in_pos + 0.0002*in_vil;
                    vec2 vil = in_vil + 0.0002*f;
                    out_vil = vec2(clamp(vil.x, -1, 1), clamp(vil.y, -1, 1));
                }
            ''',
            varyings=['out_pos', 'out_vil']
        )
        self.count = 2**18        
        
        self.parm = self.ctx.buffer(np.vstack([np.random.rand(2, self.count).astype(np.float32)*3-1.5, 
                np.random.rand(2, self.count).astype(np.float32)*2-1]).reshape((-1,),order='F').tobytes())
        self.buffer = self.ctx.buffer(reserve=self.parm.size)
        self.vao = self.ctx.vertex_array(prog, [
                (self.ctx.buffer(np.array([-0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5], np.float32).tobytes()), "2f", "vert"),
                (self.parm, "2f 2f/i", "pos", "vil"),
            ], self.ctx.buffer(np.array([0, 1, 2, 1, 2, 3], np.int32).tobytes()))
        self.compute = self.ctx.simple_vertex_array(self.transform, self.parm, "in_pos", "in_vil")
        self.scale = prog['scale']
        self.scale.value = 0.1
        self.sun = self.transform["mass1"]
        self.sun.value = (0.0, 0.0)
        self.moon = self.transform["mass2"]
        self.moon.value = (0.5, 0.0)
        self.i = 0

        self.std = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 vert;
                
                uniform vec2 translate;
                uniform float scale;
                uniform float rotation;             

                void main() {
                    mat2 rot = mat2(cos(rotation), sin(rotation), -sin(rotation), cos(rotation));
                       
                    gl_Position = vec4(translate + rot*scale*vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 color; 
                out vec4 col;
                void main() {
                    col = color;
                }
            ''')
        self.std_translate = self.std["translate"]
        self.std_scale = self.std["scale"]
        self.std_rotation = self.std["rotation"]
        self.std_color = self.std["color"]
        N = 32
        
        self.circle_vao = self.ctx.vertex_array(self.std, [
                (self.ctx.buffer(np.array(sum([[np.cos(phi), np.sin(phi)] for phi in np.linspace(0, 2*np.pi, N)], []), np.float32).tobytes()), "2f", "vert"),
            ], 
            self.ctx.buffer(np.array(sum([[0, i, i+1] for i in range(N-1)], []), np.int32).tobytes()));

        #self.fbo = self.ctx.simple_framebuffer((512, 512), samples=4)
    
    def circle(self, center, size, color):
        self.std_translate.value = center
        self.std_scale = size
        self.std_color = color
        self.circle_vao.render()

    def paintGL(self):
        self.ctx.enable(ModernGL.BLEND)
        self.ctx.viewport = (0, 0, self.width(), self.height())
        #self.scale.value = np.cos(np.pi*self.i/60)
        phi = np.pi*self.i/480
        self.moon.value = (0.5*np.cos(phi), 0.5*np.sin(phi))
        self.sun.value = (-0.2*np.cos(phi), -0.2*np.sin(phi))
        self.circle(self.sun.value, 0.1, (0, 0, 1, 0))
        self.i += 1
        self.ctx.clear(0.9, 0.9, 0.9)
        self.vao.render(instances=self.count)
        #self.fbo.use()
        #self.fbo.clear(0.9, 0.9, 0.9, 1.0)
        #self.vao.render(instances=self.count)
        #self.ctx.finish()
        #Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1).save("img{0:04d}.png".format(self.i))
        #self.ctx.screen.use()
        for _ in range(5):
            self.compute.transform(self.buffer, ModernGL.POINTS, self.count)
            self.ctx.copy_buffer(self.parm, self.buffer)
        self.ctx.finish()
        #Image.frombytes('RGB', self.ctx.screen.size, self.ctx.screen.read(), 'raw', 'RGB', 0, -1).save("img{0:04d}.png".format(self.i))
        self.update()


app = QtWidgets.QApplication([])
window = QGLControllerWidget()
window.resize(512, 512)
window.move(QtWidgets.QDesktopWidget().rect().center() - window.rect().center())
window.show()
app.exec_()
