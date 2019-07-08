#version 450

layout(location = 0) out vec3 out_color;

void main() {
    out_color = vec3(gl_VertexIndex != 0, gl_VertexIndex != 1, gl_VertexIndex != 2);
    gl_Position = vec4(
        2 * gl_VertexIndex / 2 - 1,
        2 * (gl_VertexIndex % 2) - 1,
        0.0,
        1.0
    );
//}