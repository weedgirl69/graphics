#version 450

layout(location = 0) in vec2 in_texcoord;
layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler2D test_texture_sampler;

void main() { out_color = texture(test_texture_sampler, in_texcoord); }